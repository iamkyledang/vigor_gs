"""Sparse tile routing and one shared reflected-ray Gaussian image."""
import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation
from utils.sh_utils import eval_sh


@torch.no_grad()
def tile_candidates(view,means,scales,patches,geometry,pipe):
    """Conservative bounding-sphere bins; no pixel-by-Gaussian dense tensor."""
    W,H,tile=view.image_width,view.image_height,pipe.tile_size
    nx,ny=(W+tile-1)//tile,(H+tile-1)//tile
    count=nx*ny
    N=len(means)
    radius=pipe.ray_cutoff*scales.max(-1).values
    ids=[];tiles=[];depths=[]
    yy,xx=torch.meshgrid(torch.arange(H,device=means.device),torch.arange(W,device=means.device),indexing='ij')
    pixel_tiles=(yy//tile)*nx+xx//tile
    for k,patch in enumerate(patches):
        normal,offset=patch['normal'],patch['offset']
        mirror=means-2*(means@normal+offset)[:,None]*normal
        camera=mirror@view.world_view_transform[:3,:3]+view.world_view_transform[3,:3]
        x,y,z=camera.unbind(-1)
        near=(z-radius).clamp_min(1e-5)
        far=(z+radius).clamp_min(1e-5)
        def bounds(center,focal,principal,size):
            a=(center-radius)/near; b=(center-radius)/far
            c=(center+radius)/near; d=(center+radius)/far
            low=torch.minimum(a,b)*focal+principal-0.5
            high=torch.maximum(c,d)*focal+principal-0.5
            crosses=z-radius<=1e-5
            low=torch.where(crosses,torch.zeros_like(low),low)
            high=torch.where(crosses,torch.full_like(high,size-1),high)
            return torch.floor(low/tile).long(),torch.floor(high/tile).long()+1
        x0,x1=bounds(x,view.fx,view.cx,W); y0,y1=bounds(y,view.fy,view.cy,H)
        x0=x0.clamp(0,nx); x1=x1.clamp(0,nx); y0=y0.clamp(0,ny); y1=y1.clamp(0,ny)
        valid=(z+radius>view.znear)&torch.isfinite(camera).all(-1)
        width=(x1-x0).clamp_min(0);height=(y1-y0).clamp_min(0)
        sizes=width*height*valid
        total=int(sizes.sum())
        if sum(len(a) for a in ids)+total>pipe.max_tile_entries:
            raise RuntimeError(f'Reflection bin budget exceeded ({total} entries for patch {k}). Reduce points/scales or increase --max_tile_entries on a larger GPU.')
        point=torch.repeat_interleave(torch.arange(N,device=means.device),sizes)
        if point.numel()==0: continue
        starts=torch.cumsum(sizes,0)-sizes
        offset_in=torch.arange(total,device=means.device)-starts[point]
        tile_id=(y0[point]+offset_in//width[point])*nx+x0[point]+offset_in%width[point]
        occupancy=torch.zeros(count,device=means.device,dtype=torch.int32)
        region=geometry['patches']==k
        occupancy.scatter_add_(0,pixel_tiles[region],torch.ones_like(pixel_tiles[region],dtype=torch.int32))
        keep=occupancy[tile_id]>0
        point,tile_id=point[keep],tile_id[keep]
        ids.append(point+k*N); tiles.append(tile_id); depths.append(z[point].clamp_min(1e-8))
    if not ids:
        return torch.empty(0,device=means.device,dtype=torch.int64),torch.zeros(count+1,device=means.device,dtype=torch.int64)
    ids,tiles,depths=torch.cat(ids),torch.cat(tiles),torch.cat(depths)
    # Float bit order is monotonic for positive camera depth; one 64-bit sort.
    keys=(tiles.long()<<32)|depths.contiguous().view(torch.int32).long()
    order=torch.argsort(keys)
    ranges=torch.cat([tiles.new_zeros(1),torch.bincount(tiles,minlength=count).cumsum(0)])
    return ids[order].contiguous(),ranges.contiguous()


def render_reflection(view,pc,deformation,patches,geometry,pipe,deforming=True):
    from diff_surfel_anych import render_reflected_rays

    means,rotations,scales=deformation(pc,view.time,enabled=deforming)
    if means.requires_grad: means.retain_grad()
    inverse_axes=build_rotation(rotations)/scales[:,None,:].clamp_min(1e-7)
    colors=[]
    sh=pc.get_features.transpose(1,2)
    for patch in patches:
        n,d=patch['normal'],patch['offset']
        center=view.camera_center-2*(view.camera_center@n+d)*n
        direction=F.normalize(means-center,dim=-1)
        # Linear radiance is nonnegative but may exceed one before composition.
        colors.append((eval_sh(pc.active_sh_degree,sh,direction)+0.5).clamp_min(0))
    colors=torch.cat(colors)
    entries,ranges=tile_candidates(view,means,scales,patches,geometry,pipe)
    reflection=render_reflected_rays(geometry['origins'].reshape(-1,3),geometry['directions'].reshape(-1,3),
        geometry['patches'].flatten(),means,inverse_axes,pc.get_opacity,colors,entries,ranges,
        view.image_width,view.image_height,pipe.tile_size,pipe.ray_cutoff)
    visible=torch.zeros(len(means),device=means.device,dtype=torch.bool)
    if entries.numel(): visible[entries%len(means)]=True
    return {'reflection':reflection,'deformed_means':means,'visibility_filter':visible,'tile_entries':len(entries)}
