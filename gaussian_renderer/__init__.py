"""VIGOR-GS direct 2DGS and one routed reflected-ray 3D Gaussian image."""
import math
import sys
from pathlib import Path

# Use this checkout's rebuilt extension, without replacing the RT-Splatting environment.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'submodules/diff-surfel-anych'))
import torch
import torch.nn.functional as F
from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer, render_reflected_rays
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation
from utils.point_utils import camera_rays, depth_to_normal_sobel, first_hit, dynamic_virtual_camera
from utils.sh_utils import eval_sh


def render_interface(view, pc, deformation, pipe, surface=False, deforming=True):
    means,rotation,scales=deformation(pc,view.time,enabled=deforming)
    screen=torch.zeros_like(means,requires_grad=True)+0
    if screen.requires_grad: screen.retain_grad()
    settings=GaussianRasterizationSettings(view.image_height,view.image_width,
        math.tan(view.FoVx/2),math.tan(view.FoVy/2),means.new_zeros(3),1.,
        view.world_view_transform,view.full_proj_transform,pc.active_sh_degree,view.camera_center,False,pipe.debug)
    rasterizer=GaussianRasterizer(settings)
    def raster(opacity,extras=None):
        return rasterizer(means3D=means,means2D=screen,shs=pc.get_features,extras=extras,
            opacities=opacity,scales=scales,rotations=rotation,cov3D_precomp=None)
    direct,_,radii,maps=raster(pc.get_occupancy*pc.get_opacity)
    alpha=maps[1:2]
    depth=torch.nan_to_num(maps[0:1]/alpha.clamp_min(1e-8))
    depth=(1-pipe.depth_ratio)*depth+pipe.depth_ratio*torch.nan_to_num(maps[5:6])
    normal=F.normalize((maps[2:5].permute(1,2,0)@view.world_view_transform[:3,:3].T),dim=-1).permute(2,0,1)
    depth_normal=depth_to_normal_sobel(view,depth.permute(1,2,0)).permute(2,0,1)
    # Exclude missing neighbors and image borders from the surface objective.
    valid=alpha.detach()>0.1
    valid &= (F.avg_pool2d(valid.float()[None],3,stride=1,padding=1)[0]>0.999)
    valid &= (normal.detach().norm(dim=0,keepdim=True)>0.5)&(depth_normal.detach().norm(dim=0,keepdim=True)>0.5)
    out={'direct':direct,'final_rendering':direct,'normal':normal,'depth_normal':depth_normal,
         'surface_valid':valid,'alpha':alpha,'depth':depth,'viewspace_points':screen,
         'visibility_filter':radii>0,'radii':radii}
    if surface:
        _,optical,radii,surface_maps=raster(pc.get_occupancy,pc.get_opacity)
        sa=surface_maps[1:2]
        out.update(surface_alpha=sa,surface_depth=torch.nan_to_num(surface_maps[0:1]/sa.clamp_min(1e-8)),
                   glass_map=(sa-optical[:1]).clamp(0,1),radii=radii,visibility_filter=radii>0)
    return out


def ray_geometry(view,patches,args,pipe):
    geometry=first_hit(view,patches,pipe.ray_epsilon)
    F0=((args.eta_air-args.eta_glass)/(args.eta_air+args.eta_glass))**2
    geometry['fresnel']=F0+(1-F0)*(1-geometry['cosine'])**5
    geometry['fresnel']*=geometry['visible']
    # A single function/state is reused for every encountered patch.
    state=None
    for k in range(len(patches)):
        pose=dynamic_virtual_camera(view,geometry,k)
        if pose is not None: state=pose
    geometry['virtual_camera_state']=state
    return geometry


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


def render_composite(view,interface,interface_deform,reflection,reflection_deform,patches,optical,args,pipe,
                     direct=None,geometry=None,deforming=True):
    if direct is None:
        with torch.no_grad(): direct=render_interface(view,interface,interface_deform,pipe)['direct']
    if geometry is None: geometry=ray_geometry(view,patches,args,pipe)
    out=render_reflection(view,reflection,reflection_deform,patches,geometry,pipe,deforming)
    H,W=view.image_height,view.image_width
    y,x=torch.meshgrid(torch.linspace(-1,1,H,device=direct.device),torch.linspace(-1,1,W,device=direct.device),indexing='ij')
    correction=direct.new_zeros(H,W) if args.pure_fresnel else optical(torch.stack([x,y],-1),view.time,geometry['cosine'])
    correction=correction*geometry['visible']
    alpha=(geometry['fresnel']+correction).clamp(0,1)*geometry['visible']
    composite=(1-alpha[None])*direct+alpha[None]*out['reflection']
    out.update(composite=composite,final_rendering=composite,direct=direct,alpha=alpha[None],h=correction,
               geometry=geometry)
    return out


render=render_interface
