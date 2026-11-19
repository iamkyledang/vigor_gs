"""Direct 2DGS rendering and Phase I surface diagnostics."""
import math

import torch
import torch.nn.functional as F

from utils.point_utils import depth_to_normal_sobel


def render_interface(view, pc, deformation, pipe, surface=False, deforming=True):
    from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer

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
