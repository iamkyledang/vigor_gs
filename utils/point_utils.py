"""Calibrated camera rays, depth unprojection and depth normals."""
import math
import torch
import torch.nn.functional as F


def camera_rays(view):
    device = view.world_view_transform.device
    y,x = torch.meshgrid(torch.arange(view.image_height,device=device),torch.arange(view.image_width,device=device),indexing='ij')
    fx=getattr(view,'fx',view.image_width/(2*math.tan(view.FoVx/2)))
    fy=getattr(view,'fy',view.image_height/(2*math.tan(view.FoVy/2)))
    cx,cy=getattr(view,'cx',view.image_width/2),getattr(view,'cy',view.image_height/2)
    local = torch.stack([(x+0.5-cx)/fx,(y+0.5-cy)/fy,torch.ones_like(x)],-1).float()
    c2w = view.world_view_transform.T.inverse()
    return c2w[:3,3],local @ c2w[:3,:3].T


def depths_to_points(view, depthmap):
    origin,direction=camera_rays(view)
    return depthmap*direction+origin


def depth_to_normal_sobel(view, depth):
    points=depths_to_points(view,depth)
    if points.ndim==4:
        return torch.stack([depth_to_normal_sobel(view,d) for d in depth])
    dx=points[1:-1,2:]-points[1:-1,:-2]
    dy=points[2:,1:-1]-points[:-2,1:-1]
    normal=F.normalize(torch.cross(dy,dx,dim=-1),dim=-1)
    return F.pad(normal.permute(2,0,1),(1,1,1,1)).permute(1,2,0)
