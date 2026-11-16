"""
Phase I renderer: the interface field G_I through the real camera.

The same disks are rasterized twice with the 2DGS rasterizer:

    volume pass   opacities = p * o   ->  the direct image B_t (what is seen through the glass)
    surface pass  opacities = p       ->  the interface geometry, glass included (depth, normal, glass map)
"""

import math

import torch
import torch.nn.functional as F
from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer

from scene.deform_model import apply_deformation
from scene.interface_model import InterfaceGaussianModel
from utils.point_utils import depth_to_normal_sobel


def make_rasterizer(viewpoint_camera, bg_color, active_sh_degree, scaling_modifier=1.0):
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    return GaussianRasterizer(raster_settings=raster_settings)


def unpack_allmap(allmap, viewpoint_camera, depth_ratio):
    """
    Split the rasterizer's geometry buffer.

    Returns alpha (1, H, W), depth (1, H, W), world-space normal (3, H, W), the normal implied by the depth map
    (3, H, W, weighted by alpha) and the distortion map (1, H, W).
    """
    alpha = allmap[1:2]

    normal = allmap[2:5]
    normal = (normal.movedim(0, -1) @ (viewpoint_camera.world_view_transform[:3, :3].T)).movedim(-1, 0)
    normal = F.normalize(normal, dim=0)

    depth_median = torch.nan_to_num(allmap[5:6], 0, 0)
    depth_expected = torch.nan_to_num(allmap[0:1] / alpha, 0, 0)
    depth = depth_expected * (1 - depth_ratio) + depth_ratio * depth_median

    depth_normal = depth_to_normal_sobel(viewpoint_camera, depth.movedim(0, -1)).movedim(-1, 0)
    depth_normal = depth_normal * alpha.detach()

    return alpha, depth, normal, depth_normal, allmap[6:7]


def render_interface(viewpoint_camera, pc: InterfaceGaussianModel, pipe, bg_color: torch.Tensor, offsets=None, scaling_modifier=1.0):
    """
    Render the interface field.

    offsets: optional (d_xyz, d_rotation, d_scaling) from the interface deformation MLP at this camera's time;
    None renders the canonical field. Occupancy and opacity are never deformed.
    Background tensor (bg_color) must be on GPU!
    """
    # Zero tensor used to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    if offsets is None:
        means3D, rotations, scales = pc.get_xyz, pc.get_rotation, pc.get_scaling
    else:
        means3D, rotations, scales = apply_deformation(pc.get_xyz, pc.get_rotation, pc._scaling, *offsets)

    occupancy = pc.get_occupancy
    opacity = pc.get_opacity
    shs = pc.get_features

    rasterizer = make_rasterizer(viewpoint_camera, bg_color, pc.active_sh_degree, scaling_modifier)

    # Volume pass: effective opacity p * o
    direct, _, _, volume_allmap = rasterizer(means3D=means3D, means2D=screenspace_points, shs=shs, extras=None, opacities=occupancy * opacity, scales=scales, rotations=rotations, cov3D_precomp=None)
    volume_alpha, volume_depth, volume_normal, volume_depth_normal, volume_dist = unpack_allmap(volume_allmap, viewpoint_camera, pipe.depth_ratio)

    # Surface pass: occupancy alone; the per-disk opacity is blended as an extra channel
    surface_color, surface_extras, radii, surface_allmap = rasterizer(means3D=means3D, means2D=screenspace_points, shs=shs, extras=opacity, opacities=occupancy, scales=scales, rotations=rotations, cov3D_precomp=None)
    surface_alpha, surface_depth, surface_normal, surface_depth_normal, surface_dist = unpack_allmap(surface_allmap, viewpoint_camera, pipe.depth_ratio)
    surface_opacity = surface_extras[0:1]

    return {
        'direct': direct,  # B_t
        'volume_alpha': volume_alpha,
        'volume_depth': volume_depth,
        'volume_normal': volume_normal,
        'volume_depth_normal': volume_depth_normal,
        'volume_dist': volume_dist,
        'surface_color': surface_color,
        'surface_alpha': surface_alpha,
        'surface_depth': surface_depth,
        'surface_normal': surface_normal,
        'surface_depth_normal': surface_depth_normal,
        'surface_dist': surface_dist,
        'surface_opacity': surface_opacity,
        # Where the first surface is present but lets light through. Supervised by the glass mask M_t.
        'glass_map': (1 - surface_opacity).clamp(0, 1),
        'viewspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
    }
