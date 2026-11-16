"""
Phase II renderer: the reflection field G_R through the virtual camera,

    R_t = R_3D(D_phi_R(G_R, t), C_t^v),      C_t^v = V(C_t, Pi)          (utils/virtual_camera.py)

using the standard 3DGS rasterizer. Where no Gaussian covers a pixel the reflection is black.
"""

import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from scene.deform_model import apply_deformation
from scene.reflection_model import ReflectionGaussianModel


def render_reflection(virtual_camera, pc: ReflectionGaussianModel, pipe, bg_color: torch.Tensor, offsets=None, scaling_modifier=1.0):
    """
    Render the reflection field.

    virtual_camera: the mirrored camera from `reflect_camera` (passing a real camera renders G_R un-mirrored, which
    is only meaningful for the no-virtual-camera ablation).
    offsets: optional (d_xyz, d_rotation, d_scaling) from the reflection deformation MLP; None renders the
    canonical field. Background tensor (bg_color) must be on GPU!
    """
    # Zero tensor used to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(virtual_camera.image_height),
        image_width=int(virtual_camera.image_width),
        tanfovx=math.tan(virtual_camera.FoVx * 0.5),
        tanfovy=math.tan(virtual_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=virtual_camera.world_view_transform,
        projmatrix=virtual_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=virtual_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if offsets is None:
        means3D, rotations, scales = pc.get_xyz, pc.get_rotation, pc.get_scaling
    else:
        means3D, rotations, scales = apply_deformation(pc.get_xyz, pc.get_rotation, pc._scaling, *offsets)

    reflection, radii = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    return {
        'reflection': reflection,  # R_t
        'viewspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
    }
