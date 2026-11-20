"""
Rendering of the interface field G_I with the 2DGS rasterizer of `diff_surfel_anych` (Eq. 2).

Direct pass: disks rendered with the effective opacity p * o give the direct image B_t, its depth, the rendered
normals N_t and the depth normals N_t^depth of Eq. (3a).

Surface pass (optional): the same disks rendered with the occupancy p alone, carrying o as an extra channel. Where
a disk is present (p high) but lets light through (o low), the surface pass is opaque while the direct pass is
not; `glass_map = surface_alpha - sum_i w_i o_i = sum_i w_i (1 - o_i)` measures exactly that. Phase I uses it only
for the glass-mask prior and for extracting the glass patches; it is never a blending mask (Sec. 1).
"""

import math

import torch
import torch.nn.functional as F

from utils.point_utils import depth_to_normal_sobel


def _settings(view, sh_degree, debug):
    from diff_surfel_anych import GaussianRasterizationSettings

    return GaussianRasterizationSettings(
        image_height=int(view.image_height), image_width=int(view.image_width),
        tanfovx=math.tan(view.FoVx * 0.5), tanfovy=math.tan(view.FoVy * 0.5),
        bg=torch.zeros(3, device='cuda'), scale_modifier=1.0,
        viewmatrix=view.world_view_transform, projmatrix=view.full_proj_transform,
        sh_degree=sh_degree, campos=view.camera_center, prefiltered=False, debug=debug)


def _depth(maps, alpha, depth_ratio):
    expected = torch.nan_to_num(maps[0:1] / alpha.clamp_min(1e-8), 0.0, 0.0)
    return (1 - depth_ratio) * expected + depth_ratio * torch.nan_to_num(maps[5:6], 0.0, 0.0)


def render_interface(view, field, deformation, pipe, surface=False, deforming=True):
    """
    Render G_I(t) through the real camera `view` at `view.time`.

    Returns direct (3, H, W) linear RGB B_t, depth / alpha (1, H, W) of the direct pass, normal / depth_normal
    (3, H, W) world-space N_t and N_t^depth, surface_valid (1, H, W) the pixels where both are defined, and for
    densification viewspace_points, visibility_filter, radii. With `surface`: surface_alpha, surface_depth,
    glass_map.
    """
    from diff_surfel_anych import GaussianRasterizer

    centers, rotations, scales = deformation(field, view.time, enabled=deforming)
    screen = torch.zeros_like(centers, requires_grad=True) + 0
    if screen.requires_grad:
        screen.retain_grad()
    rasterizer = GaussianRasterizer(_settings(view, field.active_sh_degree, pipe.debug))

    def rasterize(opacity, extras=None):
        return rasterizer(means3D=centers, means2D=screen, shs=field.get_features, extras=extras, opacities=opacity,
                          scales=scales, rotations=rotations, cov3D_precomp=None)

    direct, _, radii, maps = rasterize(field.get_effective_opacity)
    alpha = maps[1:2]
    depth = _depth(maps, alpha, pipe.depth_ratio)
    # Camera-space normals to world space (row-vector convention: world_view_transform[:3, :3] is camera-to-world).
    normal = F.normalize(maps[2:5].permute(1, 2, 0) @ view.world_view_transform[:3, :3].T, dim=-1).permute(2, 0, 1)
    depth_normal = depth_to_normal_sobel(view, depth.permute(1, 2, 0)).permute(2, 0, 1)
    # Omega_s of Eq. (3a): covered pixels whose 3 x 3 neighborhood is covered too (the depth normal uses it), with
    # both normals defined.
    covered = alpha.detach() > 0.1
    valid = covered & (F.avg_pool2d(covered.float()[None], 3, stride=1, padding=1)[0] > 0.999)
    valid &= (normal.detach().norm(dim=0, keepdim=True) > 0.5) & (depth_normal.detach().norm(dim=0, keepdim=True) > 0.5)
    out = {'direct': direct, 'depth': depth, 'alpha': alpha, 'normal': normal, 'depth_normal': depth_normal, 'surface_valid': valid,
           'viewspace_points': screen, 'visibility_filter': radii > 0, 'radii': radii}
    if surface:
        _, carried_opacity, surface_radii, surface_maps = rasterize(field.get_occupancy, field.get_opacity)
        surface_alpha = surface_maps[1:2]
        out.update(surface_alpha=surface_alpha, surface_depth=_depth(surface_maps, surface_alpha, pipe.depth_ratio),
                   glass_map=(surface_alpha - carried_opacity[:1]).clamp(0, 1),
                   radii=torch.max(radii, surface_radii), visibility_filter=(radii > 0) | (surface_radii > 0))
    return out
