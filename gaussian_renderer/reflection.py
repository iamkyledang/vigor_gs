"""
The reflection image R_t of Eq. (5): the shared field G_R(t) queried along the per-pixel rays of Sec. 3 ("Render").

Rays. For a glass pixel u with hit x_t(u) on patch Pi_t(u) and reflected direction r_t(u) (Eq. 9a), the physically
correct ray is x_t(u) + s r_t(u), s >= 0. These rays are one image in the real camera's pixel grid, whatever patch
each pixel sees. Three ray modes:

    exact       x_t(u) + eps r_t(u), direction r_t(u)                  the framework's renderer (default)
    pinhole     c_v + s r_t(u), c_v the Eq. (9b) pose of the region's   "using that pose alone in an ordinary pinhole
                virtual camera V_dyn                                    rasterizer is only an approximation"
    unreflected x_t(u) + eps d_t(u), direction d_t(u)                  ablation without a virtual camera: G_R is seen
                                                                        straight through the glass

Renderer. `diff_surfel_anych.render_reflected_rays` (CUDA) evaluates every ellipsoid at its closest approach to
the ray in the Gaussian metric, rejects closest points behind the ray start (s <= 0), and composites front to back
in the order of the candidate lists; its backward reaches positions, inverse axes, opacity and colors.

Candidates. Within one patch, all exact reflected rays pass through the mirrored camera center c'_k, so the pixel
whose ray passes a point P is the real-camera projection of mirror_k(P). `tile_candidates` uses this to bin
Gaussians into screen tiles with a conservative bounding sphere, separately per patch, keeping only tiles where
the patch is visible; entries are sorted by depth along the mirrored view. For the pinhole mode the rays are the
exact ones translated by c_v - c'_k, for the unreflected mode the real camera's own rays (no mirroring).

Colors use the SH of each Gaussian in the direction the ray arrives from (mean - ray center of the patch), so one
Gaussian seen in two patches gets two view-dependent colors.
"""

import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation
from utils.glass_geometry import dynamic_virtual_camera, mirror_points
from utils.sh_utils import eval_sh

RAY_MODES = ('exact', 'pinhole', 'unreflected')


def ray_bundle(view, geometry, patches, mode='exact', epsilon=1e-4):
    """
    Per-pixel rays of the reflection render and, per patch, how to bin and color Gaussians for it.

    Returns origins / directions (H, W, 3), patch (H, W) int32, and per patch k: 'centers' (K, 3) the point the
    patch's rays come from, 'shifts' (K, 3) and 'mirror' (bool) for binning (pixel of P = projection of
    mirror_k(P + shift_k)), plus 'poses' the Eq. (9b) virtual camera of each visible patch (None otherwise).
    """
    if mode not in RAY_MODES:
        raise ValueError(f'ray mode {mode!r}, expected one of {RAY_MODES}')
    visible = geometry['visible'][..., None]
    poses = [dynamic_virtual_camera(view, geometry, patches, k, epsilon) for k in range(len(patches))]
    camera = geometry['camera_center']
    if mode == 'unreflected':
        directions = geometry['real_directions'] * visible
        origins = geometry['hits'] + epsilon * directions
        centers = camera.expand(len(patches), 3)
        return {'origins': origins, 'directions': directions, 'patch': geometry['patch'], 'centers': centers,
                'shifts': torch.zeros_like(centers), 'mirror': False, 'poses': poses}
    directions = geometry['reflected']
    mirrored = torch.stack([mirror_points(camera, p['normal'], p['offset']) for p in patches])
    if mode == 'exact':
        origins = geometry['hits'] + epsilon * directions
        return {'origins': origins, 'directions': directions, 'patch': geometry['patch'], 'centers': mirrored,
                'shifts': torch.zeros_like(mirrored), 'mirror': True, 'poses': poses}
    # pinhole: one origin per region, the virtual camera center c_v.
    centers = torch.stack([pose['center'] if pose is not None else mirrored[k] for k, pose in enumerate(poses)])
    patch = geometry['patch'].long()
    origins = torch.where(visible, centers[patch.clamp_min(0)], geometry['hits'])
    return {'origins': origins, 'directions': directions, 'patch': geometry['patch'], 'centers': centers,
            'shifts': mirrored - centers, 'mirror': True, 'poses': poses}


@torch.no_grad()
def tile_candidates(view, means, scales, bundle, patches, pipe):
    """
    Sparse (Gaussian, tile) candidate lists of every patch, without a dense pixel x Gaussian tensor.

    Returns entries (int64, k * N + Gaussian index, sorted by tile then depth) and ranges (tiles + 1): the entries of
    tile i are entries[ranges[i]:ranges[i + 1]]. Raises when more than `pipe.max_tile_entries` pairs are needed.
    """
    W, H, tile = view.image_width, view.image_height, pipe.tile_size
    nx, ny = (W + tile - 1) // tile, (H + tile - 1) // tile
    count, N, device = nx * ny, len(means), means.device
    radius = pipe.ray_cutoff * scales.max(-1).values
    rows, cols = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixel_tiles = (rows // tile) * nx + cols // tile
    ids, tiles, depths, total_entries = [], [], [], 0
    for k, patch in enumerate(patches):
        region = bundle['patch'] == k
        if not region.any() or N == 0:
            continue
        mapped = means + bundle['shifts'][k]
        if bundle['mirror']:
            mapped = mirror_points(mapped, patch['normal'], patch['offset'])
        camera = mapped @ view.world_view_transform[:3, :3] + view.world_view_transform[3, :3]
        x, y, z = camera.unbind(-1)
        near, far = (z - radius).clamp_min(1e-5), (z + radius).clamp_min(1e-5)
        crosses = z - radius <= 1e-5  # the sphere reaches the camera plane: it may cover the whole image

        def bounds(center, focal, principal, size):
            low = torch.minimum((center - radius) / near, (center - radius) / far) * focal + principal - 0.5
            high = torch.maximum((center + radius) / near, (center + radius) / far) * focal + principal - 0.5
            low = torch.where(crosses, torch.zeros_like(low), low)
            high = torch.where(crosses, torch.full_like(high, size - 1), high)
            return torch.floor(low / tile).long(), torch.floor(high / tile).long() + 1

        x0, x1 = bounds(x, view.fx, view.cx, W)
        y0, y1 = bounds(y, view.fy, view.cy, H)
        x0, x1, y0, y1 = x0.clamp(0, nx), x1.clamp(0, nx), y0.clamp(0, ny), y1.clamp(0, ny)
        keep = (z + radius > view.znear) & torch.isfinite(camera).all(-1)
        width, height = (x1 - x0).clamp_min(0), (y1 - y0).clamp_min(0)
        sizes = width * height * keep
        total = int(sizes.sum())
        total_entries += total
        if total_entries > pipe.max_tile_entries:
            raise RuntimeError(f'Reflection tile budget exceeded ({total_entries} entries at patch {k}): reduce points/scales or raise --max_tile_entries')
        if total == 0:
            continue
        point = torch.repeat_interleave(torch.arange(N, device=device), sizes)
        offset = torch.arange(total, device=device) - (torch.cumsum(sizes, 0) - sizes)[point]
        tile_id = (y0[point] + offset // width[point]) * nx + x0[point] + offset % width[point]
        occupied = torch.zeros(count, device=device, dtype=torch.bool)
        occupied[pixel_tiles[region]] = True
        used = occupied[tile_id]
        point, tile_id = point[used], tile_id[used]
        ids.append(point + k * N)
        tiles.append(tile_id)
        depths.append(z[point].clamp_min(1e-8))
    if not ids:
        return torch.empty(0, device=device, dtype=torch.int64), torch.zeros(count + 1, device=device, dtype=torch.int64)
    ids, tiles, depths = torch.cat(ids), torch.cat(tiles), torch.cat(depths)
    # Positive float32 bits sort like the floats: one 64-bit key (tile, depth).
    order = torch.argsort((tiles.long() << 32) | depths.contiguous().view(torch.int32).long())
    ranges = torch.cat([tiles.new_zeros(1), torch.bincount(tiles, minlength=count).cumsum(0)])
    return ids[order].contiguous(), ranges.contiguous()


def render_reflection(view, field, deformation, bundle, patches, pipe, deforming=True):
    """R_t (3, H, W) in linear RGB (nonnegative, may exceed 1), plus what densification needs."""
    from diff_surfel_anych import render_reflected_rays

    means, rotations, scales = deformation(field, view.time, enabled=deforming)
    if means.requires_grad:
        means.retain_grad()
    inverse_axes = build_rotation(rotations) / scales[:, None, :].clamp_min(1e-7)
    sh = field.get_features.transpose(1, 2)
    colors = torch.cat([(eval_sh(field.active_sh_degree, sh, F.normalize(means - center, dim=-1)) + 0.5).clamp_min(0) for center in bundle['centers']])
    entries, ranges = tile_candidates(view, means, scales, bundle, patches, pipe)
    reflection = render_reflected_rays(bundle['origins'].reshape(-1, 3), bundle['directions'].reshape(-1, 3), bundle['patch'].flatten(),
                                       means, inverse_axes, field.get_opacity, colors, entries, ranges,
                                       view.image_width, view.image_height, pipe.tile_size, pipe.ray_cutoff)
    visible = torch.zeros(len(means), device=means.device, dtype=torch.bool)
    if entries.numel():
        visible[entries % len(means)] = True
    return {'reflection': reflection, 'deformed_means': means, 'visibility_filter': visible, 'tile_entries': int(entries.numel())}
