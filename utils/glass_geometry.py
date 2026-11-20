"""
The fixed glass patches and the per-pixel ray geometry of Sec. 3.

    P = {Pi_k = (n_k, d_k, S_k)}, n_k^T x + d_k = 0, ||n_k|| = 1     Sec. 1; S_k a convex polygon on the plane
    Pi_t(u) = FirstHit(C_t, u, P),  x_t(u) = HitPoint(...)          Eq. (9)
    r_t(u) = d_t(u) - 2 (n_t(u)^T d_t(u)) n_t(u)                    Eq. (9a)
    c_v = x_t(u_ref) + eps r_t(u_ref),  Q_v = LookAt(r_t(u_ref), h_t)   Eq. (9b)
    cos theta_t(u) = |n_t(u)^T d_t(u)|                              Eq. (10)

"First visible" hit. Eq. (9) intersects the camera ray with the patches only, so a glass patch behind an opaque
object (a pedestrian in front of a shop window) would still be selected and a reflection blended over the object.
`first_hit` therefore also takes an occluder depth from the frozen Phase I field: a hit is dropped when that pass is
opaque there and its depth lies in front of the glass. The occluder pass leaves out the disks that lie on the glass
patches themselves (`on_patches`): they are the glass layer, possibly with leftovers that Phase I painted on it,
never something in front of it. This uses the reconstructed geometry only (no learned mask, nothing trained);
`occlusion=None` gives the literal Eq. (9) (ablation `--ablate_occlusion`).

Patches are dicts {'normal': (3,), 'offset': (), 'vertices': (V, 3)} of tensors after `validate_patches`.
"""

import json

import torch
import torch.nn.functional as F

from utils.point_utils import camera_rays


def validate_patches(records, device='cuda'):
    """Unit normals, finite planar convex polygons. `records` come from JSON (lists) or a previous call."""
    if not records:
        raise ValueError('No glass patches.')
    patches = []
    for index, record in enumerate(records):
        n = torch.as_tensor(record['normal'], dtype=torch.float32, device=device)
        d = torch.as_tensor(record['offset'], dtype=torch.float32, device=device)
        vertices = torch.as_tensor(record['vertices'], dtype=torch.float32, device=device)
        if n.shape != (3,) or not torch.isfinite(n).all() or not torch.isfinite(d) or n.norm() < 1e-8:
            raise ValueError(f'Patch {index}: invalid plane normal {record["normal"]}')
        d, n = d / n.norm(), F.normalize(n, dim=0)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not torch.isfinite(vertices).all():
            raise ValueError(f'Patch {index}: a patch needs at least three finite 3D vertices')
        tolerance = max(float((vertices - vertices.mean(0)).norm(dim=-1).max()), 1.0) * 1e-4
        if (vertices @ n + d).abs().max() > tolerance:
            raise ValueError(f'Patch {index}: vertices do not lie on the plane n^T x + d = 0')
        edges = vertices.roll(-1, 0) - vertices
        area = torch.cross(vertices - vertices[0], vertices.roll(-1, 0) - vertices[0], dim=-1).sum(0).norm()
        if area < 1e-10:
            raise ValueError(f'Patch {index}: zero-area polygon')
        turns = (torch.cross(edges, edges.roll(-1, 0), dim=-1) * n).sum(-1)
        if not ((turns >= -1e-6).all() or (turns <= 1e-6).all()):
            raise ValueError(f'Patch {index}: vertices must form an ordered convex polygon')
        patches.append({'normal': n, 'offset': d, 'vertices': vertices})
    return patches


def patches_to_records(patches):
    return [{'normal': p['normal'].tolist(), 'offset': float(p['offset']), 'vertices': p['vertices'].tolist()} for p in patches]


def load_glass_geometry(path, device='cuda'):
    """(patches, record) from a glass_geometry.json written by extract_glass.py, or a list/dict of patches."""
    with open(path) as f:
        record = json.load(f)
    records = record['patches'] if isinstance(record, dict) else record
    return validate_patches(records, device), record if isinstance(record, dict) else {'patches': records}


def reflect(directions, normals):
    """Eq. (9a): specular reflection of unit directions about unit normals (both (..., 3))."""
    return directions - 2 * (directions * normals).sum(-1, keepdim=True) * normals


def mirror_points(points, normal, offset):
    """Reflection of world points across the plane n^T x + d = 0."""
    return points - 2 * (points @ normal + offset)[..., None] * normal


def _inside_polygon(points, vertices, normal):
    signs = torch.stack([(torch.cross((b - a).expand_as(points), points - a, dim=-1) * normal).sum(-1) for a, b in zip(vertices, vertices.roll(-1, 0))])
    return (signs >= -1e-6).all(0) | (signs <= 1e-6).all(0)


def on_patches(points, patches, margin):
    """(N,) bool: points within `margin` of a patch plane whose projection falls inside the patch polygon."""
    hit = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for patch in patches:
        n, d = patch['normal'], patch['offset']
        signed = points @ n + d
        near = signed.abs() < margin
        if near.any():
            projected = points[near] - signed[near, None] * n
            inside = torch.zeros_like(near)
            inside[near] = _inside_polygon(projected, patch['vertices'], n)
            hit |= inside
    return hit


def first_hit(view, patches, epsilon=1e-4, occlusion=None):
    """
    Eq. (9)-(10) for every pixel of `view`.

    occlusion: None (literal Eq. 9) or {'depth': (1, H, W), 'alpha': (1, H, W), 'tolerance': float, 'alpha_min': float}
    from the frozen direct pass: a hit is occluded where alpha >= alpha_min and depth < hit depth * (1 - tolerance).

    Returns (H, W) maps: patch (int32, -1 = no glass), visible (bool), distance along the unit ray, cosine
    |n^T d|, and (H, W, 3) maps: hits x_t(u), normals n_t(u), real_directions d_t(u) (unit), reflected r_t(u) (unit).
    Off the glass, normals and reflected directions are zero and the hit is the camera center.
    """
    origin, rays = camera_rays(view)  # rays have camera-space z = 1
    lengths = rays.norm(dim=-1)
    direction = rays / lengths[..., None]
    shape = direction.shape[:-1]
    nearest = torch.full(shape, float('inf'), device=direction.device)
    selected = torch.full(shape, -1, device=direction.device, dtype=torch.int32)
    normals = torch.zeros_like(direction)
    for index, patch in enumerate(patches):
        n, d = patch['normal'], patch['offset']
        denominator = direction @ n
        safe = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
        distance = -(origin @ n + d) / safe
        points = origin + distance[..., None] * direction
        hit = (denominator.abs() > 1e-8) & (distance > epsilon) & (distance < nearest) & _inside_polygon(points, patch['vertices'], n)
        nearest = torch.where(hit, distance, nearest)
        selected = torch.where(hit, torch.full_like(selected, index), selected)
        normals = torch.where(hit[..., None], n.expand_as(normals), normals)
    visible = selected >= 0
    occluded = torch.zeros_like(visible)
    if occlusion is not None:
        hit_depth = nearest / lengths  # camera-space z of the hit
        depth, alpha = occlusion['depth'][0], occlusion['alpha'][0]
        occluded = visible & (alpha >= occlusion['alpha_min']) & (depth < hit_depth * (1 - occlusion['tolerance']))
        visible = visible & ~occluded
    selected = torch.where(visible, selected, torch.full_like(selected, -1))
    normals = torch.where(visible[..., None], normals, torch.zeros_like(normals))
    distance = torch.where(visible, nearest, torch.zeros_like(nearest))
    hits = origin + distance[..., None] * direction
    reflected = F.normalize(reflect(direction, normals), dim=-1) * visible[..., None]
    cosine = (normals * direction).sum(-1).abs().clamp(0, 1)
    return {'patch': selected, 'visible': visible, 'occluded': occluded, 'distance': distance, 'cosine': cosine,
            'hits': hits, 'normals': normals, 'real_directions': direction, 'reflected': reflected, 'camera_center': origin}


def look_at(forward, up):
    """
    Camera-to-world rotation of a camera looking along `forward` with `up` roughly up (COLMAP axes: x right,
    y down, z forward). A proper rotation (det +1), as Eq. (9b) needs for a pinhole rasterizer.
    """
    forward = F.normalize(forward, dim=0)
    if torch.abs(up @ forward) > 0.99 * up.norm():  # up parallel to forward: any perpendicular will do
        up = F.one_hot(torch.argmin(forward.abs()), 3).to(forward)
    right = F.normalize(torch.cross(forward, up, dim=0), dim=0)
    down = torch.cross(forward, right, dim=0)
    return torch.stack([right, down, forward], 1)


def dynamic_virtual_camera(view, geometry, patches, patch_index, epsilon=1e-4):
    """
    Eq. (9b): the single virtual camera V_dyn, posed for the region whose rays meet patch `patch_index`.

    The reference pixel u_ref is the pixel of the region closest to the region's mean pixel position. Returns None
    when the patch is not visible in `view`, else {'center': c_v, 'rotation': Q_v (camera-to-world),
    'reference_pixel': (x, y), 'mirrored_center'}. `mirrored_center` is the real camera center reflected across the
    patch plane: every reflected ray of the region passes through it (extended backwards), which is what makes the
    exact reflected-ray render a central projection per patch (gaussian_renderer/reflection.py).
    """
    pixels = (geometry['patch'] == patch_index).nonzero()
    if pixels.numel() == 0:
        return None
    mean = pixels.float().mean(0)
    y, x = pixels[(pixels.float() - mean).square().sum(-1).argmin()]
    forward = geometry['reflected'][y, x]
    up = -view.world_view_transform[:3, :3][:, 1]  # world_view_transform[:3, :3] is camera-to-world; COLMAP y points down
    patch = patches[patch_index]
    return {'center': geometry['hits'][y, x] + epsilon * forward, 'rotation': look_at(forward, up), 'reference_pixel': (int(x), int(y)),
            'mirrored_center': mirror_points(geometry['camera_center'], patch['normal'], patch['offset'])}
