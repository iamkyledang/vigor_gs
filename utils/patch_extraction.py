"""
Glass patches P from the Phase I interface geometry (Sec. 1: "S_k ... from known geometry or the Phase I interface
geometry"), for scenes whose glass geometry is not known.

1. Samples. On sampled training views, the surface pass of the frozen G_I (occupancy only) gives the depth of the
   first present surface. Pixels annotated as glass where that surface is present but transparent (glass map) are
   unprojected to 3D. The training masks are used here, at extraction time only; test masks never are.
2. Planes. Sequential RANSAC: find the plane with the most inliers, refine it by SVD, remove its inliers, repeat
   until `max_patches` planes or too few points remain.
3. Patches. Coplanar inliers are split into connected components (union-find on a radius graph in plane
   coordinates); each component becomes a convex polygon (its 2D convex hull). Disjoint panes in one wall plane
   therefore become separate patches.
4. Quality. The patches are intersected with the rays of the sampled training views (the occlusion test included)
   and compared with the masks: coverage = |hits & mask| / |mask|, precision = |hits & mask| / |hits|.

Clean images and silhouettes do not determine the depth of transparent glass uniquely (Scope of the framework);
the fit residual and the mask agreement are checks, not ground truth. Convex hulls approximate the boundaries;
glass with holes or curved glass needs known geometry (--glass_geometry).
"""

import numpy as np
import torch

from utils.glass_geometry import first_hit
from utils.point_utils import depths_to_points


def sample_views(cameras, count=12):
    return cameras[::max(1, len(cameras) // count)]


@torch.no_grad()
def glass_samples(views, field, deformation, pipe, render, per_view):
    """World points of the annotated glass pixels where G_I has a present, transparent surface."""
    samples = []
    for cam in views:
        out = render(cam, field, deformation, pipe, surface=True)
        keep = cam.glass_mask[0] & cam.valid_mask[0] & (out['surface_alpha'][0] > 0.3) & (out['glass_map'][0] > 0.1) & (out['surface_depth'][0] > 0)
        points = depths_to_points(cam, out['surface_depth'].permute(1, 2, 0))[keep]
        points = points[torch.isfinite(points).all(-1)]
        if len(points) > per_view:
            points = points[torch.randperm(len(points), device=points.device)[:per_view]]
        if len(points):
            samples.append(points.cpu().numpy())
    return np.concatenate(samples) if samples else np.zeros((0, 3))


def ransac_plane(points, threshold, rng, hypotheses=512):
    """(normal, offset, inlier mask) of the plane with the most inliers, refined by SVD on them."""
    triplets = points[rng.integers(len(points), size=(hypotheses, 3))]
    normals = np.cross(triplets[:, 1] - triplets[:, 0], triplets[:, 2] - triplets[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths[:, None], 1e-12)
    offsets = -(normals * triplets[:, 0]).sum(-1)
    counts = np.zeros(hypotheses, int)
    for start in range(0, hypotheses, 64):  # bounded memory: 64 hypotheses x N points at a time
        block = slice(start, start + 64)
        counts[block] = (np.abs(points @ normals[block].T + offsets[block]) < threshold).sum(0)
    counts[lengths < 1e-8] = 0
    best = counts.argmax()
    inliers = np.abs(points @ normals[best] + offsets[best]) < threshold
    centroid = points[inliers].mean(0)
    _, _, basis = np.linalg.svd(points[inliers] - centroid, full_matrices=False)
    normal = basis[-1]
    offset = -normal @ centroid
    return normal, offset, np.abs(points @ normal + offset) < threshold, basis[:2], centroid


def connected_components(uv, threshold):
    """Labels of the connected components of 2D points linked within a radius (8x the median spacing)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(uv)
    spacing = tree.query(uv, k=2)[0][:, 1]
    radius = max(np.median(spacing) * 8, threshold * 3)
    parent = np.arange(len(uv))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in tree.query_pairs(radius):
        parent[find(a)] = find(b)
    return np.array([find(i) for i in range(len(uv))])


def fit_patches(points, threshold, min_inliers, max_patches, seed=0):
    """Finite convex patches from 3D glass samples (steps 2-3)."""
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    remaining, records = points, []
    while len(records) < max_patches and len(remaining) >= min_inliers:
        normal, offset, inliers, basis, centroid = ransac_plane(remaining, threshold, rng)
        if inliers.sum() < min_inliers:
            break
        plane_points = remaining[inliers]
        remaining = remaining[~inliers]
        uv = (plane_points - centroid) @ basis.T
        labels = connected_components(uv, threshold)
        for label in np.unique(labels):
            group = labels == label
            if group.sum() < min_inliers:
                continue
            try:
                hull = ConvexHull(uv[group])
            except Exception:  # degenerate (collinear) component
                continue
            records.append({'normal': normal.tolist(), 'offset': float(offset), 'vertices': (centroid + uv[group][hull.vertices] @ basis).tolist(),
                            'inliers': int(group.sum()), 'rmse': float(np.sqrt(np.mean((plane_points[group] @ normal + offset) ** 2)))})
            if len(records) >= max_patches:
                break
    return records


def orient_toward_cameras(records, centers):
    """Flip each normal toward the side of the plane where most cameras are (the sign is otherwise arbitrary)."""
    for record in records:
        n, d = np.asarray(record['normal']), record['offset']
        if np.mean(centers @ n + d > 0) < 0.5:
            record['normal'], record['offset'] = (-n).tolist(), -d
    return records


@torch.no_grad()
def patch_quality(views, patches, epsilon, occlusion_for=None):
    """Step 4. `occlusion_for(cam)` returns the occlusion dict of first_hit, or None (literal Eq. 9)."""
    intersection = predicted = annotated = 0
    for cam in views:
        hits = first_hit(cam, patches, epsilon, occlusion_for(cam) if occlusion_for else None)['visible']
        valid, mask = cam.valid_mask[0], cam.glass_mask[0]
        intersection += int((hits & mask & valid).sum())
        predicted += int((hits & valid).sum())
        annotated += int((mask & valid).sum())
    return {'coverage': intersection / max(annotated, 1), 'precision': intersection / max(predicted, 1), 'views': len(views)}


def extract_patches(cameras, field, deformation, pipe, glass, render, extent):
    """Steps 1-3 and the orientation: (records, number of samples). Step 4 needs the patch-aware occluder pass."""
    views = sample_views(cameras)
    points = glass_samples(views, field, deformation, pipe, render, max(200, glass.plane_max_points // len(views)))
    if len(points) < glass.plane_min_inliers:
        raise RuntimeError(f'Phase I produced {len(points)} glass surface samples (need {glass.plane_min_inliers}): '
                           'supply known geometry with --glass_geometry, or train Phase I with the glass prior (--lambda_glass > 0)')
    if len(points) > glass.plane_max_points:
        points = points[np.random.default_rng(0).choice(len(points), glass.plane_max_points, replace=False)]
    records = fit_patches(points, glass.plane_threshold * extent, glass.plane_min_inliers, glass.max_patches)
    if not records:
        raise RuntimeError('No glass plane with enough support was found; supply known geometry with --glass_geometry')
    return orient_toward_cameras(records, np.stack([cam.camera_center.cpu().numpy() for cam in cameras])), len(points)
