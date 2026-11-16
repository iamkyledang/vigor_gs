"""
Glass plane Pi: n^T x + d = 0 with |n| = 1.

The plane is not an input of VIGOR-GS. It is fitted to the interface geometry learned in Phase I and then kept
fixed. By convention the normal points to the camera side, so n^T c + d > 0 for the real cameras.
"""

import json

import torch


def plane_distance(points, normal, offset):
    """Signed distance of (N, 3) points to the plane."""
    return points @ normal + offset


def fit_plane_svd(points):
    """Least-squares plane through (N, 3) points: argmin over |n| = 1 of sum (n^T x + d)^2."""
    centroid = points.mean(dim=0)
    _, _, vh = torch.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    return normal, -(normal @ centroid)


def fit_plane_ransac(points, threshold, iterations=2000, refine_steps=3, seed=0, chunk=256):
    """
    Robust plane fit: RANSAC over point triplets, then least squares on the inliers.

    points: (N, 3). threshold: inlier distance in world units.
    Returns a dict with normal (3,), offset (float), inliers (N,) bool, inlier_ratio and rmse (over inliers).
    """
    num_points = points.shape[0]
    if num_points < 3:
        raise ValueError(f'Need at least 3 points to fit a plane, got {num_points}')
    generator = torch.Generator(device='cpu').manual_seed(seed)

    best_count, best_normal, best_offset = -1, None, None
    for start in range(0, iterations, chunk):
        batch = min(chunk, iterations - start)
        idx = torch.randint(0, num_points, (batch, 3), generator=generator).to(points.device)
        p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
        normals = torch.cross(p1 - p0, p2 - p0, dim=-1)
        lengths = normals.norm(dim=-1, keepdim=True)
        valid = lengths.squeeze(-1) > 1e-12  # skip degenerate (collinear or repeated) triplets
        normals = normals / lengths.clamp_min(1e-12)
        offsets = -(normals * p0).sum(dim=-1)

        counts = ((points @ normals.T + offsets).abs() < threshold).sum(dim=0)
        counts[~valid] = -1
        top = counts.argmax()
        if counts[top] > best_count:
            best_count, best_normal, best_offset = counts[top].item(), normals[top], offsets[top]

    normal, offset = best_normal, best_offset
    inliers = plane_distance(points, normal, offset).abs() < threshold
    for _ in range(refine_steps):
        if inliers.sum() < 3:
            break
        normal, offset = fit_plane_svd(points[inliers])
        inliers = plane_distance(points, normal, offset).abs() < threshold

    residuals = plane_distance(points[inliers], normal, offset)
    return {
        'normal': normal,
        'offset': float(offset),
        'inliers': inliers,
        'inlier_ratio': inliers.float().mean().item(),
        'rmse': residuals.pow(2).mean().sqrt().item() if residuals.numel() > 0 else float('nan'),
    }


def orient_plane(normal, offset, camera_centers):
    """Flip (n, d) if needed so that the cameras are on the positive side."""
    if plane_distance(camera_centers, normal, offset).mean() < 0:
        return -normal, -offset
    return normal, offset


def normalize_plane(normal, offset):
    """Rescale (n, d) so that |n| = 1."""
    norm = normal.norm()
    return normal / norm, offset / norm


def save_plane(path, normal, offset, **info):
    record = {'normal': [float(v) for v in normal], 'offset': float(offset)}
    record.update(info)
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)


def load_plane(path, device='cuda'):
    """Returns (normal (3,) tensor, offset float, the full record)."""
    with open(path) as f:
        record = json.load(f)
    return torch.tensor(record['normal'], dtype=torch.float32, device=device), float(record['offset']), record
