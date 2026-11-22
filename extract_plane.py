"""
VIGOR-GS: extract the glass plane Pi* = (n, d) from the interface field learned in Phase I.

For every training view the surface pass (occupancy only) is rendered, the depth of the first surface is
unprojected to 3D, and the points that fall inside the glass mask are collected. A plane is fitted to the pooled
points with RANSAC + least squares and written to <model_path>/glass_plane.json. Phase II reads it from there and
never changes it.

    python extract_plane.py -m output/<scene>
    python extract_plane.py -m output/<scene> --glass_plane 0 0 1 -2.5     # set the plane by hand instead
"""

import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer.interface import render_interface
from scene.dataset_readers import storePly
from utils.general_utils import safe_state
from utils.plane_utils import fit_plane_ransac, normalize_plane, orient_plane, plane_distance, save_plane
from utils.point_utils import depths_to_points
from utils.vigor_utils import INTERFACE_DIR, PLANE_FILE, load_interface


@torch.no_grad()
def collect_glass_points(cameras, gaussians, deform, background, alpha_threshold=0.5, glass_threshold=0.5, max_points_per_view=2000, generator=None):
    """
    3D points on the first interface surface seen through the glass mask, pooled over the cameras.

    A pixel contributes when the glass mask M_t marks it, the surface pass is solid there (alpha) and the field
    itself reads it as glass (glass map). The median depth is used so that a point lies on a surface rather than
    between the glass and what is behind it.
    """
    median_depth = Namespace(depth_ratio=1.0)
    points = []
    for cam in cameras:
        out = render_interface(cam, gaussians, median_depth, background, offsets=deform.step(gaussians.get_xyz, cam.time))
        keep = cam.gt_transparent_mask.cuda() & (out['surface_alpha'] > alpha_threshold) & (out['glass_map'] > glass_threshold) & (out['surface_depth'] > 0)
        view_points = depths_to_points(cam, out['surface_depth'].movedim(0, -1))[keep.squeeze(0)]
        if view_points.shape[0] > max_points_per_view:
            choice = torch.randperm(view_points.shape[0], generator=generator)[:max_points_per_view]
            view_points = view_points[choice.to(view_points.device)]
        points.append(view_points)
    return torch.cat(points, dim=0) if points else torch.zeros((0, 3), device='cuda')


def extract_plane(dataset, iteration=-1, threshold=-1.0, ransac_iterations=2000, max_points_per_view=2000, manual_plane=None, save_points=True):
    """Fit (or set) the glass plane and write glass_plane.json. Returns the saved record's main entries."""
    scene, gaussians, deform = load_interface(dataset, iteration)
    cameras = scene.getTrainCameras()
    camera_centers = torch.stack([cam.camera_center for cam in cameras])
    plane_path = os.path.join(dataset.model_path, PLANE_FILE)

    if manual_plane is not None:
        normal, offset = normalize_plane(torch.tensor(manual_plane[:3], dtype=torch.float32, device='cuda'), float(manual_plane[3]))
        normal, offset = orient_plane(normal, offset, camera_centers)
        save_plane(plane_path, normal, offset, source='manual')
        print(f'Glass plane set by hand: n = {normal.tolist()}, d = {float(offset):.6f}')
        return {'normal': normal, 'offset': float(offset), 'source': 'manual'}

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    generator = torch.Generator().manual_seed(0)
    points = collect_glass_points(cameras, gaussians, deform, background, max_points_per_view=max_points_per_view, generator=generator)
    if points.shape[0] < 3:
        raise RuntimeError(
            'The interface field has no solid, transparent surface inside the glass masks, so there is nothing to fit a plane to. '
            'Train Phase I longer, check the masks, or set the plane by hand with --glass_plane.'
        )

    if threshold <= 0:
        threshold = 0.01 * scene.cameras_extent  # 1% of the camera rig's radius
    fit = fit_plane_ransac(points, threshold=threshold, iterations=ransac_iterations)
    normal, offset = orient_plane(fit['normal'], fit['offset'], camera_centers)

    camera_distance = plane_distance(camera_centers, normal, offset)
    info = {
        'source': 'ransac',
        'iteration': scene.loaded_iter,
        'num_points': points.shape[0],
        'num_views': len(cameras),
        'threshold': threshold,
        'inlier_ratio': fit['inlier_ratio'],
        'rmse': fit['rmse'],
        'cameras_in_front': (camera_distance > 0).float().mean().item(),
        'mean_camera_distance': camera_distance.mean().item(),
    }
    save_plane(plane_path, normal, offset, **info)

    if save_points:  # inliers green, outliers red; open next to the interface point cloud to check the fit
        colors = np.where(fit['inliers'].cpu().numpy()[:, None], [[0, 200, 0]], [[220, 0, 0]])
        storePly(os.path.join(dataset.model_path, 'glass_points.ply'), points.cpu().numpy(), colors)

    print(f'Glass plane: n = {[round(v, 5) for v in normal.tolist()]}, d = {float(offset):.5f}')
    print(f'  {points.shape[0]} points from {len(cameras)} views, threshold {threshold:.4f}: inlier ratio {fit["inlier_ratio"]:.3f}, RMSE {fit["rmse"]:.5f}')
    print(f'  cameras on the positive side: {info["cameras_in_front"] * 100:.0f}%, mean distance {info["mean_camera_distance"]:.3f}')
    if fit['inlier_ratio'] < 0.5:
        print('[ WARN ] Fewer than half of the glass points lie on the fitted plane. The glass may not be planar, there may be several panes, or the interface geometry is not converged.')
    return {'normal': normal, 'offset': float(offset), **info}


if __name__ == '__main__':
    parser = ArgumentParser(description='VIGOR-GS: glass plane extraction')
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--plane_threshold', default=-1.0, type=float, help='RANSAC inlier distance in world units (default: 1%% of the camera extent)')
    parser.add_argument('--ransac_iterations', default=2000, type=int)
    parser.add_argument('--max_points_per_view', default=2000, type=int)
    parser.add_argument('--glass_plane', nargs=4, type=float, default=None, metavar=('NX', 'NY', 'NZ', 'D'), help='Skip the fit and use this plane n^T x + d = 0')
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser, cfg_subdir=INTERFACE_DIR)

    safe_state(args.quiet)
    # get_combined_args drops options left at None, hence the getattr
    extract_plane(model.extract(args), args.iteration, args.plane_threshold, args.ransac_iterations, args.max_points_per_view, getattr(args, 'glass_plane', None))
