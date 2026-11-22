"""
Warp from the distorted source images of a scene to its undistorted COLMAP images.

The VSGD and Wayve scenes keep only the undistorted PINHOLE model in colmap/sparse/0/; the distorted camera
that produced the COLMAP images is gone. Masks and clean images are made on the source images, so they do not
line up with the COLMAP images. This module recovers the distorted camera from point matches between a source
image and its COLMAP image, then builds the warp that maps one onto the other.

Conventions follow COLMAP (WarpImageBetweenCameras): pixel (i, j) of an image has its center at (i + 0.5, j + 0.5)
in camera coordinates. An undistorted pixel (u', v') becomes the normalized point x = (u' - cx') / fx',
y = (v' - cy') / fy', r = sqrt(x^2 + y^2), and lands on the source pixel u = fx s x + cx, v = fy s y + cy, where

    radial   (COLMAP RADIAL / SIMPLE_RADIAL with fx, fy):  s = 1 + k1 r^2 + k2 r^4
    fisheye  (COLMAP OPENCV_FISHEYE):                      s = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) / r,
                                                           theta = atan(r)

VSGD phone videos are mildly distorted (radial fits to ~0.15 px); the WayveScenes101 cameras are wide-angle
(~120 degrees) and need the fisheye model.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

MODELS = {
    'radial': ('fx', 'fy', 'cx', 'cy', 'k1', 'k2'),
    'fisheye': ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'),
}


def distort_points(points, undistorted, params, model='radial'):
    """
    Map pixel indices of the undistorted image to pixel indices of the source image.

    points: (N, 2) array of (column, row) indices in the undistorted image.
    undistorted: dict with fx, fy, cx, cy of the undistorted PINHOLE camera (COLMAP convention).
    params: the distorted camera, in the order of MODELS[model].
    """
    fx, fy, cx, cy, *k = params
    x = (points[:, 0] + 0.5 - undistorted['cx']) / undistorted['fx']
    y = (points[:, 1] + 0.5 - undistorted['cy']) / undistorted['fy']
    r2 = x * x + y * y
    if model == 'radial':
        s = 1.0 + k[0] * r2 + k[1] * r2 * r2
    elif model == 'fisheye':
        r = np.sqrt(r2)
        theta = np.arctan(r)
        t2 = theta * theta
        theta_d = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
        s = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
    else:
        raise ValueError(f'Unknown distortion model {model!r}; expected one of {list(MODELS)}')
    return np.stack([fx * s * x + cx - 0.5, fy * s * y + cy - 0.5], axis=1)


def initial_params(undistorted, source_size, model='radial', focal_scale=1.0):
    """No distortion, the focal length of the undistorted camera (times focal_scale), principal point at the source center."""
    width, height = source_size
    base = [undistorted['fx'] * focal_scale, undistorted['fy'] * focal_scale, width / 2.0, height / 2.0]
    return np.array(base + [0.0] * (len(MODELS[model]) - 4))


def fit_distortion(undist_points, source_points, undistorted, source_size, model='radial', focal_scales=(1.0, 0.7, 1.4), stages=((20.0, 30.0), (5.0, 8.0), (1.0, 3.0), (1.0, 3.0))):
    """
    Fit the distorted camera so that distort_points(undist_points) lands on source_points.

    The matches may contain outliers and the warp may move points by hundreds of pixels (a fisheye), so the fit runs
    coarse to fine: each stage is a robust (soft L1) fit with the given scale, after which only the matches within the
    given distance are kept for the next stage. Several initial focal lengths are tried and the one keeping the most
    matches within 3 px wins. Returns the parameters and the residual distance (pixels) of every match.
    """
    undist_points = np.asarray(undist_points, dtype=np.float64)
    source_points = np.asarray(source_points, dtype=np.float64)
    if len(undist_points) < 20:
        raise ValueError(f'Only {len(undist_points)} matches; too few to fit a distortion model.')

    def residuals(v, a, b):
        return (distort_points(a, undistorted, v, model) - b).ravel()

    best = None
    for focal_scale in focal_scales:
        v = initial_params(undistorted, source_size, model, focal_scale)
        keep = np.ones(len(undist_points), dtype=bool)
        try:
            for f_scale, trim in stages:
                v = least_squares(residuals, v, args=(undist_points[keep], source_points[keep]), loss='soft_l1', f_scale=f_scale, x_scale='jac').x
                dist = np.linalg.norm(distort_points(undist_points, undistorted, v, model) - source_points, axis=1)
                keep = dist < trim
                if keep.sum() < 20:
                    break
        except (ValueError, np.linalg.LinAlgError):
            continue
        inliers = int((dist < 3.0).sum())
        if best is None or inliers > best[0]:
            best = (inliers, v, dist)
    if best is None or best[0] < 20:
        raise ValueError(f'Only {0 if best is None else best[0]} matches agree with the {model} fit; the source and COLMAP images may not correspond.')
    return best[1], best[2]


def build_remap(undistorted, params, model='radial'):
    """cv2.remap maps (float32, H' x W') that sample the source image on the undistorted COLMAP pixel grid."""
    width, height = undistorted['width'], undistorted['height']
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    src = distort_points(np.stack([cols.ravel(), rows.ravel()], axis=1), undistorted, params, model)
    return src[:, 0].reshape(height, width).astype(np.float32), src[:, 1].reshape(height, width).astype(np.float32)


def warp(image, maps, nearest=False):
    """Warp a source image (H x W [x C] array) to the COLMAP grid. Nearest keeps a mask's label values."""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(image, maps[0], maps[1], interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


def match_points(source_gray, colmap_gray, ratio=0.75, max_features=8000):
    """SIFT matches (ratio test) between a source image and its COLMAP image, as (colmap_points, source_points)."""
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp_s, des_s = sift.detectAndCompute(source_gray, None)
    kp_c, des_c = sift.detectAndCompute(colmap_gray, None)
    if des_s is None or des_c is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_c, des_s, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    pc = np.array([kp_c[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    ps = np.array([kp_s[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    return pc, ps


def local_shifts(a, b, grid=3, half=96):
    """Phase-correlation shifts (pixels) between two same-size gray images on a grid x grid set of patches."""
    height, width = a.shape
    shifts = []
    for fy in np.linspace(0.1, 0.9, grid):
        for fx in np.linspace(0.08, 0.92, grid):
            cy, cx = int(fy * height), int(fx * width)
            y0, y1, x0, x1 = max(cy - half, 0), min(cy + half, height), max(cx - half, 0), min(cx + half, width)
            (dx, dy), _ = cv2.phaseCorrelate(a[y0:y1, x0:x1].astype(np.float32), b[y0:y1, x0:x1].astype(np.float32))
            shifts.append((dx, dy))
    return np.array(shifts)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255.0**2 / mse))
