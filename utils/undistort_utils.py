"""
Warp from the distorted source images of a scene to its undistorted COLMAP images.

The VSGD and Wayve scenes keep only the undistorted PINHOLE model in colmap/sparse/0/; the distorted camera
that COLMAP's image_undistorter started from is gone. Masks and clean images are made on the source images,
so they do not line up with the COLMAP images. This module recovers the distorted camera from point matches
between a source image and its COLMAP image, then builds the same warp the undistorter applied.

Conventions follow COLMAP (WarpImageBetweenCameras): pixel (i, j) of an image has its center at (i + 0.5, j + 0.5)
in camera coordinates. The distorted camera is a radial model with parameters (f, cx, cy, k1, k2):

    x = (u' - cx') / fx',  y = (v' - cy') / fy'            undistorted pixel -> normalized point
    s = 1 + k1 r^2 + k2 r^4,  r^2 = x^2 + y^2
    u = f s x + cx,  v = f s y + cy                        -> distorted (source) pixel

SIMPLE_RADIAL (COLMAP's default) is the case k2 = 0.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ('f', 'cx', 'cy', 'k1', 'k2')


def distort_points(points, undistorted, params):
    """
    Map pixel indices of the undistorted image to pixel indices of the source image.

    points: (N, 2) array of (column, row) indices in the undistorted image.
    undistorted: dict with fx, fy, cx, cy of the undistorted PINHOLE camera (COLMAP convention).
    params: (f, cx, cy, k1, k2) of the distorted camera.
    """
    f, cx, cy, k1, k2 = params
    x = (points[:, 0] + 0.5 - undistorted['cx']) / undistorted['fx']
    y = (points[:, 1] + 0.5 - undistorted['cy']) / undistorted['fy']
    r2 = x * x + y * y
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    return np.stack([f * s * x + cx - 0.5, f * s * y + cy - 0.5], axis=1)


def initial_params(undistorted, source_size):
    """No distortion, same focal length, principal point at the center of the source image."""
    width, height = source_size
    return np.array([0.5 * (undistorted['fx'] + undistorted['fy']), width / 2.0, height / 2.0, 0.0, 0.0])


def fit_distortion(undist_points, source_points, undistorted, source_size, fit_k2=True, trim_px=3.0):
    """
    Fit (f, cx, cy, k1, k2) so that distort_points(undist_points) lands on source_points.

    A robust fit (soft L1), then a refit on the matches within `trim_px` of the first fit. Returns the parameters
    and the residual distance (pixels) of every match under them.
    """
    undist_points = np.asarray(undist_points, dtype=np.float64)
    source_points = np.asarray(source_points, dtype=np.float64)
    x0 = initial_params(undistorted, source_size)
    free = [0, 1, 2, 3, 4] if fit_k2 else [0, 1, 2, 3]

    def unpack(v):
        p = x0.copy()
        p[free] = v
        return p

    def residuals(v, a, b):
        return (distort_points(a, undistorted, unpack(v)) - b).ravel()

    keep = np.ones(len(undist_points), dtype=bool)
    v = x0[free]
    for loss in ('soft_l1', 'linear'):
        result = least_squares(residuals, v, args=(undist_points[keep], source_points[keep]), loss=loss, f_scale=1.0, x_scale='jac')
        v = result.x
        dist = np.linalg.norm(distort_points(undist_points, undistorted, unpack(v)) - source_points, axis=1)
        keep = dist < trim_px
        if keep.sum() < 10:
            raise ValueError(f'Only {keep.sum()} matches agree with the distortion fit; the source and COLMAP images may not correspond.')
    return unpack(v), dist


def build_remap(undistorted, params):
    """cv2.remap maps (float32, H' x W') that sample the source image on the undistorted COLMAP pixel grid."""
    width, height = undistorted['width'], undistorted['height']
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    src = distort_points(np.stack([cols.ravel(), rows.ravel()], axis=1), undistorted, params)
    return src[:, 0].reshape(height, width).astype(np.float32), src[:, 1].reshape(height, width).astype(np.float32)


def warp(image, maps, nearest=False):
    """Warp a source image (H x W [x C] array) to the COLMAP grid. Nearest keeps a mask's label values."""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(image, maps[0], maps[1], interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


def match_points(source_gray, colmap_gray, ratio=0.75, max_shift_px=40.0, max_features=8000):
    """
    SIFT matches between a source image and its COLMAP image, as (colmap_points, source_points) in pixel indices.

    The undistortion moves a point by a few pixels at most, so matches farther apart than `max_shift_px` (after
    scaling for the size change) are dropped as outliers.
    """
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp_s, des_s = sift.detectAndCompute(source_gray, None)
    kp_c, des_c = sift.detectAndCompute(colmap_gray, None)
    if des_s is None or des_c is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_c, des_s, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    pc = np.array([kp_c[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    ps = np.array([kp_s[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    scale = np.array([source_gray.shape[1] / colmap_gray.shape[1], source_gray.shape[0] / colmap_gray.shape[0]])
    near = np.linalg.norm(pc * scale - ps, axis=1) < max_shift_px
    return pc[near], ps[near]


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
