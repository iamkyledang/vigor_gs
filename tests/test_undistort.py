"""Module 15: recovering the source -> COLMAP undistortion warp. Synthetic data only, no GPU or dataset needed."""

import cv2
import numpy as np
import pytest

from utils.undistort_utils import build_remap, distort_points, fit_distortion, initial_params, local_shifts, match_points, warp

# A camera like VSGD 0758: 1920x1080 source, undistorted by COLMAP to 1927x1080.
UNDIST = {'fx': 1208.34, 'fy': 1208.34, 'cx': 963.5, 'cy': 540.0, 'width': 1927, 'height': 1080}
SOURCE_SIZE = (1920, 1080)
TRUE = np.array([1208.34, 960.0, 540.0, -0.0073, 0.0002])


def test_no_distortion_maps_the_principal_point_onto_itself():
    params = initial_params(UNDIST, SOURCE_SIZE)
    assert np.allclose(params, [1208.34, 960.0, 540.0, 0.0, 0.0])
    center = np.array([[UNDIST['cx'] - 0.5, UNDIST['cy'] - 0.5]])
    assert np.allclose(distort_points(center, UNDIST, params), [[959.5, 539.5]])


def test_fit_recovers_the_distortion_despite_outliers():
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [UNDIST['width'] - 1, UNDIST['height'] - 1], size=(3000, 2))
    src = distort_points(pts, UNDIST, TRUE) + rng.normal(0, 0.2, size=pts.shape)
    src[:200] += rng.uniform(-30, 30, size=(200, 2))  # wrong matches

    params, dist = fit_distortion(pts, src, UNDIST, SOURCE_SIZE)
    grid = np.stack(np.meshgrid(np.linspace(0, 1926, 20), np.linspace(0, 1079, 12)), -1).reshape(-1, 2)
    assert np.abs(distort_points(grid, UNDIST, params) - distort_points(grid, UNDIST, TRUE)).max() < 0.1
    assert np.median(dist[200:]) < 0.3


def test_fit_fails_loudly_on_unrelated_points():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1000, size=(500, 2))
    with pytest.raises(ValueError, match='matches'):
        fit_distortion(pts, rng.uniform(0, 1000, size=(500, 2)), UNDIST, SOURCE_SIZE)


def test_remap_matches_distort_points_and_keeps_mask_labels():
    maps = build_remap(UNDIST, TRUE)
    assert maps[0].shape == (UNDIST['height'], UNDIST['width'])
    rows, cols = np.array([0, 540, 1079]), np.array([0, 963, 1926])
    expected = distort_points(np.stack([cols, rows], axis=1).astype(float), UNDIST, TRUE)
    assert np.allclose(np.stack([maps[0][rows, cols], maps[1][rows, cols]], axis=1), expected, atol=1e-3)

    mask = np.zeros((SOURCE_SIZE[1], SOURCE_SIZE[0], 3), np.uint8)
    mask[300:800, 400:1500] = 255
    mask[500:600, 900:1000] = 128
    assert set(np.unique(warp(mask, maps, nearest=True))) == {0, 128, 255}


def test_matching_and_fit_undo_a_synthetic_undistortion():
    """Undistort a textured source image with a known model, then recover the model from the image pair alone."""
    rng = np.random.default_rng(2)
    small = rng.integers(0, 256, size=(SOURCE_SIZE[1] // 8, SOURCE_SIZE[0] // 8), dtype=np.uint8)
    source = cv2.GaussianBlur(cv2.resize(small, SOURCE_SIZE, interpolation=cv2.INTER_NEAREST), (5, 5), 0)
    colmap = warp(source, build_remap(UNDIST, TRUE))

    pc, ps = match_points(source, colmap)
    assert len(pc) > 500
    params, _ = fit_distortion(pc, ps, UNDIST, SOURCE_SIZE)
    warped = warp(source, build_remap(UNDIST, params))
    assert np.abs(local_shifts(warped, colmap)).max() < 0.2
    resized = cv2.resize(source, (UNDIST['width'], UNDIST['height']), interpolation=cv2.INTER_AREA)
    assert np.abs(local_shifts(resized, colmap)).max() > 1.0
