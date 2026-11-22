"""Module 15: recovering the source -> COLMAP undistortion warp. Synthetic data only, no GPU or dataset needed."""

import cv2
import numpy as np
import pytest

from utils.undistort_utils import MODELS, build_remap, distort_points, fit_distortion, initial_params, local_shifts, match_points, warp

# Like VSGD 0758: a mildly distorted 1920x1080 source, undistorted by COLMAP to 1927x1080.
UNDIST = {'fx': 1208.34, 'fy': 1208.34, 'cx': 963.5, 'cy': 540.0, 'width': 1927, 'height': 1080}
SOURCE_SIZE = (1920, 1080)
RADIAL = np.array([1208.34, 1208.34, 960.0, 540.0, -0.0073, 0.0002])

# Like WayveScenes101: a ~120 degree fisheye 1920x1080 source, undistorted to a wide 3271x1224 pinhole image.
WIDE = {'fx': 908.27, 'fy': 905.47, 'cx': 1638.52, 'cy': 626.6, 'width': 3271, 'height': 1224}
FISHEYE = np.array([908.2, 905.3, 961.8, 552.9, -0.004, -0.012, 0.017, -0.008])


def grid(undistorted, n=(20, 12)):
    return np.stack(np.meshgrid(np.linspace(0, undistorted['width'] - 1, n[0]), np.linspace(0, undistorted['height'] - 1, n[1])), -1).reshape(-1, 2)


def test_no_distortion_maps_the_principal_point_onto_itself():
    for model in MODELS:
        params = initial_params(UNDIST, SOURCE_SIZE, model)
        assert len(params) == len(MODELS[model])
        center = np.array([[UNDIST['cx'] - 0.5, UNDIST['cy'] - 0.5]])
        assert np.allclose(distort_points(center, UNDIST, params, model), [[959.5, 539.5]])


@pytest.mark.parametrize('model, undistorted, truth', [('radial', UNDIST, RADIAL), ('fisheye', WIDE, FISHEYE)])
def test_fit_recovers_the_distortion_despite_outliers(model, undistorted, truth):
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [undistorted['width'] - 1, undistorted['height'] - 1], size=(3000, 2))
    src = distort_points(pts, undistorted, truth, model) + rng.normal(0, 0.2, size=pts.shape)
    src[:300] = rng.uniform([0, 0], SOURCE_SIZE, size=(300, 2))  # wrong matches anywhere in the image

    params, dist = fit_distortion(pts, src, undistorted, SOURCE_SIZE, model)
    g = grid(undistorted)
    assert np.abs(distort_points(g, undistorted, params, model) - distort_points(g, undistorted, truth, model)).max() < 0.2
    assert np.median(dist[300:]) < 0.3


def test_radial_cannot_stand_in_for_a_wide_fisheye():
    """The reason the fisheye model exists: on a Wayve-like lens the radial fit leaves a clearly larger residual."""
    rng = np.random.default_rng(3)
    pts = rng.uniform([0, 0], [WIDE['width'] - 1, WIDE['height'] - 1], size=(3000, 2))
    src = distort_points(pts, WIDE, FISHEYE, 'fisheye') + rng.normal(0, 0.2, size=pts.shape)
    _, dist_fisheye = fit_distortion(pts, src, WIDE, SOURCE_SIZE, 'fisheye')
    try:
        _, dist_radial = fit_distortion(pts, src, WIDE, SOURCE_SIZE, 'radial')
    except ValueError:
        return
    assert np.median(dist_radial) > 1.5 * np.median(dist_fisheye)


def test_fit_fails_loudly_on_unrelated_points():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1000, size=(500, 2))
    with pytest.raises(ValueError, match='matches'):
        fit_distortion(pts, rng.uniform(0, 1000, size=(500, 2)), UNDIST, SOURCE_SIZE)


def test_remap_matches_distort_points_and_keeps_mask_labels():
    for model, undistorted, truth in (('radial', UNDIST, RADIAL), ('fisheye', WIDE, FISHEYE)):
        maps = build_remap(undistorted, truth, model)
        assert maps[0].shape == (undistorted['height'], undistorted['width'])
        rows, cols = np.array([0, 540, undistorted['height'] - 1]), np.array([0, 963, undistorted['width'] - 1])
        expected = distort_points(np.stack([cols, rows], axis=1).astype(float), undistorted, truth, model)
        assert np.allclose(np.stack([maps[0][rows, cols], maps[1][rows, cols]], axis=1), expected, atol=1e-3)

    mask = np.zeros((SOURCE_SIZE[1], SOURCE_SIZE[0], 3), np.uint8)
    mask[300:800, 400:1500] = 255
    mask[500:600, 900:1000] = 128
    assert set(np.unique(warp(mask, build_remap(UNDIST, RADIAL), nearest=True))) == {0, 128, 255}


def test_matching_and_fit_undo_a_synthetic_undistortion():
    """Undistort a textured source image with a known model, then recover the model from the image pair alone."""
    rng = np.random.default_rng(2)
    small = rng.integers(0, 256, size=(SOURCE_SIZE[1] // 8, SOURCE_SIZE[0] // 8), dtype=np.uint8)
    source = cv2.GaussianBlur(cv2.resize(small, SOURCE_SIZE, interpolation=cv2.INTER_NEAREST), (5, 5), 0)
    colmap = warp(source, build_remap(UNDIST, RADIAL))

    pc, ps = match_points(source, colmap)
    assert len(pc) > 500
    params, _ = fit_distortion(pc, ps, UNDIST, SOURCE_SIZE)
    warped = warp(source, build_remap(UNDIST, params))
    assert np.abs(local_shifts(warped, colmap)).max() < 0.2
    resized = cv2.resize(source, (UNDIST['width'], UNDIST['height']), interpolation=cv2.INTER_AREA)
    assert np.abs(local_shifts(resized, colmap)).max() > 1.0
