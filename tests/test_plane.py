"""Module 05: fitting the glass plane to interface geometry."""

import math

import pytest
import torch

from utils.plane_utils import fit_plane_ransac, fit_plane_svd, load_plane, normalize_plane, orient_plane, plane_distance, save_plane


def sample_plane(normal, offset, count, extent=2.0, noise=0.0, generator=None):
    """Random points on the plane n^T x + d = 0, optionally displaced along the normal."""
    normal = torch.nn.functional.normalize(torch.tensor(normal, dtype=torch.float32), dim=0)
    helper = torch.tensor([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    u = torch.nn.functional.normalize(torch.cross(normal, helper, dim=0), dim=0)
    v = torch.cross(normal, u, dim=0)
    coords = (torch.rand(count, 2, generator=generator) - 0.5) * 2 * extent
    points = coords[:, :1] * u + coords[:, 1:] * v - offset * normal
    return points + noise * torch.randn(count, 1, generator=generator) * normal, normal


def angle_deg(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))


def test_svd_recovers_an_exact_plane():
    points, normal = sample_plane([0.0, 0.0, 1.0], 0.0, 200)
    fitted_normal, fitted_offset = fit_plane_svd(points)
    assert angle_deg(fitted_normal, normal) < 1e-3
    assert abs(fitted_offset) < 1e-5
    assert plane_distance(points, fitted_normal, fitted_offset).abs().max() < 1e-5


def test_ransac_survives_noise_and_outliers():
    gen = torch.Generator().manual_seed(1)
    true_offset = -1.7
    inlier_points, normal = sample_plane([0.3, -0.5, 0.8], true_offset, 3000, noise=0.005, generator=gen)
    outliers = (torch.rand(2000, 3, generator=gen) - 0.5) * 8
    points = torch.cat([inlier_points, outliers])[torch.randperm(5000, generator=gen)]

    # Plain least squares is dragged away by the outliers; RANSAC is not.
    naive_normal, _ = fit_plane_svd(points)
    assert angle_deg(naive_normal, normal) > 1.0

    fit = fit_plane_ransac(points, threshold=0.02)
    assert angle_deg(fit['normal'], normal) < 1.0
    signed_offset = fit['offset'] * torch.sign(fit['normal'] @ normal).item()
    assert abs(signed_offset - true_offset) < 0.01
    assert 0.55 < fit['inlier_ratio'] < 0.68
    assert fit['rmse'] < 0.01
    assert fit['inliers'].shape == (5000,) and fit['inliers'].dtype == torch.bool


def test_ransac_picks_the_dominant_of_two_planes():
    gen = torch.Generator().manual_seed(2)
    major, major_normal = sample_plane([0.0, 0.0, 1.0], -1.0, 3000, noise=0.002, generator=gen)
    minor, _ = sample_plane([1.0, 0.0, 0.0], 0.5, 1000, noise=0.002, generator=gen)
    fit = fit_plane_ransac(torch.cat([major, minor]), threshold=0.01)
    assert angle_deg(fit['normal'], major_normal) < 0.5
    assert fit['inlier_ratio'] == pytest.approx(0.75, abs=0.03)


def test_ransac_is_deterministic_and_rejects_too_few_points():
    points, _ = sample_plane([0.0, 1.0, 0.0], 0.3, 500, noise=0.01, generator=torch.Generator().manual_seed(3))
    first = fit_plane_ransac(points, threshold=0.02)
    second = fit_plane_ransac(points, threshold=0.02)
    assert torch.equal(first['normal'], second['normal']) and first['offset'] == second['offset']
    with pytest.raises(ValueError):
        fit_plane_ransac(points[:2], threshold=0.02)


def test_orient_puts_cameras_on_the_positive_side():
    cameras = torch.tensor([[0.0, 0.0, 3.0], [1.0, 0.5, 2.5]])
    normal, offset = orient_plane(torch.tensor([0.0, 0.0, -1.0]), 0.5, cameras)
    assert torch.equal(normal, torch.tensor([0.0, 0.0, 1.0])) and offset == -0.5
    assert (plane_distance(cameras, normal, offset) > 0).all()
    # Already oriented: unchanged.
    again_normal, again_offset = orient_plane(normal, offset, cameras)
    assert torch.equal(again_normal, normal) and again_offset == offset


def test_normalize_and_save_load_round_trip(tmp_path):
    normal, offset = normalize_plane(torch.tensor([0.0, 3.0, 4.0]), 10.0)
    assert torch.allclose(normal, torch.tensor([0.0, 0.6, 0.8])) and float(offset) == pytest.approx(2.0)

    path = str(tmp_path / 'glass_plane.json')
    save_plane(path, normal, offset, source='manual', inlier_ratio=0.9)
    loaded_normal, loaded_offset, record = load_plane(path, device='cpu')
    assert torch.allclose(loaded_normal, normal) and loaded_offset == pytest.approx(2.0)
    assert record['source'] == 'manual' and record['inlier_ratio'] == 0.9


def test_save_plane_accepts_numpy_and_torch_scalars(tmp_path):
    """numpy 2 (Colab) keeps np.float32 * float as float32, e.g. the RANSAC threshold built from cameras_extent."""
    import numpy as np

    path = str(tmp_path / 'glass_plane.json')
    save_plane(path, torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.5), threshold=np.float32(0.01) * 3.0, num_points=np.int64(7), rmse=torch.tensor(0.002), extent=np.array([1.0, 2.0]))
    _, offset, record = load_plane(path, device='cpu')
    assert offset == pytest.approx(0.5)
    assert record['threshold'] == pytest.approx(0.03) and record['num_points'] == 7
    assert record['rmse'] == pytest.approx(0.002) and record['extent'] == [1.0, 2.0]


# ---------------------------------------------------------------------------------------------------------------
# Extraction from an interface field


def glass_scene(glass_z=0.5, wall_z=2.0, tilt=0.0):
    """A transparent pane of disks (p high, o low) in front of an opaque wall (p high, o high)."""
    from scene.interface_model import InterfaceGaussianModel

    axis = torch.linspace(-1.6, 1.6, 44)
    gx, gy = torch.meshgrid(axis, axis, indexing='xy')
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    pane = torch.cat([grid, glass_z + tilt * grid[:, :1]], dim=-1)
    wall = torch.cat([grid * 1.5, torch.full((grid.shape[0], 1), wall_z)], dim=-1)

    model = InterfaceGaussianModel(0)
    model.create_from_points(torch.cat([pane, wall]), torch.rand(2 * grid.shape[0], 3), spatial_lr_scale=1.0)
    num = grid.shape[0]
    model._scaling.data.fill_(math.log(0.08))
    model._rotation.data[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # disks in the xy plane
    model._occupancy.data.fill_(6.0)
    model._opacity.data[:num] = -6.0  # the pane lets light through
    model._opacity.data[num:] = 6.0  # the wall does not
    return model


def glass_views(mask_value=True):
    from tests.helpers import look_at_camera

    views = []
    for eye in [(0.0, 0.0, -3.0), (0.8, 0.2, -2.8), (-0.7, -0.3, -3.2), (0.3, 0.6, -2.6)]:
        cam = look_at_camera(eye=eye, target=(0.0, 0.0, 0.5))
        cam.time = 0.0
        cam.gt_transparent_mask = torch.full((1, cam.image_height, cam.image_width), mask_value, dtype=torch.bool, device='cuda')
        views.append(cam)
    return views


@pytest.mark.gpu
def test_collected_points_lie_on_the_glass_not_the_wall():
    from extract_plane import collect_glass_points
    from scene.deform_model import DeformModel

    model = glass_scene(glass_z=0.5)
    deform = DeformModel(scale_dim=2, depth=2, width=8)
    views = glass_views()
    points = collect_glass_points(views, model, deform, torch.zeros(3, device='cuda'), max_points_per_view=1500)

    assert 1000 < points.shape[0] <= 4 * 1500
    assert (points[:, 2] - 0.5).abs().median() < 0.01

    fit = fit_plane_ransac(points, threshold=0.02)
    centers = torch.stack([v.camera_center for v in views])
    normal, offset = orient_plane(fit['normal'], fit['offset'], centers)
    assert angle_deg(normal, torch.tensor([0.0, 0.0, 1.0], device='cuda')) < 1.0
    assert normal[2] < 0  # the cameras sit at negative z, so the normal faces them
    assert offset == pytest.approx(0.5, abs=0.02)  # -z + 0.5 = 0
    assert fit['inlier_ratio'] > 0.9


@pytest.mark.gpu
def test_a_tilted_pane_is_recovered():
    from extract_plane import collect_glass_points
    from scene.deform_model import DeformModel

    tilt = 0.2
    model = glass_scene(glass_z=0.5, tilt=tilt)
    deform = DeformModel(scale_dim=2, depth=2, width=8)
    points = collect_glass_points(glass_views(), model, deform, torch.zeros(3, device='cuda'))
    fit = fit_plane_ransac(points, threshold=0.03)
    expected = torch.nn.functional.normalize(torch.tensor([-tilt, 0.0, 1.0], device='cuda'), dim=0)
    assert angle_deg(fit['normal'], expected) < 2.0


@pytest.mark.gpu
def test_no_points_outside_the_glass_mask():
    from extract_plane import collect_glass_points
    from scene.deform_model import DeformModel

    model = glass_scene()
    deform = DeformModel(scale_dim=2, depth=2, width=8)
    points = collect_glass_points(glass_views(mask_value=False), model, deform, torch.zeros(3, device='cuda'))
    assert points.shape == (0, 3)


@pytest.mark.gpu
@pytest.mark.data
def test_manual_plane_is_normalized_oriented_and_saved(trained_interface):
    from extract_plane import extract_plane
    from utils.vigor_utils import PLANE_FILE, load_interface

    result = extract_plane(trained_interface.dataset, manual_plane=[0.0, 0.0, 2.0, 1.0])
    normal, offset, record = load_plane(str(trained_interface.root / PLANE_FILE))
    assert record['source'] == 'manual'
    assert normal.norm().item() == pytest.approx(1.0, abs=1e-6)
    assert abs(offset) == pytest.approx(0.5, abs=1e-6)
    assert torch.allclose(normal, result['normal'])

    scene, gaussians, deform = load_interface(trained_interface.dataset)
    assert scene.loaded_iter == trained_interface.iterations
    assert gaussians.num_points > 1000
    centers = torch.stack([cam.camera_center for cam in scene.getTrainCameras()])
    assert plane_distance(centers, normal, offset).mean() > 0


@pytest.mark.gpu
@pytest.mark.data
def test_extraction_from_a_trained_interface_reports_its_quality(trained_interface):
    """On real data the fit either yields a unit-normal plane with its statistics or fails with a clear message."""
    from extract_plane import extract_plane
    from utils.vigor_utils import PLANE_FILE

    try:
        result = extract_plane(trained_interface.dataset, save_points=True)
    except RuntimeError as error:
        assert 'nothing to fit a plane to' in str(error)
        return
    normal, offset, record = load_plane(str(trained_interface.root / PLANE_FILE))
    assert record['source'] == 'ransac'
    assert normal.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert 0.0 < record['inlier_ratio'] <= 1.0 and record['num_points'] >= 3
    assert record['mean_camera_distance'] > 0
    assert result['offset'] == pytest.approx(offset)
    assert (trained_interface.root / 'glass_points.ply').exists()
