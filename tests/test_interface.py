"""Phase I: loss terms, the interface renderer and a short training run on the synthetic scene."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from utils.losses import glass_prior_loss, motion_loss, optical_loss, pixel_l1, surface_loss


def test_pixel_l1_sums_channels_and_averages_valid_pixels():
    prediction = torch.zeros(3, 2, 2)
    target = torch.zeros(3, 2, 2)
    target[:, 0, 0] = torch.tensor([0.1, 0.2, 0.3])  # pixel error 0.6
    target[:, 1, 1] = 5.0  # an invalid pixel
    valid = torch.tensor([[[True, True], [True, False]]])
    assert pixel_l1(prediction, target, valid).item() == pytest.approx(0.6 / 3)
    with pytest.raises(ValueError):
        pixel_l1(prediction, target, torch.zeros_like(valid))


def test_surface_motion_and_optical_terms():
    normal = torch.zeros(3, 1, 2)
    normal[2] = 1
    depth_normal = normal.clone()
    depth_normal[:, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    assert surface_loss(normal, depth_normal, valid).item() == pytest.approx(1.0)  # (|1| + |-1|) / 2 pixels
    assert surface_loss(normal, depth_normal, torch.zeros_like(valid)).item() == 0.0
    a, b = torch.rand(3, 1, 2), torch.rand(3, 1, 2)
    # Same change predicted as observed: no motion penalty, whatever the absolute images.
    assert motion_loss(a + 1, b + 1, a + 7, b + 7, valid).item() == pytest.approx(0.0, abs=1e-6)
    h = torch.tensor([[[0.5, 0.0]]])
    assert optical_loss(h, valid).item() == pytest.approx(0.125)


def test_glass_prior_pushes_the_glass_map_toward_the_mask():
    glass_map = torch.full((1, 2, 2), 0.5, requires_grad=True)
    mask = torch.tensor([[[True, False], [False, False]]])
    glass_prior_loss(glass_map, mask, torch.ones_like(mask)).backward()
    assert glass_map.grad[0, 0, 0] < 0 and glass_map.grad[0, 1, 1] > 0


@pytest.mark.gpu
def test_render_interface_outputs(synthetic_scene):
    from gaussian_renderer.interface import render_interface
    from scene import Scene
    from tests.conftest import model_args
    from train_interface import build_interface

    args = model_args(synthetic_scene.path)
    scene = Scene(args)
    field, deformation = build_interface(args, scene)
    field.create_from_points(scene.point_cloud.points, scene.point_cloud.colors, scene.extent)
    pipe = SimpleNamespace(depth_ratio=0.0, debug=False)
    cam = scene.getTrainCameras()[0]
    out = render_interface(cam, field, deformation, pipe, surface=True)
    for key in ('direct', 'normal', 'depth_normal'):
        assert out[key].shape == (3, 48, 64) and torch.isfinite(out[key]).all()
    assert out['depth'].shape == (1, 48, 64) and (out['depth'] >= 0).all()
    assert 0 <= out['glass_map'].min() and out['glass_map'].max() <= 1
    assert out['surface_valid'].any()
    # 2DGS orients the rendered normals toward the camera; the depth normals use the same orientation.
    from utils.point_utils import camera_rays

    _, rays = camera_rays(cam)
    valid = out['surface_valid'][0]
    facing = (out['normal'].permute(1, 2, 0) * rays).sum(-1)[valid]
    facing_depth = (out['depth_normal'].permute(1, 2, 0) * rays).sum(-1)[valid]
    assert (facing < 0).float().mean() > 0.9 and (facing_depth < 0).float().mean() > 0.9


@pytest.mark.gpu
@pytest.mark.slow
def test_short_phase_one_improves_and_resumes(synthetic_scene, tmp_path):
    import train_interface

    model = str(tmp_path / 'model')
    base = ['-s', synthetic_scene.path, '-m', model, '--quiet', '--deform_depth', '2', '--deform_width', '32', '--warm_up', '100',
            '--surface_from_iter', '50', '--densify_from_iter', '50', '--densification_interval', '50', '--test_interval', '150',
            '--report_interval', '50', '--checkpoint_interval', '150', '--presence_reset_interval', '0']
    train_interface.main(base + ['--iterations', '300'])
    log = [json.loads(line) for line in open(os.path.join(model, 'interface', 'training.jsonl'))]
    tests = [r['test_psnr'] for r in log if 'test_psnr' in r]
    assert len(tests) == 2 and tests[-1] > 18.0
    assert os.path.exists(os.path.join(model, 'config.json'))
    assert os.path.exists(os.path.join(model, 'interface', 'point_cloud.ply'))
    with pytest.raises(FileExistsError):
        train_interface.main(base + ['--iterations', '300'])
    train_interface.main(base + ['--iterations', '330', '--resume'])
    log = [json.loads(line) for line in open(os.path.join(model, 'interface', 'training.jsonl'))]
    assert log[-1]['iteration'] == 330 and len({r['session_id'] for r in log}) == 2
