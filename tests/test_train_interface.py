"""Module 04: Phase I training (interface field + deformation MLP against the clean targets)."""

import os
from argparse import Namespace

import pytest
import torch

from gaussian_renderer.interface import render_interface
from scene.deform_model import DeformModel
from scene.interface_model import InterfaceGaussianModel
from tests.helpers import look_at_camera
from tests.helpers import parse_interface_args as parse
from train_interface import interface_losses, time_noise_std

pytestmark = pytest.mark.gpu

PIPE = Namespace(depth_ratio=0.0)


def fake_view(cam, clean, mask):
    cam.clean_image = clean
    cam.gt_transparent_mask = mask
    return cam


@pytest.fixture
def setup():
    _, opt, _ = parse([])
    points = (torch.rand(300, 3) - 0.5) * torch.tensor([2.0, 1.4, 0.6])
    model = InterfaceGaussianModel(0)
    model.create_from_points(points, torch.rand(300, 3), spatial_lr_scale=1.0)
    cam = look_at_camera(eye=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0))
    H, W = cam.image_height, cam.image_width
    mask = torch.zeros(1, H, W, dtype=torch.bool, device='cuda')
    mask[:, :, : W // 2] = True
    view = fake_view(cam, torch.rand(3, H, W, device='cuda'), mask)
    return model, view, opt


def test_time_noise_anneals_to_zero():
    opt = Namespace(time_noise_scale=0.1, time_noise_until_iter=1000)
    assert time_noise_std(opt, 0, num_frames=100) == pytest.approx(0.001)
    assert time_noise_std(opt, 500, num_frames=100) == pytest.approx(0.0005)
    assert time_noise_std(opt, 2000, num_frames=100) == 0.0
    assert time_noise_std(Namespace(time_noise_scale=0.0, time_noise_until_iter=1000), 0, 100) == 0.0


def test_loss_terms_follow_the_schedule(setup):
    model, view, opt = setup
    bg = torch.zeros(3, device='cuda')
    out = render_interface(view, model, PIPE, bg)

    early = interface_losses(out, view, model, opt, iteration=1)
    assert set(early) == {'rgb'}

    late = interface_losses(out, view, model, opt, iteration=opt.reg_from_iter)
    assert set(late) == {'rgb', 'norm', 'mask', 'occupancy'}
    assert all(torch.isfinite(v) and v.dim() == 0 for v in late.values())

    opt.dist_loss_weight = 100.0
    opt.lambda_deform_reg = 1.0
    offsets = (torch.full((300, 3), 0.5, device='cuda'), torch.zeros(300, 4, device='cuda'), torch.zeros(300, 2, device='cuda'))
    full = interface_losses(out, view, model, opt, iteration=opt.reg_from_iter, offsets=offsets)
    assert set(full) == {'rgb', 'norm', 'dist', 'mask', 'occupancy', 'deform'}
    assert full['deform'].item() == pytest.approx(0.5)


def test_rgb_loss_compares_against_the_clean_image(setup):
    model, view, opt = setup
    bg = torch.zeros(3, device='cuda')
    out = render_interface(view, model, PIPE, bg)
    view.original_image = torch.ones_like(view.clean_image)

    view.clean_image = out['direct'].detach().clone()
    assert interface_losses(out, view, model, opt, iteration=1)['rgb'].item() == pytest.approx(0.0, abs=1e-5)


def test_mask_loss_pushes_opacity_down_inside_the_glass_mask(setup):
    model, view, opt = setup
    bg = torch.zeros(3, device='cuda')
    model._occupancy.data.fill_(2.0)  # a clearly present surface
    out = render_interface(view, model, PIPE, bg)
    interface_losses(out, view, model, opt, iteration=opt.reg_from_iter)['mask'].backward()

    # Descending the loss lowers o where the mask says glass (left half) and raises it elsewhere.
    grad = model._opacity.grad.squeeze(-1)
    on_glass = model.get_xyz[:, 0] < -0.2  # the camera looks down +z with x to the right: left half of the image
    off_glass = model.get_xyz[:, 0] > 0.2
    assert grad[on_glass].mean() > 0
    assert grad[off_glass].mean() < 0


@pytest.mark.data
def test_short_training_run_improves_and_saves(trained_interface):
    """A few hundred iterations on the dev scene: static warm-up, then the deformation MLP switches on."""
    dataset = trained_interface.dataset

    test = {h['iteration']: h for h in trained_interface.history if h['split'] == 'test'}
    assert test[400]['psnr'] > test[1]['psnr'] + 3.0
    assert test[400]['psnr'] > 15.0
    assert 0.0 <= test[400]['glass_iou'] <= 1.0

    interface = trained_interface.root / 'interface'
    assert (interface / 'cfg_args').exists()
    ply = interface / 'point_cloud' / 'iteration_400' / 'point_cloud.ply'
    assert ply.exists()
    assert (interface / 'deform' / 'iteration_400' / 'deform.pth').exists()

    # What was saved can be loaded back into a fresh field and deformation model.
    restored = InterfaceGaussianModel(dataset.sh_degree)
    restored.load_ply(str(ply))
    assert restored.num_points > 1000
    deform = DeformModel.from_args(2, dataset)
    assert deform.load_weights(str(interface)) == 400
    assert any(p.abs().sum() > 0 for p in deform.network.gaussian_warp.parameters())
    assert os.path.basename(dataset.model_path) != 'interface'  # the caller's args still point at the model root
