"""Module 03: the interface Gaussian field (2D disks with occupancy p and opacity o) and its two-pass renderer."""

from argparse import Namespace

import pytest
import torch

from gaussian_renderer.interface import render_interface
from scene.deform_model import DeformModel
from scene.interface_model import InterfaceGaussianModel
from tests.helpers import look_at_camera

pytestmark = pytest.mark.gpu

PIPE = Namespace(depth_ratio=0.0)
TRAIN_ARGS = Namespace(
    position_lr_init=1.6e-4,
    position_lr_final=1.6e-6,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30_000,
    deform_lr_max_steps=40_000,
    feature_lr=0.0025,
    occupancy_lr=0.05,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    percent_dense=0.01,
)


def make_model(num_points=400, sh_degree=1):
    points = (torch.rand(num_points, 3) - 0.5) * torch.tensor([2.0, 1.4, 0.6])
    model = InterfaceGaussianModel(sh_degree)
    model.create_from_points(points, torch.rand(num_points, 3), spatial_lr_scale=1.0)
    return model


@pytest.fixture
def cam():
    return look_at_camera(eye=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0))


@pytest.fixture
def bg():
    return torch.zeros(3, device='cuda')


def set_activated(model, name, value):
    getattr(model, '_' + name).data.fill_(torch.logit(torch.tensor(value)).item())


def test_render_outputs(cam, bg):
    out = render_interface(cam, make_model(), PIPE, bg)
    H, W = cam.image_height, cam.image_width
    channels = {'direct': 3, 'surface_color': 3, 'volume_normal': 3, 'volume_depth_normal': 3, 'surface_normal': 3, 'surface_depth_normal': 3}
    for key in ['direct', 'volume_alpha', 'volume_depth', 'volume_normal', 'volume_depth_normal', 'volume_dist', 'surface_color', 'surface_alpha', 'surface_depth', 'surface_normal', 'surface_depth_normal', 'surface_dist', 'surface_opacity', 'glass_map']:
        assert out[key].shape == (channels.get(key, 1), H, W), key
        assert torch.isfinite(out[key]).all(), key
    assert out['radii'].shape == (400,)
    assert out['visibility_filter'].any()
    assert out['surface_alpha'].max() > 0.05


def test_gradients_reach_every_parameter(cam, bg):
    model = make_model()
    model.oneupSHdegree()  # the view-dependent SH bands only contribute once they are switched on
    out = render_interface(cam, model, PIPE, bg)
    (out['direct'].mean() + out['glass_map'].mean() + out['surface_depth'].mean() + out['surface_normal'].abs().mean()).backward()
    for name in model.PARAMS:
        grad = getattr(model, '_' + name).grad
        assert grad is not None and grad.abs().sum() > 0, name
    assert out['viewspace_points'].grad.abs().sum() > 0


def test_occupancy_and_opacity_play_separate_roles(cam, bg):
    model = make_model()
    set_activated(model, 'occupancy', 0.6)

    # Present but fully transparent: nothing in the direct image, yet the surface is still there and reads as glass.
    set_activated(model, 'opacity', 1e-6)
    transparent = render_interface(cam, model, PIPE, bg)
    assert transparent['direct'].abs().max() < 1e-3
    assert transparent['volume_alpha'].max() < 1e-3
    assert transparent['surface_alpha'].max() > 0.3
    assert transparent['surface_opacity'].max() < 1e-3
    assert torch.allclose(transparent['glass_map'], torch.ones_like(transparent['glass_map']), atol=1e-3)

    # Present and opaque: both passes see the same thing and nothing reads as glass where the surface is solid.
    set_activated(model, 'opacity', 1 - 1e-6)
    opaque = render_interface(cam, model, PIPE, bg)
    assert torch.allclose(opaque['surface_alpha'], transparent['surface_alpha'], atol=1e-5)
    assert torch.allclose(opaque['volume_alpha'], opaque['surface_alpha'], atol=1e-4)
    assert torch.allclose(opaque['surface_opacity'], opaque['surface_alpha'], atol=1e-4)
    assert torch.allclose(opaque['direct'], opaque['surface_color'], atol=1e-4)


def test_zero_offsets_match_the_canonical_render(cam, bg):
    model = make_model()
    zeros = (torch.zeros(400, 3, device='cuda'), torch.zeros(400, 4, device='cuda'), torch.zeros(400, 2, device='cuda'))
    canonical = render_interface(cam, model, PIPE, bg)
    deformed = render_interface(cam, model, PIPE, bg, offsets=zeros)
    for key in ['direct', 'surface_alpha', 'surface_depth', 'glass_map']:
        assert torch.allclose(canonical[key], deformed[key], atol=1e-5), key


def test_translation_offset_equals_moving_the_gaussians(cam, bg):
    model = make_model()
    shift = torch.tensor([0.4, -0.1, 0.2], device='cuda')
    offsets = (shift.expand(400, 3), torch.zeros(400, 4, device='cuda'), torch.zeros(400, 2, device='cuda'))
    deformed = render_interface(cam, model, PIPE, bg, offsets=offsets)

    canonical = render_interface(cam, model, PIPE, bg)
    assert not torch.allclose(canonical['direct'], deformed['direct'], atol=1e-3)

    model._xyz.data += shift
    moved = render_interface(cam, model, PIPE, bg)
    assert torch.allclose(moved['direct'], deformed['direct'], atol=1e-5)
    assert torch.allclose(moved['surface_depth'], deformed['surface_depth'], atol=1e-4)


def test_gradients_reach_the_deformation_network(cam, bg):
    model = make_model()
    deform = DeformModel(scale_dim=2, depth=4, width=32)
    out = render_interface(cam, model, PIPE, bg, offsets=deform.step(model.get_xyz, 0.5))
    out['direct'].mean().backward()
    for head in (deform.network.gaussian_warp, deform.network.gaussian_rotation, deform.network.gaussian_scaling):
        assert head.weight.grad.abs().sum() > 0
    assert model._occupancy.grad.abs().sum() > 0  # the canonical field still trains alongside


def test_ply_round_trip(tmp_path):
    model = make_model(num_points=50, sh_degree=2)
    model._occupancy.data = torch.randn_like(model._occupancy)
    model._f_rest.data = torch.randn_like(model._f_rest)
    path = str(tmp_path / 'point_cloud.ply')
    model.save_ply(path)

    restored = InterfaceGaussianModel(2)
    restored.load_ply(path)
    for name in model.PARAMS:
        assert torch.allclose(getattr(model, '_' + name), getattr(restored, '_' + name), atol=1e-6), name
    assert restored._scaling.shape == (50, 2)
    assert restored.active_sh_degree == 2


def test_densify_and_prune_keeps_parameters_and_optimizer_in_sync(cam, bg):
    model = make_model()
    model.training_setup(TRAIN_ARGS)
    set_activated(model, 'occupancy', 0.5)

    out = render_interface(cam, model, PIPE, bg)
    out['direct'].mean().backward()
    model.optimizer.step()  # populate Adam state so the surgery has something to carry over
    model.add_densification_stats(out['viewspace_points'], out['visibility_filter'], iteration=10)
    model.optimizer.zero_grad(set_to_none=True)

    before = model.num_points
    model.densify_and_prune(max_grad=0.0, min_occupancy=0.05, extent=5.0, max_screen_size=None, last_reset_iter=-1)
    after = model.num_points
    assert after > before

    for group in model.optimizer.param_groups:
        param = getattr(model, '_' + group['name'])
        assert group['params'][0] is param
        assert param.shape[0] == after
        assert model.optimizer.state[param]['exp_avg'].shape == param.shape
    for stat in (model.xyz_gradient_accum, model.denom, model.last_update, model.max_radii2D):
        assert stat.shape[0] == after

    out = render_interface(cam, model, PIPE, bg)
    out['direct'].mean().backward()
    model.optimizer.step()


def test_pruning_follows_occupancy_and_spares_unseen_gaussians(cam, bg):
    model = make_model(num_points=100)
    model.training_setup(TRAIN_ARGS)
    model.xyz_gradient_accum += 0.0
    model.denom += 1.0  # zero gradient everywhere, so nothing is cloned or split

    model._occupancy.data[:60] = torch.logit(torch.tensor(0.01))  # below the cull threshold
    model._occupancy.data[60:] = torch.logit(torch.tensor(0.5))
    model._opacity.data[:] = torch.logit(torch.tensor(0.001))  # opacity must not drive pruning
    model.last_update[:30] = 500  # seen after the reset at 300
    model.last_update[30:60] = 100  # not seen since the reset

    model.densify_and_prune(max_grad=1.0, min_occupancy=0.05, extent=5.0, max_screen_size=None, last_reset_iter=300)
    assert model.num_points == 70


def test_resets_cap_the_activated_values(cam, bg):
    model = make_model(num_points=50)
    model.training_setup(TRAIN_ARGS)
    set_activated(model, 'occupancy', 0.8)
    set_activated(model, 'opacity', 0.7)

    model.reset_occupancy()
    assert torch.allclose(model.get_occupancy, torch.full((50, 1), 0.01, device='cuda'), atol=1e-6)
    assert torch.allclose(model.get_opacity, torch.full((50, 1), 0.7, device='cuda'), atol=1e-6)
    assert model.optimizer.param_groups[3]['params'][0] is model._occupancy

    model.reset_opacity()
    assert torch.allclose(model.get_opacity, torch.full((50, 1), 0.01, device='cuda'), atol=1e-6)


def test_capture_restore_round_trip(cam, bg):
    model = make_model(num_points=50)
    model.training_setup(TRAIN_ARGS)
    render_interface(cam, model, PIPE, bg)['direct'].mean().backward()
    model.optimizer.step()

    restored = InterfaceGaussianModel(1)
    restored.restore(model.capture(), TRAIN_ARGS)
    for name in model.PARAMS:
        assert torch.equal(getattr(model, '_' + name), getattr(restored, '_' + name)), name
    assert restored.optimizer.state_dict()['state'].keys() == model.optimizer.state_dict()['state'].keys()


def test_freeze_blocks_gradients(cam, bg):
    model = make_model(num_points=50)
    model.training_setup(TRAIN_ARGS)
    model.freeze()
    assert model.optimizer is None

    # The render still tracks the screen-space means, but nothing flows back into the frozen field.
    out = render_interface(cam, model, PIPE, bg)
    (out['direct'].mean() + out['glass_map'].mean()).backward()
    for name in model.PARAMS:
        param = getattr(model, '_' + name)
        assert not param.requires_grad and param.grad is None, name
