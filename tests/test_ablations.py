"""Module 11: the ablation switches. Each one removes a single design choice of VIGOR-GS."""

import copy
import shutil
from argparse import ArgumentParser, Namespace

import pytest
import torch

from gaussian_renderer.interface import render_interface
from gaussian_renderer.reflection import render_reflection
from gaussian_renderer.vigor import render_composite
from scene.interface_model import InterfaceGaussianModel
from scene.reflection_model import ReflectionGaussianModel
from tests.helpers import look_at_camera, project_points
from utils.fresnel import BlendModel, schlick_f0
from utils.virtual_camera import reflect_camera, reflect_points

pytestmark = pytest.mark.gpu

PIPE = Namespace(depth_ratio=0.0)
NORMAL = torch.tensor([0.0, 0.0, -1.0], device='cuda')  # glass at z = 1, cameras at negative z
OFFSET = 1.0


def camera():
    return look_at_camera(eye=(0.2, -0.1, -2.0), target=(0.0, 0.0, 1.0))


def interface_cloud(factorized=True, num_points=300):
    points = torch.rand(num_points, 3) * torch.tensor([2.0, 1.4, 1.0]) + torch.tensor([-1.0, -0.7, -1.0])
    model = InterfaceGaussianModel(0, factorized=factorized)
    model.create_from_points(points, torch.rand(num_points, 3), spatial_lr_scale=1.0)
    model._scaling.data.fill_(torch.log(torch.tensor(0.1)).item())
    return model


def reflection_cloud(num_points=200):
    points = torch.rand(num_points, 3) * torch.tensor([2.0, 1.4, 1.0]) + torch.tensor([-1.0, -0.7, -1.0])
    model = ReflectionGaussianModel(0)
    model.create_from_points(points, torch.rand(num_points, 3), spatial_lr_scale=1.0)
    model._scaling.data.fill_(torch.log(torch.tensor(0.08)).item())
    model._opacity.data.fill_(2.0)
    return model


# ---------------------------------------------------------------------------------------------------------------
# No p * o factorization


def test_without_factorization_one_opacity_drives_both_passes():
    bg = torch.zeros(3, device='cuda')
    model = interface_cloud(factorized=False)
    model._occupancy.data.normal_()  # whatever the stored occupancy is, it must not matter
    assert torch.equal(model.get_occupancy, torch.ones_like(model._occupancy))

    out = render_interface(camera(), model, PIPE, bg)
    assert torch.allclose(out['volume_alpha'], out['surface_alpha'], atol=1e-6)
    assert torch.allclose(out['direct'], out['surface_color'], atol=1e-6)

    (out['direct'].mean() + out['glass_map'].mean()).backward()
    assert model._occupancy.grad is None
    assert model._opacity.grad.abs().sum() > 0

    # With the factorization the two passes differ, which is the whole point of it.
    factorized = render_interface(camera(), interface_cloud(factorized=True), PIPE, bg)
    assert not torch.allclose(factorized['volume_alpha'], factorized['surface_alpha'], atol=1e-3)


def test_without_factorization_pruning_follows_opacity():
    train_args = Namespace(
        position_lr_init=1.6e-4, position_lr_final=1.6e-6, position_lr_delay_mult=0.01, position_lr_max_steps=30_000, feature_lr=0.0025, occupancy_lr=0.05, opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001, percent_dense=0.01
    )
    model = interface_cloud(factorized=False, num_points=100)
    model.training_setup(train_args)
    model.denom += 1.0
    model.last_update += 10
    model._opacity.data[:40] = torch.logit(torch.tensor(0.01))
    model._opacity.data[40:] = torch.logit(torch.tensor(0.6))
    model.densify_and_prune(max_grad=1.0, min_occupancy=0.05, extent=5.0, max_screen_size=None, last_reset_iter=-1)
    assert model.num_points == 60


def test_factorization_flag_comes_from_the_model_parameters():
    from arguments import ModelParams

    parser = ArgumentParser()
    model_params = ModelParams(parser)
    assert model_params.extract(parser.parse_args([])).ablate_factorization is False
    assert model_params.extract(parser.parse_args(['--ablate_factorization'])).ablate_factorization is True


# ---------------------------------------------------------------------------------------------------------------
# No virtual camera


def test_without_the_virtual_camera_the_reflection_is_the_unmirrored_view():
    cam = camera()
    field = reflection_cloud()
    H, W = cam.image_height, cam.image_width
    direct, glass = torch.zeros(3, H, W, device='cuda'), torch.ones(1, H, W, device='cuda')
    black = torch.zeros(3, device='cuda')

    mirrored = render_composite(cam, direct, glass, field, PIPE, NORMAL, OFFSET)
    plain = render_composite(cam, direct, glass, field, PIPE, NORMAL, OFFSET, use_virtual_camera=False)

    assert torch.allclose(mirrored['reflection'], render_reflection(reflect_camera(cam, NORMAL, OFFSET), field, PIPE, black)['reflection'])
    assert torch.allclose(plain['reflection'], render_reflection(cam, field, PIPE, black)['reflection'])
    assert not torch.allclose(mirrored['reflection'], plain['reflection'], atol=1e-2)
    assert torch.equal(mirrored['alpha'], plain['alpha'])  # the Fresnel weight depends on the real camera only


# ---------------------------------------------------------------------------------------------------------------
# No Fresnel: one learned constant


def test_learned_constant_replaces_the_fresnel_term():
    cam = look_at_camera(eye=(1.5, 0.0, -1.0), target=(0.0, 0.0, 1.0))
    H, W = cam.image_height, cam.image_width
    glass_map = torch.zeros(1, H, W, device='cuda')
    glass_map[:, :, : W // 2] = 1.0

    fresnel = BlendModel().cuda()
    constant = BlendModel(learn_constant=True).cuda()
    assert not fresnel.is_trainable and constant.is_trainable

    varying = fresnel(cam, NORMAL, OFFSET, glass_map)['alpha'][:, :, : W // 2]
    assert varying.max() - varying.min() > 1e-3  # Schlick changes with the viewing angle

    out = constant(cam, NORMAL, OFFSET, glass_map)
    on_glass = out['alpha'][:, :, : W // 2]
    assert torch.allclose(on_glass, torch.full_like(on_glass, schlick_f0()), atol=1e-6)  # starts at F0, uniform
    assert torch.count_nonzero(out['alpha'][:, :, W // 2 :]) == 0  # still confined to the glass

    # The scene-wide level can be learned.
    target = torch.full_like(on_glass, 0.3)
    optimizer = torch.optim.Adam(constant.parameters(), lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        loss = (constant(cam, NORMAL, OFFSET, glass_map)['alpha'][:, :, : W // 2] - target).abs().mean()
        loss.backward()
        optimizer.step()
    assert torch.sigmoid(constant.constant_logit).item() == pytest.approx(0.3, abs=0.02)


def test_blend_model_is_built_from_the_model_parameters_and_round_trips():
    assert BlendModel.from_args(Namespace(eta_air=1.0, eta_glass=2.0)).f0 == pytest.approx(1 / 9)
    blend = BlendModel.from_args(Namespace(ablate_fresnel=True, optical_correction=True))
    assert blend.constant_logit is not None and blend.correction is not None
    blend.constant_logit.data.fill_(1.5)

    restored = BlendModel.from_args(Namespace(ablate_fresnel=True, optical_correction=True))
    restored.load_state_dict(blend.state_dict())
    assert restored.constant_logit.item() == 1.5
    assert BlendModel().state_dict() == {}


# ---------------------------------------------------------------------------------------------------------------
# No separate reflection field: the interface seen from the virtual camera


def test_surfel_rasterizer_handles_the_mirrored_view():
    cam = camera()
    virtual = reflect_camera(cam, NORMAL, OFFSET)
    obj = torch.tensor([[0.4, 0.25, -0.5]])
    model = InterfaceGaussianModel(0)
    model.create_from_points(torch.cat([obj, obj + 5.0]), torch.ones(2, 3), spatial_lr_scale=1.0)  # the second point is out of view
    model._scaling.data.fill_(torch.log(torch.tensor(0.1)).item())
    model._rotation.data[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    model._occupancy.data.fill_(4.0)
    model._opacity.data.fill_(4.0)

    alpha = render_interface(virtual, model, PIPE, torch.zeros(3, device='cuda'))['volume_alpha'][0]
    peak = alpha.flatten().argmax()
    peak_xy = torch.stack([peak % cam.image_width, peak // cam.image_width]).float()
    expected_xy = project_points(cam, reflect_points(obj.cuda(), NORMAL, OFFSET))[0]
    assert alpha.max() > 0.5
    assert torch.allclose(peak_xy, expected_xy, atol=1.5), (peak_xy, expected_xy)


def test_shared_field_renders_the_interface_through_the_virtual_camera():
    cam = camera()
    interface = interface_cloud()
    interface.freeze()
    H, W = cam.image_height, cam.image_width
    direct, glass = torch.rand(3, H, W, device='cuda'), torch.ones(1, H, W, device='cuda')

    out = render_composite(cam, direct, glass, interface, PIPE, NORMAL, OFFSET)
    expected = render_interface(reflect_camera(cam, NORMAL, OFFSET), interface, PIPE, torch.zeros(3, device='cuda'))['direct']
    assert torch.allclose(out['reflection'], expected)
    assert out['reflection'].max() > 0.05
    assert torch.allclose(out['composite'], (1 - out['alpha']) * direct + out['alpha'] * expected)


# ---------------------------------------------------------------------------------------------------------------
# On the dev scene


def reflection_opt(argv):
    from arguments import ReflectionOptimizationParams

    parser = ArgumentParser()
    op = ReflectionOptimizationParams(parser)
    return op.extract(parser.parse_args(argv))


@pytest.fixture
def model_copy(trained_interface, tmp_path):
    """A private copy of the trained Phase I model, so a variant of Phase II can be run without disturbing others."""
    from extract_plane import extract_plane

    shutil.copytree(trained_interface.root / 'interface', tmp_path / 'interface')
    dataset = copy.copy(trained_interface.dataset)
    dataset.model_path = str(tmp_path)
    extract_plane(dataset, manual_plane=[0.0, 0.0, 1.0, 0.0])
    return dataset


@pytest.mark.data
def test_ignoring_clean_images_trains_on_the_originals(scene_dir, tmp_path):
    import json
    import os

    from PIL import Image

    from scene.dataset_readers import readVigorSceneInfo

    with open(os.path.join(scene_dir, 'colmap', 'split.json')) as f:
        split = json.load(f)
    for name in split['train'] + split['test']:
        Image.new('RGB', (8, 8)).save(tmp_path / (os.path.splitext(name)[0] + '.png'))

    with_clean = readVigorSceneInfo(scene_dir, eval=True, clean_images=str(tmp_path))
    ignored = readVigorSceneInfo(scene_dir, eval=True, clean_images=str(tmp_path), ignore_clean_images=True)
    assert all(c.clean_image is not None for c in with_clean.train_cameras)
    assert all(c.clean_image is None for c in ignored.train_cameras + ignored.test_cameras)
    # No clean folder at all is fine too when it is ignored on purpose.
    readVigorSceneInfo(scene_dir, eval=True, clean_images=str(tmp_path / 'missing'), ignore_clean_images=True)


@pytest.mark.data
def test_direct_only_render_needs_only_phase_one(model_copy, trained_interface):
    from pathlib import Path

    from render_vigor import render_sets

    assert not (Path(model_copy.model_path) / 'reflection').exists()
    label = render_sets(model_copy, trained_interface.pipe, skip_train=True, direct_only=True)
    out_dir = Path(model_copy.model_path) / 'test' / label
    assert label == f'direct_{trained_interface.iterations}'
    assert len(list((out_dir / 'renders').iterdir())) == 34
    assert len(list((out_dir / 'gt').iterdir())) == 34 and len(list((out_dir / 'transparent_masks').iterdir())) == 34
    assert not list((out_dir / 'vis').glob('reflection_*'))


@pytest.mark.data
def test_shared_field_with_pure_fresnel_has_nothing_to_train(model_copy, trained_interface):
    from pathlib import Path

    from render_vigor import render_sets
    from train_reflection import training
    from utils.vigor_utils import load_reflection

    model_copy.shared_reflection_field = True
    history = training(model_copy, reflection_opt(['--iterations', '50']), trained_interface.pipe, testing_iterations=[50], saving_iterations=[50])
    assert {h['iteration'] for h in history} == {0}

    reflection_dir = Path(model_copy.model_path) / 'reflection'
    assert (reflection_dir / 'blend' / 'iteration_0' / 'blend.pth').exists()
    assert not (reflection_dir / 'point_cloud').exists()
    gaussians, deform, blend, iteration = load_reflection(model_copy)
    assert gaussians is None and deform is None and iteration == 0 and not blend.is_trainable

    label = render_sets(model_copy, trained_interface.pipe, skip_train=True)
    assert label == 'ours_0'
    assert (Path(model_copy.model_path) / 'test' / label / 'vis' / 'reflection_00000.png').exists()


@pytest.mark.data
def test_shared_field_with_learned_constant_trains_only_the_blend(model_copy, trained_interface):
    from pathlib import Path

    from train_reflection import training
    from utils.vigor_utils import load_reflection

    model_copy.shared_reflection_field = True
    model_copy.ablate_fresnel = True
    history = training(model_copy, reflection_opt(['--iterations', '40', '--optical_lr', '0.05']), trained_interface.pipe, testing_iterations=[40], saving_iterations=[40])
    assert {h['iteration'] for h in history} == {40}

    _, _, blend, iteration = load_reflection(model_copy)
    assert iteration == 40
    assert torch.sigmoid(blend.constant_logit).item() != pytest.approx(schlick_f0(), abs=1e-4)  # it moved
    assert not (Path(model_copy.model_path) / 'reflection' / 'point_cloud').exists()


@pytest.mark.data
def test_phase_two_without_the_virtual_camera_runs(model_copy, trained_interface):
    from train_reflection import training

    model_copy.ablate_virtual_camera = True
    history = training(model_copy, reflection_opt(['--iterations', '30', '--warm_up', '15', '--refl_init_points', '2000']), trained_interface.pipe, testing_iterations=[30], saving_iterations=[30])
    final = next(h for h in history if h['split'] == 'test')
    assert final['iteration'] == 30 and torch.isfinite(torch.tensor(final['psnr']))
