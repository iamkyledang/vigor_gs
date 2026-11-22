"""Module 07: the reflection Gaussian field (3D ellipsoids) rendered through the virtual camera."""

from argparse import Namespace

import pytest
import torch

from gaussian_renderer.reflection import render_reflection
from scene.deform_model import DeformModel
from scene.reflection_model import ReflectionGaussianModel
from tests.helpers import look_at_camera, project_points
from utils.virtual_camera import reflect_camera, reflect_points

pytestmark = pytest.mark.gpu

PIPE = Namespace(depth_ratio=0.0)
TRAIN_ARGS = Namespace(
    position_lr_init=1.6e-4,
    position_lr_final=1.6e-6,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30_000,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    percent_dense=0.01,
)

# Glass at z = 1 facing the cameras, which stand at negative z. Reflected objects are on the cameras' side.
NORMAL = torch.tensor([0.0, 0.0, -1.0], device='cuda')
OFFSET = 1.0


def real_camera():
    return look_at_camera(eye=(0.2, -0.1, -2.0), target=(0.0, 0.0, 1.0))


def make_model(points, sh_degree=0, scale=0.05, opacity=0.9):
    points = torch.as_tensor(points, dtype=torch.float32)
    model = ReflectionGaussianModel(sh_degree)
    model.create_from_points(points, torch.rand(points.shape[0], 3), spatial_lr_scale=1.0)
    model._scaling.data.fill_(torch.log(torch.tensor(scale)).item())
    model._opacity.data.fill_(torch.logit(torch.tensor(opacity)).item())
    return model


def cloud(num_points=300):
    return torch.rand(num_points, 3) * torch.tensor([2.0, 1.4, 1.5]) + torch.tensor([-1.0, -0.7, -1.2])


def test_render_outputs():
    cam = real_camera()
    out = render_reflection(reflect_camera(cam, NORMAL, OFFSET), make_model(cloud()), PIPE, torch.zeros(3, device='cuda'))
    assert out['reflection'].shape == (3, cam.image_height, cam.image_width)
    assert torch.isfinite(out['reflection']).all()
    assert out['radii'].shape == (300,) and out['visibility_filter'].any()
    assert out['reflection'].max() > 0.05


def test_object_appears_where_its_mirror_image_projects():
    cam = real_camera()
    virtual = reflect_camera(cam, NORMAL, OFFSET)
    obj = torch.tensor([[0.4, 0.25, -0.5]], device='cuda')  # in front of the glass, off to one side
    out = render_reflection(virtual, make_model(obj.cpu()), PIPE, torch.zeros(3, device='cuda'))

    brightness = out['reflection'].sum(dim=0)
    peak = brightness.flatten().argmax()
    peak_xy = torch.stack([peak % cam.image_width, peak // cam.image_width]).float()
    expected_xy = project_points(cam, reflect_points(obj, NORMAL, OFFSET))[0]
    assert torch.allclose(peak_xy, expected_xy, atol=1.0), (peak_xy, expected_xy)

    # Rendered un-mirrored from the real camera, the same object would land somewhere else.
    direct_xy = project_points(cam, obj)[0]
    assert (direct_xy - expected_xy).norm() > 5


def test_virtual_view_equals_real_view_of_the_mirrored_scene():
    """The handedness flip of the mirrored view matrix is handled exactly by the rasterizer."""
    cam = real_camera()
    virtual = reflect_camera(cam, NORMAL, OFFSET)
    bg = torch.zeros(3, device='cuda')

    points = cloud()
    model = make_model(points)  # isotropic Gaussians: mirroring leaves their shape unchanged
    mirrored = make_model(reflect_points(points.cuda(), NORMAL, OFFSET).cpu())
    mirrored._f_dc.data.copy_(model._f_dc.data)

    through_virtual = render_reflection(virtual, model, PIPE, bg)['reflection']
    mirrored_scene = render_reflection(cam, mirrored, PIPE, bg)['reflection']
    assert through_virtual.max() > 0.05
    assert torch.allclose(through_virtual, mirrored_scene, atol=2e-3)


def test_gradients_reach_every_parameter_and_the_deformation_network():
    virtual = reflect_camera(real_camera(), NORMAL, OFFSET)
    model = make_model(cloud(), sh_degree=1)
    model.oneupSHdegree()
    deform = DeformModel(scale_dim=3, depth=4, width=32)

    out = render_reflection(virtual, model, PIPE, torch.zeros(3, device='cuda'), offsets=deform.step(model.get_xyz, 0.3))
    out['reflection'].mean().backward()
    for name in model.PARAMS:
        grad = getattr(model, '_' + name).grad
        assert grad is not None and grad.abs().sum() > 0, name
    for head in (deform.network.gaussian_warp, deform.network.gaussian_rotation, deform.network.gaussian_scaling):
        assert head.weight.grad.abs().sum() > 0
    assert out['viewspace_points'].grad.abs().sum() > 0


def test_translation_offset_equals_moving_the_gaussians():
    virtual = reflect_camera(real_camera(), NORMAL, OFFSET)
    bg = torch.zeros(3, device='cuda')
    model = make_model(cloud())
    shift = torch.tensor([0.3, 0.1, -0.2], device='cuda')
    offsets = (shift.expand(300, 3), torch.zeros(300, 4, device='cuda'), torch.zeros(300, 3, device='cuda'))
    deformed = render_reflection(virtual, model, PIPE, bg, offsets=offsets)['reflection']
    model._xyz.data += shift
    assert torch.allclose(render_reflection(virtual, model, PIPE, bg)['reflection'], deformed, atol=1e-5)


def test_random_initialization_stays_on_the_camera_side():
    centers = torch.tensor([[0.0, 0.0, -2.0], [0.5, 0.0, -2.2], [-0.5, 0.2, -1.8]], device='cuda')
    model = ReflectionGaussianModel(0)
    model.create_random(NORMAL, OFFSET, centers, extent=1.5, num_points=5000, radius_scale=2.0)

    assert model.num_points == 5000
    assert model._scaling.shape == (5000, 3)
    assert (model.get_xyz @ NORMAL + OFFSET > 0).all()  # in front of the glass
    assert (model.get_xyz - centers.mean(dim=0)).norm(dim=-1).max() <= 2.0 * 1.5 + 1e-4
    assert model.spatial_lr_scale == 1.5
    assert torch.allclose(model.get_opacity, torch.full((5000, 1), 0.1, device='cuda'), atol=1e-6)

    again = ReflectionGaussianModel(0)
    again.create_random(NORMAL, OFFSET, centers, extent=1.5, num_points=5000, radius_scale=2.0)
    assert torch.equal(again.get_xyz, model.get_xyz)


def test_densify_and_prune_follow_opacity():
    virtual = reflect_camera(real_camera(), NORMAL, OFFSET)
    bg = torch.zeros(3, device='cuda')
    model = make_model(cloud())
    model.training_setup(TRAIN_ARGS)

    out = render_reflection(virtual, model, PIPE, bg)
    out['reflection'].mean().backward()
    model.optimizer.step()
    model.add_densification_stats(out['viewspace_points'], out['visibility_filter'], iteration=1)
    model.optimizer.zero_grad(set_to_none=True)

    model.densify_and_prune(max_grad=0.0, min_opacity=0.005, extent=5.0, max_screen_size=None)
    grown = model.num_points
    assert grown > 300
    for group in model.optimizer.param_groups:
        param = getattr(model, '_' + group['name'])
        assert group['params'][0] is param and param.shape[0] == grown

    model._opacity.data[:100] = torch.logit(torch.tensor(0.001))
    model.denom += 1.0  # zero gradients: nothing more is cloned or split
    model.densify_and_prune(max_grad=1.0, min_opacity=0.005, extent=5.0, max_screen_size=None)
    assert model.num_points == grown - 100

    model.reset_opacity()
    assert model.get_opacity.max() <= 0.01 + 1e-6
    render_reflection(virtual, model, PIPE, bg)['reflection'].mean().backward()
    model.optimizer.step()


def test_ply_round_trip(tmp_path):
    model = make_model(cloud(40), sh_degree=1)
    model._f_rest.data = torch.randn_like(model._f_rest)
    path = str(tmp_path / 'point_cloud.ply')
    model.save_ply(path)

    restored = ReflectionGaussianModel(1)
    restored.load_ply(path)
    for name in model.PARAMS:
        assert torch.allclose(getattr(model, '_' + name), getattr(restored, '_' + name), atol=1e-6), name
    assert restored._scaling.shape == (40, 3)
    assert not hasattr(restored, '_occupancy')
