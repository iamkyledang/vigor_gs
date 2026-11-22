"""Module 09: Phase II training (reflection field + deformation MLP against the original images, Phase I frozen)."""

import hashlib
from argparse import ArgumentParser, Namespace

import pytest
import torch

from arguments import ModelParams, PipelineParams, ReflectionOptimizationParams
from gaussian_renderer.vigor import render_composite, render_direct
from scene.deform_model import DeformModel
from scene.reflection_model import ReflectionGaussianModel
from tests.helpers import look_at_camera
from tests.test_plane import glass_scene
from train_reflection import DirectCache, reflection_losses
from utils.fresnel import BlendModel
from utils.image_utils import psnr

pytestmark = pytest.mark.gpu

PIPE = Namespace(depth_ratio=0.0)
NORMAL = torch.tensor([0.0, 0.0, -1.0], device='cuda')  # glass pane of `glass_scene` at z = 0.5, cameras at negative z
OFFSET = 0.5
BLEND = BlendModel(f0=0.25)  # a strongly reflective pane keeps this small problem well conditioned


def reflection_opt(**overrides):
    parser = ArgumentParser()
    op = ReflectionOptimizationParams(parser)
    opt = op.extract(parser.parse_args([]))
    vars(opt).update(overrides)
    return opt


def views():
    cams = []
    for i, eye in enumerate([(0.0, 0.0, -3.0), (0.9, 0.2, -2.8), (-0.8, -0.3, -3.1), (0.3, 0.7, -2.6), (-0.4, 0.5, -2.9), (0.6, -0.6, -3.0)]):
        cam = look_at_camera(eye=eye, target=(0.0, 0.0, 0.5))
        cam.time, cam.uid, cam.data_device = 0.0, i, torch.device('cuda')
        cam.gt_transparent_mask = torch.ones(1, cam.image_height, cam.image_width, dtype=torch.bool, device='cuda')
        cams.append(cam)
    return cams


def blobs(positions, color):
    model = ReflectionGaussianModel(0)
    positions = torch.as_tensor(positions, dtype=torch.float32)
    model.create_from_points(positions, torch.as_tensor(color, dtype=torch.float32).expand(positions.shape[0], 3), spatial_lr_scale=1.0)
    model._scaling.data.fill_(torch.log(torch.tensor(0.25)).item())
    model._opacity.data.fill_(3.0)
    return model


def test_loss_terms():
    opt = reflection_opt()
    cam = views()[0]
    cam.original_image = torch.rand(3, cam.image_height, cam.image_width, device='cuda')
    out = {'composite': cam.original_image.clone(), 'h': None}
    assert set(reflection_losses(out, cam, opt)) == {'rgb'}
    assert reflection_losses(out, cam, opt)['rgb'].item() == pytest.approx(0.0, abs=1e-6)

    offsets = (torch.full((10, 3), 0.2, device='cuda'), None, None)
    out['h'] = torch.full((1, cam.image_height, cam.image_width), 0.5, device='cuda')
    losses = reflection_losses(out, cam, opt, offsets)
    assert set(losses) == {'rgb', 'motion', 'opt'}
    assert losses['motion'].item() == pytest.approx(opt.lambda_motion * 0.2)
    assert losses['opt'].item() == pytest.approx(opt.lambda_opt * 0.25)

    opt.lambda_motion = 0.0
    assert 'motion' not in reflection_losses(out, cam, opt, offsets)


def test_cache_holds_the_frozen_direct_image():
    interface = glass_scene()
    interface.freeze()
    deform = DeformModel(scale_dim=2, depth=2, width=8)
    deform.freeze()
    cams = views()
    bg = torch.zeros(3, device='cuda')

    cache = DirectCache(cams, interface, deform, PIPE, bg)
    direct, glass_map = cache(cams[2])
    fresh = render_direct(cams[2], interface, deform, PIPE, bg)
    assert torch.equal(direct, fresh['direct']) and torch.equal(glass_map, fresh['glass_map'])
    assert not direct.requires_grad
    assert glass_map.mean() > 0.5  # the pane covers most of the view

    gt_cache = DirectCache(cams, interface, deform, PIPE, bg, use_gt_mask=True)
    assert torch.equal(gt_cache(cams[2])[1], torch.ones_like(glass_map))


def test_reflection_field_learns_the_reflection_while_the_interface_stays_fixed():
    """A scene with a known reflection: Phase II must recover it without touching Phase I."""
    opt = reflection_opt(lambda_motion=0.0)
    interface = glass_scene()
    interface.freeze()
    interface_deform = DeformModel(scale_dim=2, depth=2, width=8)
    interface_deform.freeze()
    interface_before = {name: getattr(interface, '_' + name).clone() for name in interface.PARAMS}
    cams = views()
    bg = torch.zeros(3, device='cuda')
    cache = DirectCache(cams, interface, interface_deform, PIPE, bg)

    # Ground truth: bright objects on the camera side of the glass, seen only as reflections.
    truth_positions = [[0.5, 0.3, -1.0], [-0.6, -0.2, -1.5], [0.1, -0.5, -0.6]]
    truth = blobs(truth_positions, [3.0, 2.0, 1.0])
    with torch.no_grad():
        for cam in cams:
            direct, glass_map = cache(cam)
            cam.original_image = render_composite(cam, direct, glass_map, truth, PIPE, NORMAL, OFFSET, blend=BLEND)['composite']
            assert (cam.original_image - direct).abs().max() > 0.05  # the reflection is really in the target

    # Start from the right places but with the wrong appearance, and fit.
    reflection = blobs(truth_positions, [0.5, 0.5, 0.5])
    reflection._opacity.data.fill_(-1.0)
    reflection.training_setup(opt)
    for group in reflection.optimizer.param_groups:
        if group['name'] == 'f_dc':
            group['lr'] = 0.05

    def mean_psnr():
        with torch.no_grad():
            values = []
            for cam in cams:
                direct, glass_map = cache(cam)
                out = render_composite(cam, direct, glass_map, reflection, PIPE, NORMAL, OFFSET, blend=BLEND)
                values.append(psnr(out['composite'], cam.original_image).mean().item())
        return sum(values) / len(values)

    before = mean_psnr()
    for step in range(300):
        cam = cams[step % len(cams)]
        direct, glass_map = cache(cam)
        out = render_composite(cam, direct, glass_map, reflection, PIPE, NORMAL, OFFSET, blend=BLEND)
        loss = sum(reflection_losses(out, cam, opt).values())
        loss.backward()
        reflection.optimizer.step()
        reflection.optimizer.zero_grad(set_to_none=True)
    after = mean_psnr()

    assert after > before + 5.0, (before, after)
    for name in interface.PARAMS:
        assert torch.equal(getattr(interface, '_' + name), interface_before[name]), name


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.data
def test_short_training_run_keeps_phase_one_frozen_and_saves(trained_reflection, trained_interface):
    root = trained_interface.root
    before, history = trained_reflection.interface_digests, trained_reflection.history

    # Phase I files are byte-identical after Phase II.
    interface = root / 'interface'
    after = {str(p.relative_to(root)): file_digest(p) for p in sorted(interface.rglob('*')) if p.suffix in ('.ply', '.pth')}
    assert after == before and len(before) >= 2

    final = next(h for h in history if h['split'] == 'test' and h['iteration'] == trained_reflection.iterations)
    assert all(torch.isfinite(torch.tensor(v)) for k, v in final.items() if k not in ('iteration', 'split'))
    assert final['psnr'] > final['psnr_direct'] - 1.0  # the reflection never wrecks what the interface already explains

    reflection = root / 'reflection'
    assert (reflection / 'cfg_args').exists()
    assert (reflection / 'point_cloud' / f'iteration_{trained_reflection.iterations}' / 'point_cloud.ply').exists()
    assert (reflection / 'deform' / f'iteration_{trained_reflection.iterations}' / 'deform.pth').exists()
    assert (reflection / 'blend' / f'iteration_{trained_reflection.iterations}' / 'blend.pth').exists()


@pytest.mark.data
def test_saved_reflection_loads_back(trained_reflection, trained_interface):
    from utils.vigor_utils import load_reflection

    gaussians, deform, blend, iteration = load_reflection(trained_interface.dataset)
    assert iteration == trained_reflection.iterations
    assert gaussians.num_points > 100 and gaussians._scaling.shape[1] == 3
    assert not blend.is_trainable  # pure Fresnel mixing has no parameters
    assert deform.network.gaussian_scaling.out_features == 3
