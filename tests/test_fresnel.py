"""Module 08: Fresnel-guided blending of the direct and the reflection image."""

import math

import pytest
import torch

from tests.helpers import look_at_camera
from utils.fresnel import OpticalCorrection, composite, fresnel_weight, glass_geometry, optical_loss, schlick_f0, schlick_reflectance
from utils.point_utils import camera_rays

pytestmark = pytest.mark.gpu  # per-pixel rays are built on the GPU

GLASS_NORMAL = torch.tensor([0.0, 0.0, -1.0])  # glass at z = 1, normal towards cameras at negative z
GLASS_OFFSET = 1.0


def test_schlick_reflectance_limits():
    f0 = schlick_f0(1.0, 1.5)
    assert f0 == pytest.approx(0.04)
    cos = torch.tensor([1.0, 0.5, 0.0])
    reflectance = schlick_reflectance(cos, f0)
    assert reflectance[0].item() == pytest.approx(f0)  # head-on: almost everything is transmitted
    assert reflectance[2].item() == pytest.approx(1.0)  # grazing: almost everything is reflected
    assert reflectance[1].item() == pytest.approx(f0 + (1 - f0) * 0.5**5)
    assert (reflectance[:-1] < reflectance[1:]).all()
    assert schlick_f0(1.0, 1.0) == 0.0  # no index change, no reflection


def test_cosine_matches_the_definition_through_the_glass_point():
    cam = look_at_camera(eye=(0.4, -0.3, -2.0), target=(0.0, 0.2, 1.0))
    normal = torch.nn.functional.normalize(torch.tensor([0.2, 0.1, -1.0]), dim=0).cuda()
    cos_theta, glass_points = glass_geometry(cam, normal, GLASS_OFFSET)

    assert cos_theta.shape == (1, cam.image_height, cam.image_width)
    assert torch.allclose(glass_points @ normal + GLASS_OFFSET, torch.zeros_like(cos_theta[0]), atol=1e-4)  # on the glass

    # cos(theta) = |n^T (c - x)| / |c - x|
    to_camera = cam.camera_center - glass_points
    expected = (to_camera @ normal).abs() / to_camera.norm(dim=-1)
    assert torch.allclose(cos_theta[0], expected, atol=1e-5)

    # each glass point lies on its own pixel ray
    rays_o, rays_d = camera_rays(cam)
    along = torch.nn.functional.normalize(glass_points - rays_o, dim=-1)
    assert torch.allclose(along, torch.nn.functional.normalize(rays_d, dim=-1), atol=1e-4)


def test_head_on_view_reflects_little_and_grazing_view_reflects_a_lot():
    head_on = look_at_camera(eye=(0.0, 0.0, -2.0), target=(0.0, 0.0, 1.0), fov_deg=60.0)
    weight = fresnel_weight(head_on, GLASS_NORMAL, GLASS_OFFSET)
    H, W = head_on.image_height, head_on.image_width
    center = weight['alpha'][0, H // 2, W // 2].item()
    assert center == pytest.approx(schlick_f0(), abs=1e-3)
    assert weight['alpha'].max() < 0.05
    assert weight['alpha'][0, H // 2, 0] > weight['alpha'][0, H // 2, W // 2]  # more oblique towards the edge

    grazing = look_at_camera(eye=(-6.0, 0.0, 0.9), target=(0.0, 0.0, 0.98), fov_deg=20.0)
    weight = fresnel_weight(grazing, GLASS_NORMAL, GLASS_OFFSET)
    assert weight['cos_theta'].max() < math.cos(math.radians(75))
    assert weight['alpha'].mean() > 0.5


def test_glass_map_confines_the_reflection():
    cam = look_at_camera(eye=(1.5, 0.0, -1.0), target=(0.0, 0.0, 1.0))
    H, W = cam.image_height, cam.image_width
    glass_map = torch.zeros(1, H, W, device='cuda')
    glass_map[:, :, : W // 2] = 1.0
    glass_map[:, :, W // 2 : W // 2 + 4] = 0.5

    free = fresnel_weight(cam, GLASS_NORMAL, GLASS_OFFSET)
    gated = fresnel_weight(cam, GLASS_NORMAL, GLASS_OFFSET, glass_map=glass_map)
    assert torch.equal(gated['alpha'][:, :, : W // 2], free['alpha'][:, :, : W // 2])
    assert torch.allclose(gated['alpha'][:, :, W // 2 : W // 2 + 4], 0.5 * free['alpha'][:, :, W // 2 : W // 2 + 4])
    assert torch.count_nonzero(gated['alpha'][:, :, W // 2 + 4 :]) == 0

    direct, reflection = torch.rand(3, H, W, device='cuda'), torch.rand(3, H, W, device='cuda')
    blended = composite(direct, reflection, gated['alpha'])
    assert torch.equal(blended[:, :, W // 2 + 4 :], direct[:, :, W // 2 + 4 :])  # off the glass: the direct image


def test_composite_is_a_linear_blend():
    direct = torch.full((3, 4, 5), 0.2)
    reflection = torch.full((3, 4, 5), 1.0)
    assert torch.equal(composite(direct, reflection, torch.zeros(1, 4, 5)), direct)
    assert torch.equal(composite(direct, reflection, torch.ones(1, 4, 5)), reflection)
    assert torch.allclose(composite(direct, reflection, torch.full((1, 4, 5), 0.25)), torch.full((3, 4, 5), 0.4))


def test_fresh_correction_is_pure_fresnel_and_trains():
    cam = look_at_camera(eye=(1.0, 0.3, -1.5), target=(0.0, 0.0, 1.0))
    cam.time = 0.4
    correction = OpticalCorrection(width=16, depth=2).cuda()

    pure = fresnel_weight(cam, GLASS_NORMAL, GLASS_OFFSET)
    corrected = fresnel_weight(cam, GLASS_NORMAL, GLASS_OFFSET, correction=correction)
    assert pure['h'] is None
    assert torch.count_nonzero(corrected['h']) == 0
    assert torch.equal(corrected['alpha'], pure['alpha'])
    assert optical_loss(corrected['h']).item() == 0.0

    target = (pure['alpha'] + 0.2).clamp(0, 1)
    optimizer = torch.optim.Adam(correction.parameters(), lr=1e-2)
    for _ in range(100):
        optimizer.zero_grad()
        out = fresnel_weight(cam, GLASS_NORMAL, GLASS_OFFSET, correction=correction)
        loss = (out['alpha'] - target).abs().mean()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
    assert optical_loss(out['h']).item() > 0.0
    assert out['alpha'].min() >= 0.0 and out['alpha'].max() <= 1.0


def test_alpha_stays_in_range_with_a_large_correction():
    cam = look_at_camera(eye=(1.0, 0.3, -1.5), target=(0.0, 0.0, 1.0))
    correction = OpticalCorrection(width=16, depth=2).cuda()
    torch.nn.init.constant_(correction.head.bias, 5.0)
    assert torch.equal(fresnel_weight(cam, GLASS_NORMAL, GLASS_OFFSET, correction=correction)['alpha'], torch.ones(1, cam.image_height, cam.image_width, device='cuda'))
    torch.nn.init.constant_(correction.head.bias, -5.0)
    assert torch.count_nonzero(fresnel_weight(cam, GLASS_NORMAL, GLASS_OFFSET, correction=correction)['alpha']) == 0
