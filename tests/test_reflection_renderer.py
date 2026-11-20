"""Reflected-ray rendering of G_R (Sec. 3): ray modes, tile candidates, CUDA renderer, Fresnel blend."""

from types import SimpleNamespace

import pytest
import torch

from gaussian_renderer.composition import ConstantReflectance, OpticalCorrection, reflection_weight, schlick
from gaussian_renderer.reflection import ray_bundle, tile_candidates
from tests.test_glass_geometry import square, view
from utils.glass_geometry import first_hit, mirror_points, validate_patches

PIPE = SimpleNamespace(tile_size=4, ray_cutoff=3.5, max_tile_entries=1_000_000, ray_epsilon=1e-4)


def scene_geometry(device='cpu'):
    cam = view(center=(0.4, 0.1, 3.0), width=17, height=13, focal=20.0)
    cam.znear = 0.01
    if device == 'cuda':
        cam.world_view_transform = cam.world_view_transform.cuda()
        cam.camera_center = cam.camera_center.cuda()
    patches = validate_patches([square(0.0, half=1.0, cx=-0.9), {'normal': [0.2, 0.0, 1.0], 'offset': -0.3,
                                'vertices': [[0.2, -1, 0.26], [1.6, -1, -0.02], [1.6, 1, -0.02], [0.2, 1, 0.26]]}], device=device)
    return cam, patches, first_hit(cam, patches)


def intersected(origin, direction, means, scales):
    """Gaussians whose closest approach to a ray (axis-aligned ellipsoids) lies ahead of it within the cutoff."""
    v, w = (means - origin) / scales, direction / scales
    distance = (v * w).sum(-1) / w.square().sum(-1).clamp_min(1e-20)
    q = (distance[:, None] * w - v).square().sum(-1)
    return ((distance > 0) & (q <= PIPE.ray_cutoff ** 2)).nonzero().flatten()


@pytest.mark.parametrize('mode', ['exact', 'pinhole', 'unreflected'])
def test_tile_candidates_cover_every_intersected_gaussian(mode):
    cam, patches, geometry = scene_geometry()
    bundle = ray_bundle(cam, geometry, patches, mode)
    generator = torch.Generator().manual_seed(3)
    means = torch.rand(60, 3, generator=generator) * torch.tensor([3.0, 2.0, 2.5]) - torch.tensor([1.5, 1.0, 0.2 if mode != 'unreflected' else 2.5])
    scales = 0.02 + torch.rand(60, 3, generator=generator) * 0.1
    entries, ranges = tile_candidates(cam, means, scales, bundle, patches, PIPE)
    covered = 0
    for y in range(13):
        for x in range(17):
            k = int(bundle['patch'][y, x])
            if k < 0:
                continue
            hit = intersected(bundle['origins'][y, x], bundle['directions'][y, x], means, scales)
            tile = (y // 4) * 5 + x // 4
            listed = set(entries[ranges[tile]:ranges[tile + 1]].tolist())
            assert {int(i) + k * 60 for i in hit} <= listed
            covered += len(hit)
    assert covered > 0


def test_ray_modes_share_directions_but_not_origins():
    cam, patches, geometry = scene_geometry()
    exact, pinhole, straight = (ray_bundle(cam, geometry, patches, mode) for mode in ('exact', 'pinhole', 'unreflected'))
    glass = geometry['visible']
    assert torch.allclose(exact['directions'][glass], pinhole['directions'][glass])
    assert torch.allclose(exact['origins'][glass], geometry['hits'][glass] + 1e-4 * geometry['reflected'][glass])
    for k, pose in enumerate(pinhole['poses']):
        region = geometry['patch'] == k
        assert torch.allclose(pinhole['origins'][region], pose['center'].expand(int(region.sum()), 3))
    assert torch.allclose(straight['directions'][glass], geometry['real_directions'][glass])
    assert torch.allclose(exact['centers'][1], mirror_points(cam.camera_center, patches[1]['normal'], patches[1]['offset']))


def test_tile_budget_and_empty_field():
    cam, patches, geometry = scene_geometry()
    bundle = ray_bundle(cam, geometry, patches)
    means, scales = torch.zeros(5, 3) + torch.tensor([0.0, 0.0, 1.0]), torch.full((5, 3), 0.3)
    with pytest.raises(RuntimeError, match='budget exceeded'):
        tile_candidates(cam, means, scales, bundle, patches, SimpleNamespace(**{**vars(PIPE), 'max_tile_entries': 1}))
    entries, ranges = tile_candidates(cam, means[:0], scales[:0], bundle, patches, PIPE)
    assert entries.numel() == 0 and ranges.numel() == 21 and (ranges == 0).all()


def test_fresnel_weight_is_zero_off_glass_and_starts_as_pure_schlick():
    cam, patches, geometry = scene_geometry()
    params = SimpleNamespace(eta_air=1.0, eta_glass=1.5)
    optical = OpticalCorrection(torch.zeros(3), 2.0)
    alpha, h = reflection_weight(geometry, SimpleNamespace(time=0.5), params, optical)
    expected = torch.where(geometry['visible'], schlick(geometry['cosine'], 1.0, 1.5), torch.zeros_like(geometry['cosine']))
    torch.testing.assert_close(alpha[0], expected)
    assert torch.all(h == 0)
    assert schlick(torch.tensor(1.0), 1.0, 1.5).item() == pytest.approx(0.04)
    assert schlick(torch.tensor(0.0), 1.0, 1.5).item() == pytest.approx(1.0)
    # The correction is learnable and only acts on the glass.
    alpha.sum().backward()
    assert optical.net[-1].weight.grad.abs().sum() > 0
    constant = ConstantReflectance(0.2)
    alpha, _ = reflection_weight(geometry, SimpleNamespace(time=0.5), params, None, constant)
    assert torch.allclose(alpha[0][geometry['visible']], torch.full_like(alpha[0][geometry['visible']], 0.2))
    assert torch.all(alpha[0][~geometry['visible']] == 0)


def torch_reference(origins, directions, patches, means, axes, opacity, colors, entries, width, height):
    """Scalar front-to-back integration, deliberately independent of the CUDA code."""
    pixels, count = [], len(means)
    for pixel in range(width * height):
        value, transmittance = means.new_zeros(3), means.new_tensor(1.0)
        for entry in entries.tolist():
            if int(patches[pixel]) != entry // count:
                continue
            index = entry % count
            v = (means[index] - origins[pixel]) @ axes[index]
            w = directions[pixel] @ axes[index]
            distance = (v * w).sum() / w.square().sum()
            q = (distance * w - v).square().sum()
            alpha = (opacity[index, 0] * torch.exp(-0.5 * q)).clamp_max(0.99)
            if float(distance) <= 0 or float(q) > 3.5 ** 2 or float(alpha) < 1 / 255:
                continue
            value = value + transmittance * alpha * colors[entry]
            transmittance = transmittance * (1 - alpha)
            if float(transmittance) < 1e-4:
                break
        pixels.append(value)
    return torch.stack(pixels).reshape(height, width, 3).permute(2, 0, 1)


@pytest.mark.gpu
class TestCudaRenderer:
    def setup_method(self):
        import gaussian_renderer  # noqa: F401  (puts this checkout's extension first on sys.path)
        from diff_surfel_anych import render_reflected_rays

        self.render = render_reflected_rays
        self.width, self.height = 5, 3
        self.origins = torch.zeros(15, 3, device='cuda')
        self.origins[:, 0] = torch.linspace(-0.3, 0.3, 15, device='cuda')
        self.directions = torch.tensor([0.0, 0.0, 1.0], device='cuda').expand(15, 3).contiguous()
        self.patches = torch.tensor([0, 1, -1, 0, 1] * 3, device='cuda', dtype=torch.int32)
        self.means = torch.tensor([[0.1, 0.1, 1.0], [-0.1, -0.1, 2.0], [0.0, 0.0, -0.5]], device='cuda', requires_grad=True)
        self.axes = torch.tensor([[[2.0, 0.1, 0.0], [0.2, 3.0, 0.1], [0.0, 0.2, 1.5]]] * 3, device='cuda', requires_grad=True)
        self.opacity = torch.tensor([[0.3], [0.5], [0.4]], device='cuda', requires_grad=True)
        self.colors = torch.linspace(0.1, 1.8, 18, device='cuda').reshape(6, 3).requires_grad_()
        self.entries = torch.tensor([0, 3, 1, 4, 2, 5], device='cuda', dtype=torch.int64)
        self.ranges = torch.tensor([0, 6], device='cuda', dtype=torch.int64)

    def forward(self, entries=None, ranges=None):
        return self.render(self.origins, self.directions, self.patches, self.means, self.axes, self.opacity, self.colors,
                           self.entries if entries is None else entries, self.ranges if ranges is None else ranges, self.width, self.height, 16, 3.5)

    def test_forward_and_gradients_match_the_reference(self):
        actual = self.forward()
        expected = torch_reference(self.origins, self.directions, self.patches, self.means, self.axes, self.opacity, self.colors, self.entries, self.width, self.height)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        weights = torch.linspace(0.1, 1.0, actual.numel(), device='cuda').reshape_as(actual)
        params = (self.means, self.axes, self.opacity, self.colors)
        for result, reference in zip(torch.autograd.grad((actual * weights).sum(), params), torch.autograd.grad((expected * weights).sum(), params)):
            torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)

    def test_empty_bins_render_black_with_zero_gradients(self):
        actual = self.forward(self.entries[:0], torch.zeros_like(self.ranges))
        assert torch.all(actual == 0)
        for gradient in torch.autograd.grad(actual.sum(), (self.means, self.axes, self.opacity, self.colors)):
            assert torch.all(gradient == 0)

    def test_nondefault_cuda_stream(self):
        expected = self.forward()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = self.forward()
            gradients = torch.autograd.grad(actual.sum(), (self.means, self.axes))
        stream.synchronize()
        torch.testing.assert_close(actual, expected)
        assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['exact', 'pinhole'])
def test_a_gaussian_in_front_of_the_glass_appears_at_its_mirror_image(mode):
    """Exact rays put a point P where the real camera sees mirror(P); the V_dyn pinhole shifts it (parallax)."""
    from gaussian_renderer.reflection import render_reflection
    from scene.deformation import Deformation
    from scene.gaussian_model import ReflectionField

    cam = view(center=(0.3, 0.0, 2.0), width=65, height=49, focal=60.0)
    cam.znear = 0.01
    cam.world_view_transform, cam.camera_center = cam.world_view_transform.cuda(), cam.camera_center.cuda()
    cam.time = 0.0
    patches = validate_patches([square(0.0, half=3.0)])
    geometry = first_hit(cam, patches)
    field = ReflectionField(0)
    point = torch.tensor([[-0.2, 0.15, 0.4]], device='cuda')  # in front of the glass, close to it
    field.create_from_points(point, torch.ones(1, 3), 1.0)
    with torch.no_grad():
        field._scaling[:] = -4.0
        field._opacity[:] = 4.0
    deformation = Deformation(3, SimpleNamespace(deform_xyz_freqs=1, deform_time_freqs=1, deform_depth=1, deform_width=4), static=True).cuda()
    pipe = SimpleNamespace(tile_size=16, ray_cutoff=3.5, max_tile_entries=1_000_000)
    image = render_reflection(cam, field, deformation, ray_bundle(cam, geometry, patches, mode), patches, pipe)['reflection']
    brightest = torch.nonzero(image[0] == image[0].max())[0].float()
    mirrored = mirror_points(point[0], patches[0]['normal'], patches[0]['offset'])
    camera = mirrored @ cam.world_view_transform[:3, :3] + cam.world_view_transform[3, :3]
    expected = torch.stack([camera[1] / camera[2] * cam.fy + cam.cy - 0.5, camera[0] / camera[2] * cam.fx + cam.cx - 0.5])
    error = (brightest - expected).norm().item()
    if mode == 'exact':
        assert image.max() > 0.5 and error <= 1.0
    else:
        assert error > 2.0  # the pinhole at the glass sees a near object from the wrong place
