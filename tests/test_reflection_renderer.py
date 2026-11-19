"""Conservative CPU routing and CUDA gradients against an independent reference."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_renderer.composition import OpticalCorrection, render_composite
from gaussian_renderer.reflection import tile_candidates
from utils.glass_geometry import first_hit, validate_patches


class TileRoutingTests(unittest.TestCase):
    def setUp(self):
        self.view = SimpleNamespace(image_width=17, image_height=13,
            world_view_transform=torch.eye(4), camera_center=torch.zeros(3),
            FoVx=1., FoVy=.8, fx=20., fy=20., cx=8.5, cy=6.5, znear=.01)
        self.patches = validate_patches([dict(normal=[0, 0, 1], offset=-1.5,
            vertices=[[-2, -2, 1.5], [2, -2, 1.5], [2, 2, 1.5], [-2, 2, 1.5]])], device='cpu')
        self.geometry = first_hit(self.view, self.patches)
        generator = torch.Generator().manual_seed(8)
        self.means = torch.randn(20, 3, generator=generator) * .25
        self.means[:, 2] = torch.rand(20, generator=generator) * 1.3
        self.scales = .02 + torch.rand(20, 3, generator=generator) * .15
        self.pipe = SimpleNamespace(tile_size=4, ray_cutoff=3.5, max_tile_entries=100000)

    def test_bins_cover_every_intersected_gaussian(self):
        entries, ranges = tile_candidates(self.view, self.means, self.scales,
                                         self.patches, self.geometry, self.pipe)
        covered = 0
        for y in range(13):
            for x in range(17):
                origin = self.geometry['origins'][y, x]
                direction = self.geometry['directions'][y, x]
                v = (self.means - origin) / self.scales
                w = direction / self.scales
                distance = (v*w).sum(-1) / w.square().sum(-1)
                q = (distance[:, None]*w-v).square().sum(-1)
                intersected = ((distance > 0) & (q <= 3.5**2)).nonzero().flatten()
                tile_id = (y//4)*5+x//4
                candidates = set(entries[ranges[tile_id]:ranges[tile_id+1]].tolist())
                self.assertTrue(set(intersected.tolist()).issubset(candidates))
                covered += len(intersected)
        self.assertGreater(covered, 0)

    def test_budget_and_empty_candidates(self):
        self.pipe.max_tile_entries = 1
        with self.assertRaisesRegex(RuntimeError, 'budget exceeded'):
            tile_candidates(self.view, self.means, self.scales, self.patches, self.geometry, self.pipe)
        self.pipe.max_tile_entries = 100000
        entries, ranges = tile_candidates(self.view, self.means[:0], self.scales[:0],
                                         self.patches, self.geometry, self.pipe)
        self.assertEqual(entries.numel(), 0)
        self.assertEqual(ranges.numel(), 21)
        self.assertTrue((ranges == 0).all())


class CompositionTests(unittest.TestCase):
    def test_frozen_direct_and_zero_weight_outside_glass(self):
        direct = torch.full((3, 2, 2), .25)
        reflection = torch.full((3, 2, 2), .75, requires_grad=True)
        visible = torch.tensor([[True, False], [True, False]])
        geometry = dict(visible=visible, fresnel=visible.float()*.04,
                        cosine=torch.ones(2, 2))
        view = SimpleNamespace(image_height=2, image_width=2, time=.5)
        args = SimpleNamespace(pure_fresnel=False)
        optical = OpticalCorrection()
        with patch('gaussian_renderer.composition.render_reflection', return_value={'reflection': reflection}):
            out = render_composite(view, None, None, None, None, [], optical,
                                   args, None, direct=direct, geometry=geometry)
        torch.testing.assert_close(out['composite'][:, :, 1], direct[:, :, 1])
        torch.testing.assert_close(out['alpha'][0], visible.float()*.04)
        out['composite'].sum().backward()
        torch.testing.assert_close(reflection.grad, out['alpha'].expand_as(reflection))
        self.assertTrue(any(p.grad is not None for p in optical.parameters()))
        self.assertIsNone(direct.grad)


class ComparisonVideoTests(unittest.TestCase):
    def test_encoded_video_has_ours_left_gt_right_and_preserves_frame_order(self):
        import tempfile
        from pathlib import Path
        import cv2
        import numpy as np
        from PIL import Image
        from render import create_comparison_video

        with tempfile.TemporaryDirectory(prefix='vigor-comparison-') as directory:
            root=Path(directory)
            for sub in ('renders','gt'): (root/sub).mkdir()
            # Odd dimensions exercise codec padding; the second frame is smaller.
            for name,size,ours,gt in [('later',(25,33),(0,255,0),(255,255,0)),
                                      ('earlier',(21,29),(255,0,0),(0,0,255))]:
                for sub,color in [('renders',ours),('gt',gt)]:
                    Image.fromarray(np.full((*size,3),color,dtype=np.uint8)).save(root/sub/(name+'.png'))
            path=create_comparison_video(root,['earlier','later'],fps=12.)
            capture=cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS),12.,places=2)
                for left,right in [((0,0,255),(255,0,0)),((0,255,0),(0,255,255))]:
                    ok,frame=capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape,(66,66,3))
                    # OpenCV decoding is BGR; allow lossy MPEG-4 quantization.
                    np.testing.assert_allclose(frame[52,16],left,atol=15)
                    np.testing.assert_allclose(frame[52,49],right,atol=15)
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()

    def test_invalid_video_inputs_are_rejected(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from render import create_comparison_video

        with tempfile.TemporaryDirectory(prefix='vigor-comparison-') as directory:
            root=Path(directory)
            for fps in (0.,-1.,float('nan'),float('inf')):
                with self.subTest(fps=fps), self.assertRaises(ValueError):
                    create_comparison_video(root,['frame'],fps)
            with self.assertRaises(ValueError): create_comparison_video(root,[])
            for sub in ('renders','gt'): (root/sub).mkdir()
            Image.new('RGB',(32,24)).save(root/'renders/frame.png')
            Image.new('RGB',(32,26)).save(root/'gt/frame.png')
            with self.assertRaisesRegex(ValueError,'size mismatch'):
                create_comparison_video(root,['frame'])


def torch_reference(origins, directions, patches, means, axes, opacity, colors, entries, width, height):
    """Scalar front-to-back integration, deliberately independent of CUDA code."""
    pixels = []
    count = len(means)
    for pixel in range(width*height):
        value = means.new_zeros(3)
        transmittance = means.new_tensor(1.)
        for entry in entries.tolist():
            if int(patches[pixel]) != entry//count:
                continue
            index = entry % count
            v = (means[index]-origins[pixel]) @ axes[index]
            w = directions[pixel] @ axes[index]
            distance = (v*w).sum()/w.square().sum()
            q = (distance*w-v).square().sum()
            alpha = (opacity[index, 0]*torch.exp(-.5*q)).clamp_max(.99)
            if float(distance) <= 0 or float(q) > 3.5**2 or float(alpha) < 1/255:
                continue
            value = value + transmittance*alpha*colors[entry]
            transmittance = transmittance*(1-alpha)
            if float(transmittance) < 1e-4:
                break
        pixels.append(value)
    return torch.stack(pixels).reshape(height, width, 3).permute(2, 0, 1)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class CudaReflectionTests(unittest.TestCase):
    def setUp(self):
        from diff_surfel_anych import render_reflected_rays
        self.render = render_reflected_rays
        self.width, self.height = 5, 3
        self.origins = torch.zeros(15, 3, device='cuda')
        self.origins[:, 0] = torch.linspace(-.3, .3, 15, device='cuda')
        self.directions = torch.tensor([0., 0., 1.], device='cuda').expand(15, 3).contiguous()
        self.patches = torch.tensor([0, 1, -1, 0, 1]*3, device='cuda', dtype=torch.int32)
        self.means = torch.tensor([[.1, .1, 1.], [-.1, -.1, 2.], [0., 0., -.5]], device='cuda', requires_grad=True)
        self.axes = torch.tensor([[[2., .1, 0.], [.2, 3., .1], [0., .2, 1.5]]]*3, device='cuda', requires_grad=True)
        self.opacity = torch.tensor([[.3], [.5], [.4]], device='cuda', requires_grad=True)
        self.colors = torch.linspace(.1, 1.8, 18, device='cuda').reshape(6, 3).requires_grad_()
        self.entries = torch.tensor([0, 3, 1, 4, 2, 5], device='cuda', dtype=torch.int64)
        self.ranges = torch.tensor([0, 6], device='cuda', dtype=torch.int64)

    def forward(self, entries=None, ranges=None):
        return self.render(self.origins, self.directions, self.patches, self.means,
            self.axes, self.opacity, self.colors,
            self.entries if entries is None else entries,
            self.ranges if ranges is None else ranges, self.width, self.height, 16, 3.5)

    def test_forward_and_parameter_gradients_match_reference(self):
        actual = self.forward()
        expected = torch_reference(self.origins, self.directions, self.patches, self.means,
            self.axes, self.opacity, self.colors, self.entries, self.width, self.height)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        weights = torch.linspace(.1, 1., actual.numel(), device='cuda').reshape_as(actual)
        params = (self.means, self.axes, self.opacity, self.colors)
        actual_grad = torch.autograd.grad((actual*weights).sum(), params)
        expected_grad = torch.autograd.grad((expected*weights).sum(), params)
        for name, result, reference in zip(('means', 'axes', 'opacity', 'colors'), actual_grad, expected_grad):
            with self.subTest(parameter=name):
                torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)

    def test_empty_bins_return_black_and_zero_gradients(self):
        actual = self.forward(self.entries[:0], torch.zeros_like(self.ranges))
        torch.testing.assert_close(actual, torch.zeros_like(actual))
        gradients = torch.autograd.grad(actual.sum(), (self.means, self.axes, self.opacity, self.colors))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_nondefault_cuda_stream(self):
        expected = self.forward()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = self.forward()
            gradients = torch.autograd.grad(actual.sum(), (self.means, self.axes))
        stream.synchronize()
        torch.testing.assert_close(actual, expected)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == '__main__':
    unittest.main()
