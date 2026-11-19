"""CPU checks for finite supports, ray conventions and camera calibration."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.calibration_utils import build_remap, camera_calibration, distort_points
from utils.glass_geometry import dynamic_virtual_camera, first_hit, validate_patches
from utils.point_utils import camera_rays, depth_to_normal_sobel


def camera(width=17, height=13):
    return SimpleNamespace(image_width=width, image_height=height,
        world_view_transform=torch.eye(4), camera_center=torch.zeros(3),
        FoVx=1.0, FoVy=0.8, fx=20., fy=20., cx=width/2, cy=height/2,
        znear=0.01, time=0.)


def square(z, radius=1.):
    return dict(normal=[0, 0, 1], offset=-z,
        vertices=[[-radius, -radius, z], [radius, -radius, z],
                  [radius, radius, z], [-radius, radius, z]])


class GlassGeometryTests(unittest.TestCase):
    def test_nearest_hit_and_reflection(self):
        view = camera()
        patches = validate_patches([square(2.), square(1.5)], device='cpu')
        geometry = first_hit(view, patches)
        self.assertTrue((geometry['patches'] == 1).all())
        torch.testing.assert_close(geometry['hits'][..., 2], torch.full((13, 17), 1.5))
        expected = geometry['real_directions'].clone()
        expected[..., 2] *= -1
        torch.testing.assert_close(geometry['directions'], expected)
        torch.testing.assert_close(geometry['origins'], geometry['hits'] + 1e-4 * expected)

    def test_finite_support_and_hits_behind_camera(self):
        view = camera()
        patches = validate_patches([square(-1.), square(1., .1)], device='cpu')
        geometry = first_hit(view, patches)
        self.assertEqual(int(geometry['patches'][6, 8]), 1)
        self.assertEqual(int(geometry['patches'][0, 0]), -1)
        self.assertFalse(geometry['visible'][0, 0])
        self.assertIsNone(dynamic_virtual_camera(view, geometry, 0))

    def test_virtual_camera_is_a_proper_rotation(self):
        view = camera()
        geometry = first_hit(view, validate_patches([square(1.)], device='cpu'))
        pose = dynamic_virtual_camera(view, geometry, 0)
        rotation = pose['rotation']
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(float(torch.linalg.det(rotation)), 1., places=6)
        x, y = pose['reference_pixel']
        torch.testing.assert_close(rotation[:, 2], geometry['directions'][y, x])

    def test_invalid_polygons_are_rejected(self):
        bad = [dict(normal=[0, 0, 1], offset=0., vertices=[[0, 0, 0]] * 3),
               dict(normal=[0, 0, 0], offset=0., vertices=square(0.)['vertices']),
               dict(normal=[0, 0, 1], offset=0., vertices=square(1.)['vertices'])]
        for record in bad:
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_patches([record], device='cpu')
        with self.assertRaises(ValueError):
            validate_patches([], device='cpu')

    def test_pixel_centers_and_planar_depth_normals(self):
        view = camera()
        origin, rays = camera_rays(view)
        torch.testing.assert_close(origin, torch.zeros(3))
        torch.testing.assert_close(rays[6, 8], torch.tensor([0., 0., 1.]))
        normals = depth_to_normal_sobel(view, torch.ones(13, 17, 1))
        expected = torch.tensor([0., 0., -1.]).expand(11, 15, 3)
        torch.testing.assert_close(normals[1:-1, 1:-1], expected)

    def test_undistorted_remap_preserves_pixel_indices(self):
        intr = SimpleNamespace(model='PINHOLE', params=[20., 21., 8.5, 6.5])
        matrix, distortion = camera_calibration(intr)
        np.testing.assert_array_equal(distortion, np.zeros(5))
        target = dict(fx=matrix[0, 0], fy=matrix[1, 1], cx=8.5, cy=6.5,
                      width=17, height=13)
        params = [20., 21., 8.5, 6.5, 0., 0.]
        points = np.array([[0., 0.], [8., 6.], [16., 12.]])
        np.testing.assert_allclose(distort_points(points, target, params), points, atol=1e-12)
        x, y = build_remap(target, params)
        np.testing.assert_allclose(x, np.tile(np.arange(17), (13, 1)), atol=1e-6)
        np.testing.assert_allclose(y, np.tile(np.arange(13)[:, None], (1, 17)), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
