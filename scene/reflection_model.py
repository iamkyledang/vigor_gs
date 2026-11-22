"""
Reflection Gaussian field G_R: 3D Gaussian ellipsoids (mu, q, s^(3), o, a).

G_R holds the reflected scene in real world coordinates, i.e. on the camera side of the glass. It is rendered
through the virtual (mirrored) camera, never mirrored itself. It shares nothing with the interface field.
"""

import torch

from scene.gaussian_field import GaussianField


class ReflectionGaussianModel(GaussianField):
    SCALE_DIM = 3
    PARAMS = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')
    INIT_OPACITY = 0.1

    def create_random(self, normal, offset, camera_centers, extent, num_points=100_000, radius_scale=3.0, color=0.5, seed=0):
        """
        Initialize without SfM points: the reflected scene has none.

        Points are drawn uniformly from a ball of radius `radius_scale * extent` around the cameras and kept only
        on the camera side of the glass (n^T x + d > 0), where a reflected object can be.
        """
        generator = torch.Generator().manual_seed(seed)
        center = camera_centers.mean(dim=0).cpu()
        normal_cpu, radius = normal.cpu(), radius_scale * extent

        points = torch.zeros((0, 3))
        while points.shape[0] < num_points:
            directions = torch.nn.functional.normalize(torch.randn(2 * num_points, 3, generator=generator), dim=-1)
            radii = radius * torch.rand(2 * num_points, 1, generator=generator) ** (1.0 / 3.0)
            candidates = center + directions * radii
            points = torch.cat([points, candidates[candidates @ normal_cpu + offset > 0]])
        points = points[:num_points]

        self.create_from_points(points, torch.full((num_points, 3), float(color)), spatial_lr_scale=extent)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        self.densify(max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
