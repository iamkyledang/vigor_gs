"""
Interface Gaussian field G_I: 2D Gaussian disks (mu, q, s^(2), p, o, a).

Each disk carries two separate scalars:
    occupancy p  - geometric presence, used alone in the surface pass (recovers the glass surface),
    opacity   o  - optical contribution given presence; the volume pass renders with p * o.

Densification and pruning follow the occupancy, as in RT-Splatting.
"""

import torch

from scene.gaussian_field import GaussianField
from utils.general_utils import inverse_sigmoid


class InterfaceGaussianModel(GaussianField):
    SCALE_DIM = 2
    PARAMS = ('xyz', 'f_dc', 'f_rest', 'occupancy', 'opacity', 'scaling', 'rotation')
    INIT_OPACITY = 0.5
    INIT_OCCUPANCY = 0.1
    INIT_SCALE_FACTOR = 0.1

    def __init__(self, sh_degree: int, factorized: bool = True):
        """
        factorized=False is the ablation without the p * o factorization: every disk has p = 1, so a single
        opacity has to play both roles and both passes render the same thing.
        """
        super().__init__(sh_degree)
        self.factorized = factorized
        self.occupancy_activation = torch.sigmoid
        self.inverse_occupancy_activation = inverse_sigmoid

    @property
    def get_occupancy(self):
        if not self.factorized:
            return torch.ones_like(self._occupancy)
        return self.occupancy_activation(self._occupancy)

    @property
    def get_effective_opacity(self):
        """p * o, the opacity of the volume pass."""
        return self.get_occupancy * self.get_opacity

    def _initial_rotation(self, num_points):
        return torch.rand((num_points, 4), device='cuda')

    def _initial_extra_params(self, num_points):
        return {'occupancy': self.inverse_occupancy_activation(self.INIT_OCCUPANCY * torch.ones((num_points, 1), device='cuda'))}

    def _learning_rates(self, training_args):
        lrs = super()._learning_rates(training_args)
        lrs['occupancy'] = training_args.occupancy_lr
        return lrs

    def densify_and_prune(self, max_grad, min_occupancy, extent, max_screen_size, last_reset_iter):
        self.densify(max_grad, extent)

        presence = self.get_occupancy if self.factorized else self.get_opacity
        prune_mask = (presence < min_occupancy).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # Do not prune a Gaussian that has not been observed since the last occupancy reset.
        prune_mask = torch.logical_and(prune_mask, self.last_update.squeeze(-1) > last_reset_iter)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def reset_occupancy(self):
        self._reset_activated('occupancy', self.occupancy_activation, self.inverse_occupancy_activation)
