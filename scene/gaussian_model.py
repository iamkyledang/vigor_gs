"""
The two canonical Gaussian fields of VIGOR-GS (Sec. 1):

    G_I^0 = {(mu_i, q_i, s_i^(2), p_i, o_i, a_i)}   InterfaceField: 2D disks, occupancy p and opacity o, rendered with p * o
    G_R^0 = {(mu_j, q_j, s_j^(3), o_j, a_j)}        ReflectionField: 3D ellipsoids, rendered with o

The fields have independent parameters and optimizers. `GaussianField` holds what does not depend on the primitive
type: parameter storage, the optimizer surgery that densification and pruning need, the 3DGS densification
schedule and persistence. Per-Gaussian parameters are listed once in `PARAMS`; a subclass that adds one extends the
tuple and provides its initial value and learning rate. Occupancy, opacity and appearance are canonical, so they
are time independent; only geometry is deformed (scene/deformation.py).

Derived from the 3DGS / 2DGS GaussianModel (Inria GRAPHDECO, see LICENSE.md).
"""

import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from utils.general_utils import build_rotation, get_expon_lr_func, inverse_sigmoid
from utils.sh_utils import RGB2SH


class GaussianField:
    SCALE_DIM = 3
    PARAMS = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')  # stored as self._<name>
    INIT_OPACITY = 0.1
    INIT_SCALE_FACTOR = 1.0  # multiplies the squared nearest-neighbor distance used for the initial scale

    def __init__(self, sh_degree: int, max_points: int = 1_000_000):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.max_points = max_points
        for name in self.PARAMS:
            setattr(self, '_' + name, torch.empty(0))
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0.0
        self.spatial_lr_scale = 0.0
        self.xyz_scheduler = None

    # ------------------------------------------------------------------ accessors

    @property
    def num_points(self):
        return self._xyz.shape[0]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._f_dc, self._f_rest), dim=1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_presence(self):
        """What densification prunes on and resets: the opacity here, the occupancy of the interface."""
        return self.get_opacity

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ------------------------------------------------------------------ initialization

    def _initial_rotation(self, num_points):
        rotation = torch.zeros((num_points, 4), device='cuda')
        rotation[:, 0] = 1
        return rotation

    def _initial_extra_params(self, num_points):
        return {}

    def create_from_points(self, points, colors, spatial_lr_scale: float):
        """points: (N, 3) world positions, colors: (N, 3) linear RGB (may exceed 1 for the reflection field)."""
        from simple_knn._C import distCUDA2

        self.spatial_lr_scale = spatial_lr_scale
        points = torch.as_tensor(points, dtype=torch.float32).cuda()
        colors = torch.as_tensor(colors, dtype=torch.float32).cuda()
        num_points = points.shape[0]
        features = torch.zeros((num_points, 3, (self.max_sh_degree + 1) ** 2), device='cuda')
        features[:, :, 0] = RGB2SH(colors)
        dist2 = torch.clamp_min(distCUDA2(points), 1e-7) * self.INIT_SCALE_FACTOR
        initial = {
            'xyz': points,
            'f_dc': features[:, :, 0:1].transpose(1, 2).contiguous(),
            'f_rest': features[:, :, 1:].transpose(1, 2).contiguous(),
            'opacity': inverse_sigmoid(self.INIT_OPACITY * torch.ones((num_points, 1), device='cuda')),
            'scaling': torch.log(torch.sqrt(dist2))[..., None].repeat(1, self.SCALE_DIM),
            'rotation': self._initial_rotation(num_points),
        }
        initial.update(self._initial_extra_params(num_points))
        for name in self.PARAMS:
            setattr(self, '_' + name, nn.Parameter(initial[name].requires_grad_(True)))
        self.max_radii2D = torch.zeros(num_points, device='cuda')

    # ------------------------------------------------------------------ optimization

    def _learning_rates(self, opt):
        return {
            'xyz': opt.position_lr_init * self.spatial_lr_scale,
            'f_dc': opt.feature_lr,
            'f_rest': opt.feature_lr / 20.0,
            'opacity': opt.opacity_lr,
            'scaling': opt.scaling_lr,
            'rotation': opt.rotation_lr,
        }

    def training_setup(self, opt):
        self.percent_dense = opt.percent_dense
        self._reset_densification_stats()
        rates = self._learning_rates(opt)
        groups = [{'params': [getattr(self, '_' + name)], 'lr': rates[name], 'name': name} for name in self.PARAMS]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(lr_init=opt.position_lr_init * self.spatial_lr_scale, lr_final=opt.position_lr_final * self.spatial_lr_scale,
                                               lr_delay_mult=opt.position_lr_delay_mult, max_steps=opt.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for group in self.optimizer.param_groups:
            if group['name'] == 'xyz':
                group['lr'] = self.xyz_scheduler(iteration)

    def freeze(self):
        """Turn the field into a constant: no gradients, no optimizer, no densification buffers."""
        for name in self.PARAMS:
            setattr(self, '_' + name, getattr(self, '_' + name).detach().requires_grad_(False))
        self.optimizer = None
        self.xyz_gradient_accum = self.denom = torch.empty(0)

    # ------------------------------------------------------------------ persistence

    def capture(self):
        return {
            'active_sh_degree': self.active_sh_degree,
            'spatial_lr_scale': self.spatial_lr_scale,
            'params': {name: getattr(self, '_' + name).detach() for name in self.PARAMS},
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
        }

    def restore(self, state, opt=None):
        """Load a captured state; with `opt`, also rebuild the optimizer and its moments (resume)."""
        self.active_sh_degree = state['active_sh_degree']
        self.spatial_lr_scale = state['spatial_lr_scale']
        for name in self.PARAMS:
            setattr(self, '_' + name, nn.Parameter(state['params'][name].cuda().requires_grad_(opt is not None)))
        self.max_radii2D = state['max_radii2D'].cuda()
        if opt is None:
            self.freeze()
            return
        self.training_setup(opt)
        self.xyz_gradient_accum = state['xyz_gradient_accum'].cuda()
        self.denom = state['denom'].cuda()
        if state.get('optimizer') is not None:
            self.optimizer.load_state_dict(state['optimizer'])

    def save_ply(self, path):
        """Canonical field as a PLY in the 3DGS layout, for viewers."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        names, columns = [], []
        for name in self.PARAMS:
            tensor = getattr(self, '_' + name).detach()
            if name in ('f_dc', 'f_rest'):
                tensor = tensor.transpose(1, 2).flatten(start_dim=1)
            columns.append(tensor.cpu().numpy().reshape(self.num_points, -1))
            prefix = {'scaling': 'scale', 'rotation': 'rot'}.get(name, name)
            if name == 'xyz':
                names += ['x', 'y', 'z']
                columns.append(np.zeros_like(columns[-1]))
                names += ['nx', 'ny', 'nz']
            elif columns[-1].shape[1] == 1 and name not in ('scaling',):
                names.append(prefix)
            else:
                names += [f'{prefix}_{i}' for i in range(columns[-1].shape[1])]
        elements = np.empty(self.num_points, dtype=[(n, 'f4') for n in names])
        elements[:] = list(map(tuple, np.concatenate(columns, axis=1)))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    # ------------------------------------------------------------------ optimizer surgery

    def _set_params(self, tensors):
        for name, tensor in tensors.items():
            setattr(self, '_' + name, tensor)

    def _replace_in_optimizer(self, tensor, name):
        """Swap the parameter `name` for `tensor`, zeroing its Adam moments."""
        for group in self.optimizer.param_groups:
            if group['name'] != name:
                continue
            state = self.optimizer.state.pop(group['params'][0], None)
            group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
            if state is not None:
                state['exp_avg'] = torch.zeros_like(tensor)
                state['exp_avg_sq'] = torch.zeros_like(tensor)
                self.optimizer.state[group['params'][0]] = state
            self._set_params({name: group['params'][0]})

    def _prune_optimizer(self, keep):
        tensors = {}
        for group in self.optimizer.param_groups:
            state = self.optimizer.state.pop(group['params'][0], None)
            group['params'][0] = nn.Parameter(group['params'][0][keep].requires_grad_(True))
            if state is not None:
                state['exp_avg'] = state['exp_avg'][keep]
                state['exp_avg_sq'] = state['exp_avg_sq'][keep]
                self.optimizer.state[group['params'][0]] = state
            tensors[group['name']] = group['params'][0]
        return tensors

    def _extend_optimizer(self, new):
        """Append new Gaussians; surviving Gaussians keep their Adam moments, new ones start at zero."""
        tensors = {}
        for group in self.optimizer.param_groups:
            extension = new[group['name']]
            state = self.optimizer.state.pop(group['params'][0], None)
            group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension), dim=0).requires_grad_(True))
            if state is not None:
                state['exp_avg'] = torch.cat((state['exp_avg'], torch.zeros_like(extension)), dim=0)
                state['exp_avg_sq'] = torch.cat((state['exp_avg_sq'], torch.zeros_like(extension)), dim=0)
                self.optimizer.state[group['params'][0]] = state
            tensors[group['name']] = group['params'][0]
        return tensors

    def _reset_densification_stats(self):
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device='cuda')
        self.denom = torch.zeros((self.num_points, 1), device='cuda')

    def prune_points(self, remove):
        keep = ~remove
        self._set_params(self._prune_optimizer(keep))
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]

    def _add_points(self, new):
        self._set_params(self._extend_optimizer(new))
        self._reset_densification_stats()
        self.max_radii2D = torch.zeros(self.num_points, device='cuda')

    # ------------------------------------------------------------------ densification (3DGS schedule)

    def add_densification_stats(self, gradient_norm, visible):
        """Accumulate a per-Gaussian positional gradient norm (screen space for disks, world space for rays)."""
        self.xyz_gradient_accum[visible] += gradient_norm[visible]
        self.denom[visible] += 1

    def update_max_radii(self, radii, visible):
        self.max_radii2D[visible] = torch.max(self.max_radii2D[visible], radii[visible].float())

    def _split(self, selected, N=2):
        new = {}
        for name in self.PARAMS:
            param = getattr(self, '_' + name)[selected]
            new[name] = param.repeat(N, *([1] * (param.dim() - 1)))
        stds = self.get_scaling[selected].repeat(N, 1)
        if self.SCALE_DIM == 2:  # a disk has no extent along its normal
            stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rotations = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new['xyz'] = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(N, 1)
        new['scaling'] = torch.log(self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        self._add_points(new)
        self.prune_points(torch.cat((selected, torch.zeros(N * int(selected.sum()), device='cuda', dtype=torch.bool))))

    def densify(self, max_grad, extent):
        """Clone small and split large Gaussians with a large mean gradient, within the point budget."""
        grads = (self.xyz_gradient_accum / self.denom).nan_to_num(0.0).squeeze(-1)
        budget = self.max_points - self.num_points
        if budget <= 0:
            return
        candidates = (grads >= max_grad).nonzero().squeeze(-1)
        if candidates.numel() > budget:  # a clone and a split each add one Gaussian: keep the strongest
            candidates = candidates[torch.topk(grads[candidates], budget).indices]
        selected = torch.zeros_like(grads, dtype=torch.bool)
        selected[candidates] = True
        large = self.get_scaling.max(dim=1).values > self.percent_dense * extent
        clone = selected & ~large
        if clone.any():
            self._add_points({name: getattr(self, '_' + name)[clone] for name in self.PARAMS})
            selected = torch.cat([selected, torch.zeros(int(clone.sum()), device='cuda', dtype=torch.bool)])
            large = torch.cat([large, torch.zeros(int(clone.sum()), device='cuda', dtype=torch.bool)])
        split = selected & large
        if split.any():
            self._split(split)

    def densify_and_prune(self, max_grad, min_presence, extent, max_screen_size=None, prune_large=False):
        """
        3DGS densification step. After the first presence reset (`prune_large`), Gaussians larger than 10% of the
        scene extent are removed, and with `max_screen_size` also those whose screen radius exceeded it: without
        this, a few huge Gaussians in wide scenes cover every tile and make the rasterizer buffers explode.
        """
        self.densify(max_grad, extent)
        remove = (self.get_presence[:, 0] < min_presence) | ~torch.isfinite(self._xyz).all(-1)
        if prune_large:
            remove |= self.get_scaling.max(dim=1).values > 0.1 * extent
            if max_screen_size:
                remove |= self.max_radii2D > max_screen_size
        if remove.all():
            remove[self.get_presence[:, 0].argmax()] = False
        self.prune_points(remove)
        torch.cuda.empty_cache()

    def _presence_name(self):
        return 'opacity'

    def reset_presence(self, ceiling=0.01):
        """Clamp the presence to `ceiling` (3DGS opacity reset), so Gaussians that are not needed get pruned."""
        name = self._presence_name()
        value = torch.sigmoid(getattr(self, '_' + name))
        self._replace_in_optimizer(inverse_sigmoid(torch.clamp_max(value, ceiling)), name)


class InterfaceField(GaussianField):
    """
    G_I^0: 2D Gaussian disks with occupancy p (soft presence) and opacity o (given presence). The direct pass uses
    p * o, as in RT-Splatting's 2DGS renderer. `factorized=False` is the ablation without the factorization: p = 1
    for every disk and a single opacity plays both roles.
    """

    SCALE_DIM = 2
    INIT_OPACITY = 0.5
    INIT_OCCUPANCY = 0.1
    INIT_SCALE_FACTOR = 0.1

    def __init__(self, sh_degree, max_points=500_000, factorized=True):
        self.factorized = factorized
        self.PARAMS = GaussianField.PARAMS + (('occupancy',) if factorized else ())
        super().__init__(sh_degree, max_points)

    @property
    def get_occupancy(self):
        if not self.factorized:
            return torch.ones_like(self._opacity)
        return torch.sigmoid(self._occupancy)

    @property
    def get_effective_opacity(self):
        """p * o: the opacity of the direct image B_t (Eq. 2)."""
        return self.get_occupancy * self.get_opacity

    @property
    def get_presence(self):
        return self.get_occupancy if self.factorized else self.get_opacity

    def _presence_name(self):
        return 'occupancy' if self.factorized else 'opacity'

    def _initial_rotation(self, num_points):
        return nn.functional.normalize(torch.randn((num_points, 4), device='cuda'), dim=-1)

    def _initial_extra_params(self, num_points):
        if not self.factorized:
            return {}
        return {'occupancy': inverse_sigmoid(self.INIT_OCCUPANCY * torch.ones((num_points, 1), device='cuda'))}

    def _learning_rates(self, opt):
        rates = super()._learning_rates(opt)
        if self.factorized:
            rates['occupancy'] = opt.occupancy_lr
        return rates


class ReflectionField(GaussianField):
    """G_R^0: 3D Gaussian ellipsoids, independent of G_I; queried along the reflected rays (Sec. 3)."""

    SCALE_DIM = 3
    INIT_OPACITY = 0.1

    def __init__(self, sh_degree, max_points=250_000):
        super().__init__(sh_degree, max_points)
