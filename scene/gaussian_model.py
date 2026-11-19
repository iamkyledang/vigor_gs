"""
Shared machinery for the two VIGOR-GS Gaussian fields.

`GaussianField` owns everything that does not depend on whether the primitives are 2D disks or 3D ellipsoids:
parameter storage, the optimizer and the surgery it needs when Gaussians are cloned, split or pruned, PLY
persistence and checkpoints. Per-Gaussian parameters are listed once in `PARAMS`; a subclass that adds one only
extends that tuple (plus its initial value and learning rate) and everything else follows.

Derived from the 3DGS / 2DGS `GaussianModel` (Inria GRAPHDECO, see LICENSE.md).
"""

import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from torch import nn

from utils.general_utils import build_rotation, get_expon_lr_func, inverse_sigmoid
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class GaussianField:
    SCALE_DIM = 3
    # Optimizer group names. The parameter for group `name` is stored as `self._<name>`.
    PARAMS = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')
    INIT_OPACITY = 0.1
    INIT_SCALE_FACTOR = 1.0  # multiplies the squared nearest-neighbor distance used for the initial scale

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        for name in self.PARAMS:
            setattr(self, '_' + name, torch.empty(0))
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.last_update = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

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
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ------------------------------------------------------------------ initialization

    def _initial_rotation(self, num_points):
        rots = torch.zeros((num_points, 4), device='cuda')
        rots[:, 0] = 1
        return rots

    def _initial_extra_params(self, num_points):
        """Initial raw values for the parameters a subclass adds to PARAMS."""
        return {}

    def create_from_points(self, points, colors, spatial_lr_scale: float):
        """points: (N, 3) world positions, colors: (N, 3) RGB in [0, 1]."""
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.as_tensor(np.asarray(points) if not torch.is_tensor(points) else points).float().cuda()
        colors = torch.as_tensor(np.asarray(colors) if not torch.is_tensor(colors) else colors).float().cuda()
        num_points = points.shape[0]

        features = torch.zeros((num_points, 3, (self.max_sh_degree + 1) ** 2), device='cuda')
        features[:, :, 0] = RGB2SH(colors)

        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001) * self.INIT_SCALE_FACTOR
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, self.SCALE_DIM)
        opacities = self.inverse_opacity_activation(self.INIT_OPACITY * torch.ones((num_points, 1), device='cuda'))

        initial = {
            'xyz': points,
            'f_dc': features[:, :, 0:1].transpose(1, 2).contiguous(),
            'f_rest': features[:, :, 1:].transpose(1, 2).contiguous(),
            'opacity': opacities,
            'scaling': scales,
            'rotation': self._initial_rotation(num_points),
        }
        initial.update(self._initial_extra_params(num_points))
        for name in self.PARAMS:
            setattr(self, '_' + name, nn.Parameter(initial[name].requires_grad_(True)))
        self.max_radii2D = torch.zeros((num_points), device='cuda')
        print('Number of points at initialisation : ', num_points)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.create_from_points(pcd.points, pcd.colors, spatial_lr_scale)

    # ------------------------------------------------------------------ optimization

    def _learning_rates(self, training_args):
        return {
            'xyz': training_args.position_lr_init * self.spatial_lr_scale,
            'f_dc': training_args.feature_lr,
            'f_rest': training_args.feature_lr / 20.0,
            'opacity': training_args.opacity_lr,
            'scaling': training_args.scaling_lr,
            'rotation': training_args.rotation_lr,
        }

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self._reset_densification_stats()
        self.last_update = torch.zeros((self.num_points, 1), device='cuda', dtype=torch.int)

        lrs = self._learning_rates(training_args)
        groups = [{'params': [getattr(self, '_' + name)], 'lr': lrs[name], 'name': name} for name in self.PARAMS]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def freeze(self):
        """Turn the field into a constant: no gradients, no optimizer."""
        for name in self.PARAMS:
            getattr(self, '_' + name).requires_grad_(False)
        self.optimizer = None

    # ------------------------------------------------------------------ persistence

    def capture(self):
        return {
            'active_sh_degree': self.active_sh_degree,
            'params': {name: getattr(self, '_' + name) for name in self.PARAMS},
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'last_update': self.last_update,
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'spatial_lr_scale': self.spatial_lr_scale,
        }

    def restore(self, state, training_args):
        self.active_sh_degree = state['active_sh_degree']
        self.spatial_lr_scale = state['spatial_lr_scale']
        for name in self.PARAMS:
            setattr(self, '_' + name, state['params'][name])
        self.training_setup(training_args)
        self.max_radii2D = state['max_radii2D']
        self.xyz_gradient_accum = state['xyz_gradient_accum']
        self.denom = state['denom']
        self.last_update = state['last_update']
        if state.get('optimizer') is not None:
            self.optimizer.load_state_dict(state['optimizer'])

    def _param_dim(self, name):
        if name == 'xyz':
            return 3
        if name == 'f_dc':
            return 3
        if name == 'f_rest':
            return 3 * ((self.max_sh_degree + 1) ** 2 - 1)
        if name == 'scaling':
            return self.SCALE_DIM
        if name == 'rotation':
            return 4
        return 1

    @staticmethod
    def _ply_names(name, dim):
        if name == 'xyz':
            return ['x', 'y', 'z']
        prefix = {'scaling': 'scale', 'rotation': 'rot'}.get(name, name)
        if dim == 1 and name not in ('f_dc', 'f_rest', 'scaling', 'rotation'):
            return [prefix]
        return ['{}_{}'.format(prefix, i) for i in range(dim)]

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        names, columns = [], []
        for name in self.PARAMS:
            tensor = getattr(self, '_' + name).detach()
            if name in ('f_dc', 'f_rest'):
                tensor = tensor.transpose(1, 2).flatten(start_dim=1).contiguous()
            columns.append(tensor.cpu().numpy())
            names.extend(self._ply_names(name, columns[-1].shape[1]))
            if name == 'xyz':  # viewers expect normals right after the position
                columns.append(np.zeros_like(columns[-1]))
                names.extend(['nx', 'ny', 'nz'])

        elements = np.empty(self.num_points, dtype=[(attribute, 'f4') for attribute in names])
        elements[:] = list(map(tuple, np.concatenate(columns, axis=1)))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    def load_ply(self, path):
        vertex = PlyData.read(path).elements[0]
        for name in self.PARAMS:
            dim = self._param_dim(name)
            values = np.stack([np.asarray(vertex[column]) for column in self._ply_names(name, dim)], axis=1) if dim > 0 else np.zeros((vertex.count, 0))
            tensor = torch.tensor(values, dtype=torch.float, device='cuda')
            if name in ('f_dc', 'f_rest'):
                tensor = tensor.reshape(tensor.shape[0], 3, -1).transpose(1, 2).contiguous()
            setattr(self, '_' + name, nn.Parameter(tensor.requires_grad_(True)))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.num_points), device='cuda')
        print('Load Path', path)

    # ------------------------------------------------------------------ optimizer surgery

    def _sync_params(self, optimizable_tensors):
        for name, tensor in optimizable_tensors.items():
            setattr(self, '_' + name, tensor)

    def replace_tensor_to_optimizer(self, tensor, name):
        for group in self.optimizer.param_groups:
            if group['name'] != name:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
            group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group['params'][0]] = stored_state
            self._sync_params({name: group['params'][0]})

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                del self.optimizer.state[group['params'][0]]
            group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
            group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _reset_densification_stats(self):
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device='cuda')
        self.denom = torch.zeros((self.num_points, 1), device='cuda')

    def prune_points(self, mask):
        """Remove the Gaussians where `mask` is True."""
        valid_points_mask = ~mask
        self._sync_params(self._prune_optimizer(valid_points_mask))
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.last_update = self.last_update[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, new_params, new_last_update):
        self._sync_params(self._cat_tensors_to_optimizer(new_params))
        self.last_update = torch.cat((self.last_update, new_last_update), dim=0)
        self._reset_densification_stats()
        self.max_radii2D = torch.zeros((self.num_points), device='cuda')

    # ------------------------------------------------------------------ densification

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.num_points
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        new_params = {}
        for name in self.PARAMS:
            param = getattr(self, '_' + name)[selected_pts_mask]
            new_params[name] = param.repeat(N, *([1] * (param.dim() - 1)))

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        if self.SCALE_DIM == 2:  # a disk has no extent along its normal
            stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_params['xyz'] = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_params['scaling'] = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))

        if not selected_pts_mask.any():
            return
        self.densification_postfix(new_params, self.last_update[selected_pts_mask].repeat(N, 1))

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        if not selected_pts_mask.any():
            return
        new_params = {name: getattr(self, '_' + name)[selected_pts_mask] for name in self.PARAMS}
        self.densification_postfix(new_params, self.last_update[selected_pts_mask])

    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        budget = max(0, getattr(self, 'max_points', 1000000) - self.num_points)
        if budget == 0:
            return
        scores = grads.flatten()
        eligible = (scores >= max_grad).nonzero().flatten()
        if eligible.numel() > budget:
            keep = eligible[torch.topk(scores[eligible], budget).indices]
            grads = torch.zeros_like(grads)
            grads[keep] = scores[keep, None]
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter, iteration):
        if viewspace_point_tensor.grad is None:
            return
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.last_update[update_filter] = iteration

    def _reset_activated(self, name, activation, inverse_activation, ceiling=0.01):
        """Clamp an activated per-Gaussian value to at most `ceiling` and reset its optimizer state."""
        raw = getattr(self, '_' + name)
        for other in ('xyz', 'scaling', 'rotation', name):
            raw.data[torch.isnan(getattr(self, '_' + other).data.mean(dim=-1))] = 0.0
        activated = activation(raw)
        self.replace_tensor_to_optimizer(inverse_activation(torch.min(activated, torch.ones_like(activated) * ceiling)), name)

    def reset_opacity(self):
        self._reset_activated('opacity', self.opacity_activation, self.inverse_opacity_activation)


class GaussianModel(GaussianField):
    """RT-Splatting occupancy/opacity disks, or an independent 3D reflection field."""
    def __init__(self, sh_degree, args=None, field='interface'):
        self.field = field
        self.SCALE_DIM = 2 if field == 'interface' else 3
        self.PARAMS = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')
        self.INIT_OPACITY = 0.5 if field == 'interface' else 0.1
        self.INIT_SCALE_FACTOR = 0.1 if field == 'interface' else 1.0
        if field == 'interface':
            self.PARAMS += ('occupancy',)
        super().__init__(sh_degree)
        self.max_points = getattr(args, 'max_interface_points' if field == 'interface' else 'max_reflection_points', 500000)

    @property
    def get_occupancy(self):
        return torch.sigmoid(self._occupancy) if self.field == 'interface' else torch.ones_like(self._opacity)

    def _initial_extra_params(self, num_points):
        if self.field == 'interface':
            return {'occupancy': inverse_sigmoid(torch.full((num_points, 1), 0.1, device='cuda'))}
        return {}

    def _initial_rotation(self, num_points):
        if self.field == 'interface':
            return torch.nn.functional.normalize(torch.randn(num_points, 4, device='cuda'), dim=-1)
        return super()._initial_rotation(num_points)

    def _learning_rates(self, args):
        rates = super()._learning_rates(args)
        rates['occupancy'] = args.occupancy_lr
        return rates

    def densify_and_prune(self, threshold, extent, min_presence=0.005):
        # Preserve screen sizes: optimizer surgery resets those buffers during growth.
        self.densify(threshold, extent)
        presence = self.get_occupancy if self.field == 'interface' else self.get_opacity
        remove = (presence[:, 0] < min_presence) | ~torch.isfinite(self._xyz).all(-1)
        remove |= self.get_scaling.max(-1).values > extent
        if remove.all():
            remove[presence[:, 0].argmax()] = False
        self.prune_points(remove)


class Deformation(torch.nn.Module):
    """Independent time-conditioned geometry MLP; occupancy and opacity stay canonical."""
    def __init__(self, scale_dim, args, center=None, extent=1.0):
        super().__init__()
        self.scale_dim = scale_dim
        self.register_buffer('center', torch.zeros(3) if center is None else torch.as_tensor(center).float())
        self.register_buffer('extent', torch.tensor(max(float(extent), 1e-6)))
        self.register_buffer('xyz_freq', 2.0 ** torch.arange(args.deform_xyz_freqs))
        self.register_buffer('time_freq', 2.0 ** torch.arange(args.deform_time_freqs))
        in_dim = 3 * (1 + 2 * args.deform_xyz_freqs) + 1 + 2 * args.deform_time_freqs
        layers = []
        for i in range(args.deform_depth):
            layers.extend([nn.Linear(in_dim if i == 0 else args.deform_width, args.deform_width), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(args.deform_width, 7 + scale_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def encode(x, freqs):
        phase = x[..., None] * freqs * torch.pi
        return torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)

    def forward(self, pc, time, enabled=True):
        if not enabled:
            return pc.get_xyz, pc.get_rotation, pc.get_scaling
        xyz = (pc.get_xyz.detach() - self.center) / self.extent
        t = xyz.new_full((xyz.shape[0], 1), float(time))
        features = torch.cat([self.encode(xyz, self.xyz_freq), self.encode(t, self.time_freq)], -1)
        delta = self.head(self.trunk(features))
        position = pc.get_xyz + self.extent * delta[:, :3]
        rotation = torch.nn.functional.normalize(pc.get_rotation + delta[:, 3:7], dim=-1)
        scales = torch.exp((pc._scaling + delta[:, 7:]).clamp(-14, 8))
        return position, rotation, scales


class OpticalCorrection(nn.Module):
    """Bounded, zero-initialized h(u,t), without a learned glass mask."""
    def __init__(self, bound=0.1):
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, uv, time, cosine):
        t = uv.new_full((*uv.shape[:-1], 1), float(time))
        features = torch.cat([uv, t, cosine[..., None]], -1)
        return self.bound * torch.tanh(self.net(features)[..., 0])
