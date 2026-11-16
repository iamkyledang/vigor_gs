"""
Deformation MLP shared by the interface field (D_phi_I) and the reflection field (D_phi_R).

Adapted from "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction"
(Yang et al., CVPR 2024, https://github.com/ingra14m/Deformable-3D-Gaussians).

The network maps a canonical Gaussian center and a timestamp to offsets of the geometry only:

    (d_xyz, d_rotation, d_scaling) = D_phi(gamma(mu), gamma(t))

Occupancy, opacity and appearance are time independent and are never touched here. How the offsets are
applied is up to the renderer (see `apply_deformation`).
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int, num_freqs: int):
        super().__init__()
        self.register_buffer('freqs', 2.0 ** torch.arange(num_freqs, dtype=torch.float32), persistent=False)
        self.out_dim = in_dim * (1 + 2 * num_freqs)

    def forward(self, x):
        scaled = x[..., None] * self.freqs
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(-2)
        return torch.cat([x, encoded], dim=-1)


class DeformNetwork(nn.Module):
    def __init__(self, scale_dim: int = 3, depth: int = 8, width: int = 256, xyz_multires: int = 10, time_multires: int = 6):
        """
        scale_dim: 2 for the interface disks (2DGS), 3 for the reflection ellipsoids (3DGS).
        """
        super().__init__()
        self.scale_dim = scale_dim
        self.skips = [depth // 2]

        self.embed_xyz = PositionalEncoding(3, xyz_multires)
        self.embed_time = PositionalEncoding(1, time_multires)
        self.input_ch = self.embed_xyz.out_dim + self.embed_time.out_dim

        layers = [nn.Linear(self.input_ch, width)]
        for i in range(depth - 1):
            layers.append(nn.Linear(width + self.input_ch if i in self.skips else width, width))
        self.linear = nn.ModuleList(layers)

        self.gaussian_warp = nn.Linear(width, 3)
        self.gaussian_rotation = nn.Linear(width, 4)
        self.gaussian_scaling = nn.Linear(width, scale_dim)

        # Start as the identity deformation: a fresh network leaves the canonical field untouched.
        for head in (self.gaussian_warp, self.gaussian_rotation, self.gaussian_scaling):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, xyz, time):
        """
        xyz: (N, 3) canonical centers, time: (N, 1) timestamps in [0, 1].
        Returns d_xyz (N, 3), d_rotation (N, 4), d_scaling (N, scale_dim).
        """
        inputs = torch.cat([self.embed_xyz(xyz), self.embed_time(time)], dim=-1)
        h = inputs
        for i, layer in enumerate(self.linear):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([inputs, h], dim=-1)
        return self.gaussian_warp(h), self.gaussian_rotation(h), self.gaussian_scaling(h)


def apply_deformation(xyz, rotation, log_scaling, d_xyz, d_rotation, d_scaling):
    """
    Deform canonical geometry.

    xyz: (N, 3) centers; rotation: (N, 4) quaternions; log_scaling: (N, S) scales before the exp activation.
    Centers and quaternions are offset additively (the quaternion is re-normalized). Scales are offset in log
    space, so a deformed scale is always positive.
    Returns (xyz, unit quaternion, positive scales).
    """
    return xyz + d_xyz, F.normalize(rotation + d_rotation, dim=-1), torch.exp(log_scaling + d_scaling)


class DeformModel:
    """A DeformNetwork with its optimizer, learning-rate schedule and checkpoints."""

    def __init__(self, scale_dim: int = 3, depth: int = 8, width: int = 256, xyz_multires: int = 10, time_multires: int = 6, device='cuda'):
        self.network = DeformNetwork(scale_dim, depth, width, xyz_multires, time_multires).to(device)
        self.optimizer = None
        self.lr_schedule = None

    @classmethod
    def from_args(cls, scale_dim, args, device='cuda'):
        """Build from the `deform_*` model parameters, falling back to the defaults for older configs."""
        return cls(
            scale_dim,
            depth=getattr(args, 'deform_depth', 8),
            width=getattr(args, 'deform_width', 256),
            xyz_multires=getattr(args, 'deform_xyz_multires', 10),
            time_multires=getattr(args, 'deform_time_multires', 6),
            device=device,
        )

    def step(self, xyz, time, time_noise: float = 0.0):
        """
        Offsets for all Gaussians at one timestamp.

        xyz: (N, 3) canonical centers. They are detached: the deformation loss does not move the canonical field
        through the network input. time: float or 0-d tensor in [0, 1]. time_noise: std of a per-call jitter on
        the timestamp (annealed smoothing for real captures).
        """
        t = torch.as_tensor(time, dtype=xyz.dtype, device=xyz.device).reshape(1, 1)
        if time_noise > 0:
            t = t + torch.randn_like(t) * time_noise
        return self.network(xyz.detach(), t.expand(xyz.shape[0], 1))

    def train_setting(self, training_args, spatial_lr_scale: float = 1.0):
        lr_init = training_args.position_lr_init * spatial_lr_scale
        self.optimizer = torch.optim.Adam([{'params': list(self.network.parameters()), 'lr': lr_init, 'name': 'deform'}], lr=0.0, eps=1e-15)
        self.lr_schedule = get_expon_lr_func(lr_init=lr_init, lr_final=training_args.position_lr_final, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.deform_lr_max_steps)

    def update_learning_rate(self, iteration):
        lr = self.lr_schedule(iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def freeze(self):
        self.network.requires_grad_(False)
        self.network.eval()
        self.optimizer = None

    def save_weights(self, model_path, iteration):
        out_dir = os.path.join(model_path, 'deform', 'iteration_{}'.format(iteration))
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(out_dir, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(model_path, 'deform'))
        weights_path = os.path.join(model_path, 'deform', 'iteration_{}'.format(iteration), 'deform.pth')
        self.network.load_state_dict(torch.load(weights_path, map_location=next(self.network.parameters()).device))
        return iteration
