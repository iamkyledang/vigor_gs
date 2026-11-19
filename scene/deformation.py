"""Independent time-conditioned geometry networks for both Gaussian fields.

Canonical field storage and optimizer surgery live in gaussian_model.py.
"""
import torch
from torch import nn


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
