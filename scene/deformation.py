"""
Time-conditioned geometry MLPs D_I and D_R of Eq. (1), after Yang et al. (Deformable 3D Gaussians, CVPR 2024):

    (dmu, dq, ds) = D_phi(gamma(mu_hat), gamma(t)),   mu' = mu + extent * dmu,   q' = normalize(q + dq),   s' = exp(log s + ds)

`mu_hat` is the canonical center normalized by the scene center and extent, gamma the sin/cos positional encoding.
Only geometry changes over time; occupancy, opacity and appearance stay canonical (Sec. 1). The output head starts
at zero, so a new network is the identity. Each field gets its own instance: the two networks share the class, not
the weights.

The canonical center is detached at the network input, as in Yang et al.: the gradient still reaches mu through
mu + dmu, but cannot move the canonical field through the encoding, which keeps canonical field and deformation
separable. `static=True` (ablations static_interface / static_reflection) removes the network: the field does not
move over time.
"""

import torch
from torch import nn


class Deformation(nn.Module):
    def __init__(self, scale_dim, args, center=None, extent=1.0, static=False):
        super().__init__()
        self.scale_dim = scale_dim
        self.static = static
        self.register_buffer('center', torch.zeros(3) if center is None else torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer('extent', torch.tensor(max(float(extent), 1e-6)))
        self.register_buffer('xyz_freqs', 2.0 ** torch.arange(args.deform_xyz_freqs))
        self.register_buffer('time_freqs', 2.0 ** torch.arange(args.deform_time_freqs))
        in_dim = 3 * (1 + 2 * args.deform_xyz_freqs) + 1 + 2 * args.deform_time_freqs
        layers = []
        for i in range(args.deform_depth):
            layers += [nn.Linear(in_dim if i == 0 else args.deform_width, args.deform_width), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(args.deform_width, 3 + 4 + scale_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.optimizer = None

    @staticmethod
    def encode(x, freqs):
        phase = x[..., None] * freqs * torch.pi
        return torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)

    def forward(self, field, time, enabled=True):
        """Deformed (centers, rotations, scales) of `field` at `time`; canonical ones when disabled or static."""
        if self.static or not enabled:
            return field.get_xyz, field.get_rotation, field.get_scaling
        xyz = (field.get_xyz.detach() - self.center) / self.extent
        t = xyz.new_full((xyz.shape[0], 1), float(time))
        delta = self.head(self.trunk(torch.cat([self.encode(xyz, self.xyz_freqs), self.encode(t, self.time_freqs)], -1)))
        centers = field.get_xyz + self.extent * delta[:, :3]
        rotations = nn.functional.normalize(field.get_rotation + delta[:, 3:7], dim=-1)
        scales = torch.exp((field._scaling + delta[:, 7:]).clamp(-14, 8))
        return centers, rotations, scales

    def training_setup(self, lr):
        self.optimizer = None if self.static else torch.optim.Adam(self.parameters(), lr=lr)

    def update_learning_rate(self, lr_init, iteration, max_steps):
        """Exponential decay to 1% of the initial rate over the run."""
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            group['lr'] = lr_init * (0.01 ** min(iteration / max(max_steps, 1), 1.0))

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        self.optimizer = None
