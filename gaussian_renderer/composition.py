"""
Fresnel weight and the two-term blend of Phase II and inference (Eq. 6, 10, 11, 13):

    F0 = ((eta_a - eta_g) / (eta_a + eta_g))^2,   F(theta) = F0 + (1 - F0)(1 - cos theta)^5                Eq. (10)
    alpha_t(u) = clip_[0,1](F(theta_t(u)) + h_psi(u, t)) on the selected glass, 0 elsewhere                   Eq. (11)
    I_hat_t(u) = (1 - alpha_t(u)) B_t(u) + alpha_t(u) R_t(u)                                                  Eq. (6)

h_psi is a small MLP, zero at initialization (pure Fresnel) and penalized by L_opt (Eq. 7c). The framework writes
it as a function of the pixel u and the time t; its input here is the glass point x_t(u) hit by the pixel's ray
(normalized by the scene), the time and cos theta_t(u). Given the camera, x_t(u) is a function of u, and unlike raw
pixel coordinates it names the same spot of the glass in every view, so a correction learned in the training
views applies to the same glass in a test view. It is evaluated on the glass pixels only.

Ablations: --pure_fresnel (h_psi = 0), --ablate_fresnel (F(theta) replaced by one learned constant sigmoid(c) on
the glass; h_psi, if enabled, adds on top of it).
"""

import torch
from torch import nn

from gaussian_renderer.reflection import ray_bundle, render_reflection
from utils.glass_geometry import first_hit


def schlick(cosine, eta_air, eta_glass):
    f0 = ((eta_air - eta_glass) / (eta_air + eta_glass)) ** 2
    return f0 + (1 - f0) * (1 - cosine).clamp(0, 1) ** 5


class OpticalCorrection(nn.Module):
    """h_psi(u, t) = MLP(gamma(x_hat_t(u)), gamma(t), cos theta_t(u)); zero-initialized output layer."""

    def __init__(self, center, extent, bound=0.0, xyz_freqs=4, time_freqs=2, width=64):
        super().__init__()
        self.bound = float(bound)
        self.register_buffer('center', torch.as_tensor(center, dtype=torch.float32).reshape(3))
        self.register_buffer('extent', torch.tensor(max(float(extent), 1e-6)))
        self.register_buffer('xyz_freqs', 2.0 ** torch.arange(xyz_freqs))
        self.register_buffer('time_freqs', 2.0 ** torch.arange(time_freqs))
        in_dim = 3 * (1 + 2 * xyz_freqs) + (1 + 2 * time_freqs) + 1
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def encode(x, freqs):
        phase = x[..., None] * freqs * torch.pi
        return torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)

    def forward(self, points, time, cosine):
        """points (P, 3) glass hits, cosine (P,) -> h (P,)."""
        x = (points - self.center) / self.extent
        t = points.new_full((points.shape[0], 1), float(time))
        h = self.net(torch.cat([self.encode(x, self.xyz_freqs), self.encode(t, self.time_freqs), cosine[:, None]], -1))[:, 0]
        return self.bound * torch.tanh(h / self.bound) if self.bound > 0 else h


class ConstantReflectance(nn.Module):
    """Ablation --ablate_fresnel: one learned reflection weight sigmoid(c) for every glass pixel."""

    def __init__(self, initial):
        super().__init__()
        self.logit = nn.Parameter(torch.logit(torch.tensor(float(initial))))

    def forward(self):
        return torch.sigmoid(self.logit)


def build_blend(params, center, extent):
    """(optical, constant) modules of a Phase II model; either may be None."""
    optical = None if params.pure_fresnel else OpticalCorrection(center, extent, params.optical_bound)
    constant = ConstantReflectance(schlick(torch.tensor(1.0), params.eta_air, params.eta_glass)) if params.ablate_fresnel else None
    return optical, constant


def ray_mode(params):
    if params.ablate_virtual_camera:
        return 'unreflected'
    return 'pinhole' if params.pinhole_virtual_camera else 'exact'


def reflection_weight(geometry, view, params, optical=None, constant=None):
    """alpha (1, H, W) of Eq. (11) and the correction h (1, H, W), zero off the glass."""
    visible = geometry['visible']
    base = constant().expand_as(geometry['cosine']) if constant is not None else schlick(geometry['cosine'], params.eta_air, params.eta_glass)
    h = torch.zeros_like(geometry['cosine'])
    if optical is not None and visible.any():
        h = h.masked_scatter(visible, optical(geometry['hits'][visible], view.time, geometry['cosine'][visible]))
    alpha = torch.where(visible, (base + h).clamp(0, 1), torch.zeros_like(base))
    return alpha[None], h[None]


def render_composite(view, direct, reflection_field, reflection_deformation, patches, settings, params, pipe,
                     optical=None, constant=None, deforming=True):
    """
    I_hat_t of Eq. (6) for one view.

    direct: {'direct', 'depth', 'alpha'} of the frozen interface (utils/model_io.DirectPass); settings: the first-hit
    rule of glass_geometry.json. Returns composite, direct, reflection, alpha, h (zero off the glass), geometry, the
    Eq. (9b) virtual-camera poses and what densification of G_R needs.
    """
    from utils.model_io import occlusion_settings

    geometry = first_hit(view, patches, pipe.ray_epsilon, occlusion_settings(settings, direct))
    bundle = ray_bundle(view, geometry, patches, ray_mode(params), pipe.ray_epsilon)
    out = render_reflection(view, reflection_field, reflection_deformation, bundle, patches, pipe, deforming)
    alpha, h = reflection_weight(geometry, view, params, optical, constant)
    composite = (1 - alpha) * direct['direct'] + alpha * out['reflection']
    out.update(composite=composite, direct=direct['direct'], alpha=alpha, h=h, geometry=geometry, poses=bundle['poses'])
    return out
