"""
Fresnel-guided compositing of the direct image B_t and the reflection image R_t:

    cos(theta_t(u)) = |n^T (c_t - x_t(u))| / |c_t - x_t(u)|          x_t(u): where the pixel's ray meets the glass
    F(theta)        = F0 + (1 - F0) (1 - cos(theta))^5,  F0 = ((eta_g - eta_a) / (eta_g + eta_a))^2     (Schlick)
    alpha_t(u)      = clip_[0,1](F(theta_t(u)) + h_psi(u, t)) * glass_map_t(u)
    I_hat_t(u)      = (1 - alpha_t(u)) B_t(u) + alpha_t(u) R_t(u)                                  (linear RGB)

Since x_t(u) lies on the pixel's own ray, c_t - x_t(u) is parallel to the ray direction, and the cosine reduces to
|n^T r(u)| for the unit ray direction r(u). The glass map restricts the reflection to the glass; elsewhere
alpha = 0 and the composite equals the direct image.
"""

import torch
import torch.nn.functional as F
from torch import nn

from scene.deform_model import PositionalEncoding
from utils.point_utils import camera_rays


def schlick_f0(eta_air: float = 1.0, eta_glass: float = 1.5):
    """Reflectance at normal incidence."""
    return ((eta_glass - eta_air) / (eta_glass + eta_air)) ** 2


def schlick_reflectance(cos_theta, f0):
    return f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(0.0, 1.0) ** 5


def glass_geometry(camera, normal, offset):
    """
    Per-pixel geometry of the glass as seen from a camera.

    Returns cos_theta (1, H, W), the absolute cosine between each pixel's ray and the glass normal, and
    glass_points (H, W, 3), where each ray meets the plane (rays that never reach it get the camera center).
    """
    rays_o, rays_d = camera_rays(camera)
    rays_d = F.normalize(rays_d, dim=-1)
    normal = normal.to(rays_d)
    cos_signed = rays_d @ normal
    distance = -(rays_o @ normal + offset) / torch.where(cos_signed.abs() < 1e-8, torch.full_like(cos_signed, 1e-8), cos_signed)
    distance = distance.clamp_min(0.0)
    glass_points = rays_o + distance.unsqueeze(-1) * rays_d
    return cos_signed.abs().unsqueeze(0), glass_points


class OpticalCorrection(nn.Module):
    """
    h_psi: a small, optional correction added to the Schlick reflectance.

    It sees the glass point x_t(u), the viewing cosine and the timestamp. The output layer starts at zero, so a
    fresh correction leaves pure Fresnel mixing untouched, and L_opt = mean(h^2) keeps it small afterwards.
    """

    def __init__(self, width: int = 64, depth: int = 3, xyz_multires: int = 6, time_multires: int = 4):
        super().__init__()
        self.embed_xyz = PositionalEncoding(3, xyz_multires)
        self.embed_time = PositionalEncoding(1, time_multires)
        in_dim = self.embed_xyz.out_dim + self.embed_time.out_dim + 1
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else width, width), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, glass_points, cos_theta, time):
        """glass_points: (H, W, 3), cos_theta: (1, H, W), time: float. Returns h (1, H, W)."""
        height, width = cos_theta.shape[-2:]
        xyz = glass_points.reshape(-1, 3)
        t = torch.full((xyz.shape[0], 1), float(time), dtype=xyz.dtype, device=xyz.device)
        features = torch.cat([self.embed_xyz(xyz), self.embed_time(t), cos_theta.reshape(-1, 1)], dim=-1)
        return self.head(self.trunk(features)).reshape(1, height, width)


def fresnel_weight(camera, normal, offset, glass_map=None, f0=None, correction: OpticalCorrection = None):
    """
    The reflection weight alpha_t(u) for a camera.

    glass_map: (1, H, W) in [0, 1], where the interface is glass (None: glass everywhere).
    Returns a dict with alpha, fresnel (Schlick term alone), cos_theta and h (None without a correction), all
    (1, H, W).
    """
    f0 = schlick_f0() if f0 is None else f0
    cos_theta, glass_points = glass_geometry(camera, normal, offset)
    fresnel = schlick_reflectance(cos_theta, f0)
    h = correction(glass_points, cos_theta, getattr(camera, 'time', 0.0)) if correction is not None else None
    alpha = (fresnel if h is None else fresnel + h).clamp(0.0, 1.0)
    if glass_map is not None:
        alpha = alpha * glass_map
    return {'alpha': alpha, 'fresnel': fresnel, 'cos_theta': cos_theta, 'h': h}


def composite(direct, reflection, alpha):
    """I_hat = (1 - alpha) B + alpha R, in linear RGB."""
    return (1.0 - alpha) * direct + alpha * reflection


def optical_loss(h):
    """L_opt = mean over pixels of h^2."""
    return (h**2).mean()
