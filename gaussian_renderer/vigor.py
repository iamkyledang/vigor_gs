"""
The full VIGOR-GS forward pass for one camera:

    B_t   = R_2D(D_phi_I(G_I, t), C_t)                   direct image, from the frozen interface   (render_direct)
    R_t   = R_3D(D_phi_R(G_R, t), V(C_t, Pi))            reflection image through the virtual camera
    I_hat = (1 - alpha_t) B_t + alpha_t R_t              Fresnel-guided blend                      (render_composite)
"""

import torch

from gaussian_renderer.interface import render_interface
from gaussian_renderer.reflection import render_reflection
from utils.fresnel import composite, fresnel_weight
from utils.virtual_camera import reflect_camera


@torch.no_grad()
def render_direct(camera, interface, interface_deform, pipe, background):
    """Render the interface field at the camera's timestamp. Nothing here carries gradients: Phase I is frozen."""
    return render_interface(camera, interface, pipe, background, offsets=interface_deform.step(interface.get_xyz, camera.time))


def render_composite(camera, direct, glass_map, reflection_field, pipe, normal, offset, reflection_offsets=None, f0=None, correction=None, use_virtual_camera=True):
    """
    Render the reflection field and blend it with a direct image.

    direct: (3, H, W) B_t and glass_map: (1, H, W), both from `render_direct` (or its cache).
    reflection_offsets: output of the reflection deformation MLP, or None for the canonical field.
    use_virtual_camera=False renders G_R from the real camera instead (ablation).
    Returns the reflection render's entries plus alpha, fresnel, cos_theta, h, direct, glass_map and composite.
    """
    view = reflect_camera(camera, normal, offset) if use_virtual_camera else camera
    black = torch.zeros(3, dtype=torch.float32, device='cuda')  # no Gaussian, no reflection
    out = render_reflection(view, reflection_field, pipe, black, offsets=reflection_offsets)
    out.update(fresnel_weight(camera, normal, offset, glass_map=glass_map, f0=f0, correction=correction))
    out['direct'] = direct
    out['glass_map'] = glass_map
    out['composite'] = composite(direct, out['reflection'], out['alpha'])
    return out
