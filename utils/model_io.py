"""
Loading the trained steps of a model folder, shared by extract_glass.py, train_reflection.py and render.py:

    <model>/config.json                ModelParams (written by Phase I)
    <model>/interface/latest.pth       frozen G_I and D_I
    <model>/glass_geometry.json        patches P and the first-hit settings (extract_glass.py)
    <model>/reflection/latest.pth      G_R, D_R and the optical correction (train_reflection.py)
"""

import os
from collections import OrderedDict

import torch

from gaussian_renderer.interface import render_interface
from scene.deformation import Deformation
from scene.gaussian_model import InterfaceField
from utils.checkpoint_utils import load_checkpoint, restore_phase
from utils.glass_geometry import load_glass_geometry

GLASS_GEOMETRY = 'glass_geometry.json'


def load_interface(model_args, scene):
    """The frozen Phase I field and deformation network."""
    path = os.path.join(model_args.model_path, 'interface', 'latest.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: run train_interface.py first')
    state = load_checkpoint(path)
    field = InterfaceField(model_args.sh_degree, model_args.max_interface_points, factorized=not model_args.ablate_factorization)
    deformation = Deformation(2, model_args, scene.center, scene.extent, static=model_args.static_interface)
    restore_phase(state, field, deformation)
    return field, deformation, state['iteration']


def load_glass(model_args):
    """(patches, record): the record holds 'settings' (first-hit rule) and 'quality'."""
    path = os.path.join(model_args.model_path, GLASS_GEOMETRY)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: run extract_glass.py first')
    return load_glass_geometry(path)


class DirectPass:
    """
    The frozen direct pass of G_I per view: B_t, its depth and alpha. Phase II needs B_t for Eq. (6) and the depth
    for the first visible hit of Eq. (9). Renders are cached in host RAM (least recently used first out), so a long
    sequence does not hold every frame on the GPU and an evicted frame is simply rendered again.
    """

    def __init__(self, field, deformation, pipe, capacity=32):
        self.field, self.deformation, self.pipe = field, deformation, pipe
        self.capacity = capacity
        self.entries = OrderedDict()

    @torch.no_grad()
    def __call__(self, cam):
        key = (cam.uid, float(cam.time))
        if key not in self.entries:
            out = render_interface(cam, self.field, self.deformation, self.pipe)
            self.entries[key] = {name: out[name].detach().to('cpu', non_blocking=True) for name in ('direct', 'depth', 'alpha')}
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        return {name: value.cuda(non_blocking=True) for name, value in self.entries[key].items()}


def occlusion_settings(settings, direct):
    """The `occlusion` argument of first_hit for one direct pass, or None for the literal Eq. (9)."""
    if settings.get('ablate_occlusion'):
        return None
    return {'depth': direct['depth'], 'alpha': direct['alpha'], 'tolerance': settings['occlusion_tolerance'], 'alpha_min': settings['occlusion_alpha']}
