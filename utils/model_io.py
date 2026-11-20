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
from utils.glass_geometry import load_glass_geometry, on_patches

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
    The frozen direct pass of G_I per view: B_t (Eq. 6) and, for the first visible hit of Eq. (9), the depth and alpha
    of an occluder pass. With `patches`, the occluder pass leaves out the disks within `margin` of a glass patch (the
    glass layer itself); without them it is the direct pass. Renders are cached in host RAM (least recently used
    first out): Phase II never changes them, and an evicted frame is simply rendered again.
    """

    def __init__(self, field, deformation, pipe, capacity=32, patches=None, margin=0.0):
        self.field, self.deformation, self.pipe = field, deformation, pipe
        self.capacity = capacity
        self.patches, self.margin = patches, margin
        self.entries = OrderedDict()

    @torch.no_grad()
    def render(self, cam):
        out = render_interface(cam, self.field, self.deformation, self.pipe)
        entry = {'direct': out['direct'], 'depth': out['depth'], 'alpha': out['alpha']}
        entry['occluder_depth'], entry['occluder_alpha'] = out['depth'], out['alpha']
        if self.patches and self.margin > 0:
            centers = self.deformation(self.field, cam.time)[0]
            glass = on_patches(centers, self.patches, self.margin)
            if glass.any():
                occluder = render_interface(cam, self.field, self.deformation, self.pipe, drop=glass)
                entry['occluder_depth'], entry['occluder_alpha'] = occluder['depth'], occluder['alpha']
        return entry

    def __call__(self, cam):
        key = (cam.uid, float(cam.time))
        if key not in self.entries:
            self.entries[key] = {name: value.detach().cpu() for name, value in self.render(cam).items()}
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        return {name: value.cuda(non_blocking=True) for name, value in self.entries[key].items()}


def occlusion_settings(settings, direct):
    """The `occlusion` argument of first_hit for one DirectPass entry, or None for the literal Eq. (9)."""
    if settings.get('ablate_occlusion'):
        return None
    return {'depth': direct['occluder_depth'], 'alpha': direct['occluder_alpha'], 'tolerance': settings['occlusion_tolerance'],
            'alpha_min': settings['occlusion_alpha']}


def direct_pass_for(field, deformation, pipe, patches, settings, capacity=32):
    """The DirectPass of a model whose glass is fixed: its occluder pass leaves out the glass layer."""
    return DirectPass(field, deformation, pipe, capacity, patches, settings.get('glass_margin', 0.0))
