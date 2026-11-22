"""
Where the VIGOR-GS phases keep their results, and how to load them back.

    <model_path>/interface/     Phase I: cfg_args, point_cloud/iteration_N/point_cloud.ply, deform/iteration_N/deform.pth
    <model_path>/glass_plane.json
    <model_path>/reflection/    Phase II: same structure, plus blend/iteration_N/blend.pth (the alpha parameters)
"""

import copy
import os

import torch

from scene import Scene
from scene.deform_model import DeformModel
from scene.interface_model import InterfaceGaussianModel
from scene.reflection_model import ReflectionGaussianModel
from utils.fresnel import BlendModel
from utils.system_utils import searchForMaxIteration

INTERFACE_DIR = 'interface'
REFLECTION_DIR = 'reflection'
PLANE_FILE = 'glass_plane.json'


def phase_args(dataset, phase_dir):
    """A copy of the model parameters whose model_path points at one phase's folder (what `Scene` expects)."""
    args = copy.copy(dataset)
    args.model_path = os.path.join(dataset.model_path, phase_dir)
    return args


def load_interface(dataset, iteration=-1, shuffle=False):
    """
    Load a trained Phase I result: the cameras, the interface field and its deformation MLP.

    dataset: model parameters with model_path at the model root. Returns (scene, gaussians, deform).
    """
    gaussians = InterfaceGaussianModel(dataset.sh_degree, factorized=not getattr(dataset, 'ablate_factorization', False))
    scene = Scene(phase_args(dataset, INTERFACE_DIR), gaussians, load_iteration=iteration, shuffle=shuffle)
    deform = DeformModel.from_args(InterfaceGaussianModel.SCALE_DIM, dataset)
    deform.load_weights(scene.model_path, scene.loaded_iter)
    return scene, gaussians, deform


def save_reflection(model_path, iteration, blend, gaussians=None, deform=None):
    """
    Write a Phase II result under <model_path>/reflection.

    The blend parameters are always saved (the file is all but empty for pure Fresnel mixing). The field and its
    deformation MLP are left out in the shared-field ablation, which has no reflection field of its own.
    """
    reflection_path = os.path.join(model_path, REFLECTION_DIR)
    blend_dir = os.path.join(reflection_path, 'blend', 'iteration_{}'.format(iteration))
    os.makedirs(blend_dir, exist_ok=True)
    torch.save(blend.state_dict(), os.path.join(blend_dir, 'blend.pth'))
    if gaussians is not None:
        gaussians.save_ply(os.path.join(reflection_path, 'point_cloud', 'iteration_{}'.format(iteration), 'point_cloud.ply'))
        deform.save_weights(reflection_path, iteration)


def load_reflection(dataset, iteration=-1):
    """
    Load a trained Phase II result. Returns (gaussians, deform, blend, iteration).

    In the shared-field ablation gaussians and deform are None: the caller renders the interface field instead.
    """
    reflection_path = os.path.join(dataset.model_path, REFLECTION_DIR)
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(reflection_path, 'blend'))

    blend = BlendModel.from_args(dataset).cuda()
    blend.load_state_dict(torch.load(os.path.join(reflection_path, 'blend', 'iteration_{}'.format(iteration), 'blend.pth')))

    if getattr(dataset, 'shared_reflection_field', False):
        return None, None, blend, iteration

    gaussians = ReflectionGaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(reflection_path, 'point_cloud', 'iteration_{}'.format(iteration), 'point_cloud.ply'))
    deform = DeformModel.from_args(ReflectionGaussianModel.SCALE_DIM, dataset)
    deform.load_weights(reflection_path, iteration)
    return gaussians, deform, blend, iteration
