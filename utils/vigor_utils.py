"""
Where the VIGOR-GS phases keep their results, and how to load them back.

    <model_path>/interface/     Phase I: cfg_args, point_cloud/iteration_N/point_cloud.ply, deform/iteration_N/deform.pth
    <model_path>/glass_plane.json
    <model_path>/reflection/    Phase II, same structure
"""

import copy
import os

import torch

from scene import Scene
from scene.deform_model import DeformModel
from scene.interface_model import InterfaceGaussianModel
from scene.reflection_model import ReflectionGaussianModel
from utils.fresnel import OpticalCorrection
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
    gaussians = InterfaceGaussianModel(dataset.sh_degree)
    scene = Scene(phase_args(dataset, INTERFACE_DIR), gaussians, load_iteration=iteration, shuffle=shuffle)
    deform = DeformModel.from_args(InterfaceGaussianModel.SCALE_DIM, dataset)
    deform.load_weights(scene.model_path, scene.loaded_iter)
    return scene, gaussians, deform


def save_reflection(model_path, iteration, gaussians, deform, correction=None):
    """Write a Phase II result under <model_path>/reflection."""
    reflection_path = os.path.join(model_path, REFLECTION_DIR)
    gaussians.save_ply(os.path.join(reflection_path, 'point_cloud', 'iteration_{}'.format(iteration), 'point_cloud.ply'))
    deform.save_weights(reflection_path, iteration)
    if correction is not None:
        optical_dir = os.path.join(reflection_path, 'optical', 'iteration_{}'.format(iteration))
        os.makedirs(optical_dir, exist_ok=True)
        torch.save(correction.state_dict(), os.path.join(optical_dir, 'optical.pth'))


def load_reflection(dataset, iteration=-1):
    """
    Load a trained Phase II result. Returns (gaussians, deform, correction or None, iteration).

    The optical correction h_psi is restored when the model was trained with --optical_correction.
    """
    reflection_path = os.path.join(dataset.model_path, REFLECTION_DIR)
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(reflection_path, 'point_cloud'))

    gaussians = ReflectionGaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(reflection_path, 'point_cloud', 'iteration_{}'.format(iteration), 'point_cloud.ply'))
    deform = DeformModel.from_args(ReflectionGaussianModel.SCALE_DIM, dataset)
    deform.load_weights(reflection_path, iteration)

    correction = None
    if getattr(dataset, 'optical_correction', False):
        correction = OpticalCorrection().cuda()
        correction.load_state_dict(torch.load(os.path.join(reflection_path, 'optical', 'iteration_{}'.format(iteration), 'optical.pth')))
    return gaussians, deform, correction, iteration
