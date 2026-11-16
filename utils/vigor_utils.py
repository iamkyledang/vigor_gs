"""
Where the VIGOR-GS phases keep their results, and how to load them back.

    <model_path>/interface/     Phase I: cfg_args, point_cloud/iteration_N/point_cloud.ply, deform/iteration_N/deform.pth
    <model_path>/glass_plane.json
    <model_path>/reflection/    Phase II, same structure
"""

import copy
import os

from scene import Scene
from scene.deform_model import DeformModel
from scene.interface_model import InterfaceGaussianModel

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
