#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith('_'):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, action='store_true')
                elif t == list:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, nargs='+')
                else:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument('--' + key, default=value, action='store_true')
                elif t == list:
                    group.add_argument('--' + key, ('-' + key[0:1]), default=value, nargs='+')
                else:
                    group.add_argument('--' + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ('_' + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        # A cfg_args written before a parameter existed does not mention it: fall back to its default.
        for key, value in vars(self).items():
            name = key[1:] if key.startswith('_') else key
            if not hasattr(group, name):
                setattr(group, name, value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ''
        self._model_path = ''
        self._images = 'images'
        self._resolution = -1
        self._white_background = False
        self.data_device = 'cuda'
        self.eval = False

        self.run_dim = 256
        self.rand_init = False

        self.env_scope_center = [0.0, 0.0, 0.0]
        self.env_scope_radius = 0.0
        self.xyz_axis = [0.0, 0.0, 0.0]

        # VIGOR-GS scene layout (see readVigorSceneInfo)
        self.clean_images = 'clean'  # folder of reflection-removed targets, absolute or relative to the scene
        self.clean_fallback_original = False  # use the original image where a clean one is missing
        self.linear_rgb = False  # convert targets from sRGB to linear RGB at load time

        # Deformation MLP architecture (scene/deform_model.py), shared by both VIGOR-GS fields
        self.deform_depth = 8
        self.deform_width = 256
        self.deform_xyz_multires = 10
        self.deform_time_multires = 6

        # Fresnel blend of VIGOR-GS (utils/fresnel.py)
        self.eta_air = 1.0
        self.eta_glass = 1.5
        self.optical_correction = False  # learn the small reflectance correction h_psi on top of Schlick

        super().__init__(parser, 'Loading Parameters', sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        self.init_stage = False
        super().__init__(parser, 'Pipeline Parameters')


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 61000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.occupancy_lr = 0.05
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.reflectance_lr = 0.005
        self.roughness_lr = 0.002
        self.transmissivity_lr = 0.01
        self.feature_lr = 0.002
        self.encoding_lr = 0.002
        self.mlp_lr = 0.0005

        self.percent_dense = 0.01
        self.occupancy_cull = 0.05
        self.lambda_dssim = 0.2
        self.lambda_lpips = 0.01
        self.lpips_loss_from_iter = 15000
        self.dist_loss_weight = 0
        self.dist_loss_from_iter = 0
        self.norm_loss_weight = 0.05
        self.norm_loss_from_iter = 0
        self.occupancy_decay_weight = 0.001
        self.mask_loss_weight = 0.01
        self.mask_loss_from_iter = -1
        self.transmissivity_loss_weight = 0.01
        self.consistency_loss_weight = 0.000002
        self.local_var_scale = 4

        self.densification_interval = 100
        self.occupancy_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.gsrgb_loss = False
        self.init_until_iter = 0
        self.alpha_until_iter = -1
        super().__init__(parser, 'Optimization Parameters')


class InterfaceOptimizationParams(ParamGroup):
    """Phase I of VIGOR-GS: the interface field G_I and its deformation MLP (train_interface.py)."""

    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.occupancy_lr = 0.05
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        # L_I = L_rgb(B_t, I_clean) + lambda_s * L_surf + mask and occupancy terms
        self.lambda_dssim = 0.2  # 0 gives the plain L1 of the framework document
        self.norm_loss_weight = 0.05  # L_surf: normal consistency of the surface pass
        self.norm_loss_from_iter = 700
        self.dist_loss_weight = 0.0  # L_surf: depth distortion of the volume pass
        self.dist_loss_from_iter = 0
        self.mask_loss_weight = 0.01  # BCE between the rendered glass map and the glass mask M_t
        self.occupancy_decay_weight = 0.001
        self.reg_from_iter = 700  # the mask and occupancy terms start here, once a coarse scene exists

        self.percent_dense = 0.01
        self.occupancy_cull = 0.05
        self.densification_interval = 100
        self.occupancy_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # Interface deformation MLP D_phi_I
        self.warm_up = 3000  # iterations trained as a static field before the deformation is switched on
        self.deform_lr_max_steps = 40_000
        self.time_noise_scale = 0.1  # timestamp jitter, in frame intervals, annealed to 0 (0 disables it)
        self.time_noise_until_iter = 20_000
        self.lambda_deform_reg = 0.0  # penalty on |d_xyz|, to keep the interface (and the glass) stable
        super().__init__(parser, 'Interface Optimization Parameters')


class ReflectionOptimizationParams(ParamGroup):
    """Phase II of VIGOR-GS: the reflection field G_R and its deformation MLP (train_reflection.py)."""

    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        # L_R = L_rgb(I_hat_t, I_t) + lambda_m * L_motion + lambda_o * L_opt
        self.lambda_dssim = 0.2  # 0 gives the plain L1 of the framework document
        self.lambda_motion = 0.01  # L_motion: mean |d_xyz| of the reflection deformation
        self.lambda_opt = 0.01  # L_opt: mean h_psi^2, only with --optical_correction
        self.optical_lr = 0.001
        self.gt_glass_mask = False  # gate alpha with the ground-truth glass mask instead of the rendered glass map

        # G_R has no SfM points: it starts from random points on the camera side of the glass
        self.refl_init_points = 100_000
        self.refl_init_radius_scale = 3.0  # radius of the initial ball, in units of the camera extent
        self.refl_init_color = 0.5

        self.percent_dense = 0.01
        self.min_opacity = 0.005
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # Reflection deformation MLP D_phi_R
        self.warm_up = 3000
        self.deform_lr_max_steps = 40_000
        self.time_noise_scale = 0.1
        self.time_noise_until_iter = 20_000
        super().__init__(parser, 'Reflection Optimization Parameters')


def get_combined_args(parser: ArgumentParser, cfg_subdir: str = ''):
    """
    Command-line arguments merged over the cfg_args saved at training time.

    cfg_subdir: folder below the model path that holds cfg_args (VIGOR-GS keeps one per phase, e.g. 'interface').
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = 'Namespace()'
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, cfg_subdir, 'cfg_args')
        print('Looking for config file in', cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print('Config file found: {}'.format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print('Config file not found at')
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
