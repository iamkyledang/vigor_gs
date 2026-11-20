"""
Command-line settings of every VIGOR-GS step.

Each step script parses `ModelParams` and `PipelineParams` plus its own group, and writes what it used to
`<model>/<step>/config.json`. `ModelParams` (the scene, resolution, field architecture and Phase I ablations) is
also written to `<model>/config.json` by Phase I; later steps start from it and only change what is passed
explicitly (`load_model_args`), so a Phase II run cannot silently use another scene or architecture.

Loss weights follow the per-pixel normalization of the framework (docs/framework.pdf, "Reading the losses").
Values that the framework does not fix are implementation choices, listed in docs/implementation_details.md.
"""

import json
import os
import sys
from argparse import ArgumentParser, Namespace


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, sentinel: bool = False):
        group = parser.add_argument_group(name)
        self.names = []
        for key, value in list(vars(self).items()):
            if key == 'names':
                continue
            name = key.lstrip('_')
            self.names.append(name)
            flags = ['--' + name] + (['-' + name[0]] if key.startswith('_') else [])
            kwargs = {'default': None if sentinel else value}
            if isinstance(value, bool):
                kwargs['action'] = 'store_true'
            else:
                kwargs['type'] = type(value)
            group.add_argument(*flags, **kwargs)

    def extract(self, args) -> Namespace:
        values = {name: getattr(args, name) for name in self.names}
        if values.get('source_path'):
            values['source_path'] = os.path.abspath(values['source_path'])
        return Namespace(**values)


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ''
        self._model_path = ''
        self._resolution = 1  # downsampling factor of the COLMAP images
        self.data_device = 'cpu'  # where the uint8 images wait between iterations: cpu (host RAM) or cuda
        self.clean_images = 'clean_image'  # clean targets, under <scene>/colmap/ (aligned) or <scene>/
        self.init_points = 80000  # SfM points kept to initialize G_I (0 keeps all)
        self.sh_degree = 3
        self.seed = 0
        # Deformation MLPs D_I and D_R (Eq. 1), canonical-space deformation of Yang et al.
        self.deform_depth = 4
        self.deform_width = 128
        self.deform_xyz_freqs = 6
        self.deform_time_freqs = 6
        self.max_interface_points = 500000
        # Phase I ablations. They are part of the model, so they are stored with it and apply to every later step.
        self.ignore_clean_images = False  # no separation: one field fits the original images, no Phase II
        self.ablate_factorization = False  # no occupancy: the interface has a single opacity (p = 1)
        self.static_interface = False  # no D_I: G_I does not move over time
        super().__init__(parser, 'Model', sentinel)


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.depth_ratio = 0.0  # 2DGS depth: 0 = expected depth, 1 = median depth
        self.tile_size = 16  # reflected-ray renderer tiles
        self.ray_cutoff = 3.5  # Mahalanobis radius of a reflection Gaussian
        self.ray_epsilon = 0.0001  # offset of a reflected ray start from its glass hit, world units
        self.max_tile_entries = 64000000  # (Gaussian, tile) pairs of one reflected-ray render; an error if exceeded
        self.debug = False
        super().__init__(parser, 'Rendering')


class InterfaceOptimizationParams(ParamGroup):
    """Phase I: Eq. (3)-(4)."""

    def __init__(self, parser):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.occupancy_lr = 0.025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.deform_lr = 0.00016
        self.percent_dense = 0.01
        self.lambda_surface = 0.05  # lambda_s of Eq. (3)
        self.surface_from_iter = 1000
        self.lambda_glass = 0.05  # glass-mask prior (not in Eq. 3): lets Phase I geometry carry the glass patches
        self.lambda_dssim = 0.0  # optional D-SSIM term; the framework omits SSIM
        self.warm_up = 1000  # iterations before D_I is used (canonical field first)
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densification_interval = 100
        self.densify_grad_threshold = 0.0002
        self.min_presence = 0.005  # prune disks whose occupancy falls below this
        self.presence_reset_interval = 3000  # clamp the occupancy to 0.01 (2DGS/3DGS opacity reset); 0 disables it
        self.max_screen_size = 20  # pixels; after the first reset, larger disks are pruned
        self.test_interval = 5000
        self.checkpoint_interval = 5000
        self.report_interval = 100
        super().__init__(parser, 'Phase I optimization')


class GlassParams(ParamGroup):
    """The fixed glass patches P = {(n_k, d_k, S_k)} of Sec. 1."""

    def __init__(self, parser):
        self.glass_geometry = ''  # known patches (JSON); empty = extract them from the Phase I geometry
        self.max_patches = 4  # upper bound on K
        self.plane_threshold = 0.01  # RANSAC inlier distance, fraction of the scene extent
        self.plane_min_inliers = 100
        self.plane_max_points = 12000  # surface samples used by RANSAC
        self.plane_min_coverage = 0.5  # refuse patches that miss more than half of the annotated glass
        self.plane_min_precision = 0.5  # or that put glass where the annotation has none
        super().__init__(parser, 'Glass patches')


class ReflectionParams(ParamGroup):
    """Phase II model: Sec. 3 (rays, virtual camera, Fresnel weight) and its ablations."""

    def __init__(self, parser):
        self.max_reflection_points = 250000
        self.reflection_init_points = 50000
        self.eta_air = 1.0
        self.eta_glass = 1.5
        self.pure_fresnel = False  # h_psi = 0 (Eq. 11 without the learned correction)
        self.optical_bound = 0.0  # > 0 bounds h_psi to +-bound with tanh; 0 = unbounded, as in Eq. (11)
        self.occlusion_tolerance = 0.02  # a glass hit is occluded when the frozen depth is this fraction in front of it
        self.occlusion_alpha = 0.5  # ... and the frozen direct pass is at least this opaque there
        # Ablations of Phase II.
        self.ablate_occlusion = False  # literal Eq. (9): glass hits are not tested against the frozen G_I depth
        self.ablate_virtual_camera = False  # G_R is queried along the unreflected real rays
        self.pinhole_virtual_camera = False  # rays start at the Eq. (9b) pose c_v instead of each glass hit
        self.ablate_fresnel = False  # alpha is one learned constant on the glass instead of F(theta) + h_psi
        self.static_reflection = False  # no D_R: G_R does not move over time
        super().__init__(parser, 'Phase II model')


class ReflectionOptimizationParams(ParamGroup):
    """Phase II: Eq. (7)-(8)."""

    def __init__(self, parser):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.deform_lr = 0.00016
        self.optical_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_motion = 0.1  # lambda_m of Eq. (7)
        self.lambda_optical = 0.01  # lambda_o of Eq. (7)
        self.lambda_dssim = 0.0
        self.warm_up = 1000  # iterations before D_R is used
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densification_interval = 100
        self.densify_grad_threshold = 0.00002  # on 3D position gradients (there is no screen space for rays)
        self.min_presence = 0.005
        self.presence_reset_interval = 3000
        self.cache_views = 32  # frozen direct renders kept in host RAM
        self.test_interval = 5000
        self.checkpoint_interval = 5000
        self.report_interval = 100
        super().__init__(parser, 'Phase II optimization')


MODEL_CONFIG = 'config.json'


def save_config(path, *namespaces):
    values = {}
    for namespace in namespaces:
        values.update(vars(namespace))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(values, f, indent=2)


def load_model_args(model_args: Namespace) -> Namespace:
    """
    The `ModelParams` saved by Phase I, updated with the values given on this command line (a `ModelParams`
    parsed with `sentinel=True` holds None for everything not given).
    """
    path = os.path.join(model_args.model_path, MODEL_CONFIG)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: train Phase I first (train_interface.py -m {model_args.model_path})')
    with open(path) as f:
        saved = json.load(f)
    given = {k: v for k, v in vars(model_args).items() if v is not None and k != 'model_path'}
    architecture = ('sh_degree', 'deform_depth', 'deform_width', 'deform_xyz_freqs', 'deform_time_freqs', 'ignore_clean_images', 'ablate_factorization', 'static_interface')
    changed = [k for k in architecture if k in given and given[k] != saved.get(k) and not (isinstance(given[k], bool) and not given[k])]
    if changed:
        raise ValueError(f'{", ".join(changed)} belong to the trained Phase I model ({path}) and cannot change in a later step')
    saved.update({k: v for k, v in given.items() if not (isinstance(v, bool) and not v)})
    saved['model_path'] = model_args.model_path
    return Namespace(**saved)


def command_line():
    return ' '.join(sys.argv)
