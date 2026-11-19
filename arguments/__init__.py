"""VIGOR-GS configuration. Defaults target full-resolution quality runs."""
import json
import os
from argparse import Namespace


class ParamGroup:
    def __init__(self, parser, name, sentinel=False):
        group = parser.add_argument_group(name)
        self.names = []
        for key, value in list(vars(self).items()):
            if key == 'names':
                continue
            name = key.lstrip('_')
            self.names.append(name)
            flags = ['--' + name]
            if key.startswith('_'):
                flags.append('-' + name[0])
            kwargs = {'default': None if sentinel else value}
            if isinstance(value, bool):
                kwargs['action'] = 'store_true'
            else:
                kwargs['type'] = type(value)
            group.add_argument(*flags, **kwargs)

    def extract(self, args):
        values = {name: getattr(args, name) for name in self.names}
        if 'source_path' in values:
            values['source_path'] = os.path.abspath(values['source_path'])
        return Namespace(**values)


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ''
        self._model_path = ''
        self._resolution = 1
        self.sh_degree = 3
        self.data_device = 'cpu'
        self.eval = True
        self.clean_images = ''
        self.glass_geometry = ''
        self.deform_depth = 4
        self.deform_width = 128
        self.deform_xyz_freqs = 6
        self.deform_time_freqs = 6
        self.max_interface_points = 500000
        self.max_reflection_points = 250000
        self.init_points = 80000
        self.reflection_init_points = 50000
        self.max_patches = 4
        self.plane_threshold = 0.01
        self.plane_min_inliers = 100
        self.plane_min_coverage = 0.5
        self.plane_min_precision = 0.5
        self.plane_max_points = 12000
        self.eta_air = 1.0
        self.eta_glass = 1.5
        self.optical_bound = 0.1
        self.pure_fresnel = False
        self.seed = 0
        super().__init__(parser, 'Model and dataset', sentinel)


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.depth_ratio = 0.0
        self.tile_size = 16
        self.ray_cutoff = 3.5
        self.ray_epsilon = 0.0001
        self.max_tile_entries = 64000000
        self.debug = False
        super().__init__(parser, 'Rendering')


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30000
        self.reflection_iterations = 30000
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
        self.optical_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_surface = 0.05
        self.lambda_motion = 0.1
        self.lambda_optical = 0.01
        self.lambda_glass = 0.05
        self.lambda_dssim = 0.0
        self.warm_up = 1000
        self.surface_from_iter = 1000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densification_interval = 100
        self.densify_grad_threshold = 0.0002
        self.reflection_grad_threshold = 0.00002
        self.checkpoint_interval = 5000
        self.report_interval = 100
        super().__init__(parser, 'Optimization')


def get_combined_args(parser):
    args = parser.parse_args()
    if not args.model_path:
        parser.error('--model_path/-m is required')
    path = os.path.join(args.model_path, 'config.json')
    with open(path) as stream:
        saved = json.load(stream)
    saved.update({k: v for k, v in vars(args).items() if v is not None})
    return Namespace(**saved)
