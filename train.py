"""
VIGOR-GS end to end: Phase I -> glass patches -> Phase II -> test renders -> metrics.

    python train.py -s <scene> -m <model> [-r 2] [any flag of the steps below]

    train_interface.py   Phase I (Eq. 2-4)               skipped when its checkpoint reached its iterations,
    extract_glass.py     patches P (Sec. 1, Eq. 9)        resumed when it stopped earlier
    train_reflection.py  Phase II (Eq. 5-8)
    render.py            test views, with --components
    metrics.py           results.json

Every flag goes to the steps that know it, so a flag two steps share (--iterations, --lambda_dssim, the learning
rates) sets both phases. Settings for one step only go in --interface_args / --glass_args / --reflection_args, e.g.
--reflection_args "--iterations 20000 --lambda_motion 0.05". A flag that no step knows is an error. With
--ignore_clean_images (the no-separation ablation) there is no Phase II: the direct image is rendered and scored.
"""

import os
import shlex
import sys
from argparse import ArgumentParser, Namespace

import torch

import extract_glass
import metrics
import render
import train_interface
import train_reflection
from arguments import GlassParams, InterfaceOptimizationParams, ModelParams, PipelineParams, ReflectionOptimizationParams, ReflectionParams, load_model_args
from utils.model_io import GLASS_GEOMETRY

STEP_FLAGS = {'--resume', '--quiet'}


def known_flags():
    flags = set(STEP_FLAGS)
    for group in (ModelParams, PipelineParams, InterfaceOptimizationParams, GlassParams, ReflectionParams, ReflectionOptimizationParams):
        parser = ArgumentParser(add_help=False)
        group(parser)
        flags |= {option for action in parser._actions for option in action.option_strings}
    return flags


def unknown_flags(tokens):
    flags = known_flags()
    return [t.split('=')[0] for t in tokens if t.startswith('-') and not t.lstrip('-').replace('.', '', 1).isdigit() and t.split('=')[0] not in flags]


def without_source(tokens):
    """The tokens without -s / --source_path and its value: later steps read the scene from the model config."""
    kept, skip = [], False
    for token in tokens:
        if skip:
            skip = False
        elif token in ('-s', '--source_path'):
            skip = True
        elif not token.startswith('--source_path='):
            kept.append(token)
    return kept


def checkpoint_iteration(path):
    return torch.load(path, map_location='cpu', weights_only=False)['iteration'] if os.path.exists(path) else None


def run_phase(module, argv, phase, model_path):
    """Train one phase: skip it when finished, resume it when interrupted."""
    parsed = module.parse(argv, known=True)
    opt = parsed[1] if module is train_interface else parsed[3]
    done = checkpoint_iteration(os.path.join(model_path, phase, 'latest.pth'))
    if done is not None and done >= opt.iterations:
        print(f'[ INFO ] {phase}: already at iteration {done}, skipped')
        return
    module.main(argv + (['--resume'] if done is not None and '--resume' not in argv else []), known=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    outer = ArgumentParser(add_help=False)
    for step in ('interface', 'glass', 'reflection'):
        outer.add_argument(f'--{step}_args', default='')
    outer.add_argument('--only_interface', action='store_true', help='stop after Phase I')
    outer.add_argument('--skip_render', action='store_true')
    outer.add_argument('--skip_metrics', action='store_true')
    outer.add_argument('--skip_lpips', action='store_true')
    chain, shared = outer.parse_known_args(argv)
    if '-h' in shared or '--help' in shared:
        print(__doc__)
        return
    unknown = unknown_flags(shared + shlex.split(chain.interface_args) + shlex.split(chain.glass_args) + shlex.split(chain.reflection_args))
    if unknown:
        raise SystemExit(f'Unknown arguments (no step uses them): {" ".join(unknown)}')

    interface_args = shared + shlex.split(chain.interface_args)
    model_path = os.path.abspath(train_interface.parse(interface_args, known=True)[0].model_path)
    run_phase(train_interface, interface_args, 'interface', model_path)
    if chain.only_interface:
        return
    model_args = load_model_args(Namespace(model_path=model_path))
    later = without_source(shared)
    if not model_args.ignore_clean_images:
        if chain.glass_args or not os.path.exists(os.path.join(model_path, GLASS_GEOMETRY)):
            extract_glass.main(later + shlex.split(chain.glass_args), known=True)
        run_phase(train_reflection, later + shlex.split(chain.reflection_args), 'reflection', model_path)
    if not chain.skip_render:
        render.main(['-m', model_path, '--skip_train', '--components'])
    if not chain.skip_metrics:
        metrics.evaluate([model_path], skip_lpips=chain.skip_lpips)


if __name__ == '__main__':
    main()
