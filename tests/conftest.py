import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Development scene: synthetic, PINHOLE, masks aligned with the images.
DEV_SCENE = os.environ.get('VIGOR_TEST_SCENE', os.path.join(REPO_ROOT, 'data', 'datasets', 'datasets', '3D-FRONT-T', 'scene_1'))


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason='no CUDA device')
    skip_data = pytest.mark.skip(reason=f'dataset scene not found: {DEV_SCENE}')
    has_cuda = torch.cuda.is_available()
    has_data = os.path.isdir(DEV_SCENE)
    for item in items:
        if 'gpu' in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if 'data' in item.keywords and not has_data:
            item.add_marker(skip_data)


@pytest.fixture(scope='session')
def scene_dir():
    return DEV_SCENE


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope='session')
def trained_interface(scene_dir, tmp_path_factory):
    """
    A short Phase I run on the dev scene, shared by every test that needs a trained interface.

    400 iterations at 1/8 resolution with a small deformation MLP: 200 static warm-up iterations, then the
    deformation switches on. Tests that use this must be marked `gpu` and `data`.
    """
    from argparse import Namespace

    from tests.helpers import parse_interface_args
    from train_interface import training

    root = tmp_path_factory.mktemp('model')
    dataset, opt, pipe = parse_interface_args(
        ['-s', scene_dir, '-m', str(root), '-r', '8', '--eval', '--clean_fallback_original', '--deform_depth', '4', '--deform_width', '64']
        + ['--iterations', '400', '--warm_up', '200', '--reg_from_iter', '100', '--norm_loss_from_iter', '100', '--densify_from_iter', '100']
    )
    dataset.linear_rgb = True
    history = training(dataset, opt, pipe, testing_iterations=[1, 400], saving_iterations=[400])
    return Namespace(dataset=dataset, opt=opt, pipe=pipe, history=history, root=root, iterations=400)


@pytest.fixture(scope='session')
def trained_reflection(trained_interface):
    """
    A short Phase II run on top of `trained_interface`, with a glass plane fitted to it (or, if the short Phase I
    run left nothing to fit, a hand-set plane). Tests that use this must be marked `gpu` and `data`.
    """
    import hashlib
    from argparse import ArgumentParser, Namespace

    from arguments import ReflectionOptimizationParams
    from extract_plane import extract_plane
    from train_reflection import training

    root, dataset = trained_interface.root, trained_interface.dataset
    try:
        extract_plane(dataset)
    except RuntimeError:
        extract_plane(dataset, manual_plane=[0.0, 0.0, 1.0, 0.0])

    interface_files = [p for p in sorted((root / 'interface').rglob('*')) if p.suffix in ('.ply', '.pth')]
    digests = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in interface_files}

    parser = ArgumentParser()
    op = ReflectionOptimizationParams(parser)
    opt = op.extract(parser.parse_args(['--iterations', '300', '--warm_up', '150', '--refl_init_points', '5000', '--densify_from_iter', '100']))
    history = training(dataset, opt, trained_interface.pipe, testing_iterations=[1, 300], saving_iterations=[300])
    return Namespace(opt=opt, history=history, iterations=300, interface_digests=digests)
