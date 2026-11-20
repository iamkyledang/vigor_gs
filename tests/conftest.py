import os
import sys
from argparse import Namespace

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason='no CUDA device')
    if torch.cuda.is_available():
        return
    for item in items:
        if 'gpu' in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope='session')
def synthetic_scene(tmp_path_factory):
    """The synthetic glass scene of tests/synthetic.py: 12 views at 64 x 48."""
    from tests.synthetic import write_scene

    root = tmp_path_factory.mktemp('scene')
    truth = write_scene(str(root))
    return Namespace(path=str(root), **truth)


def model_args(source_path, model_path='', **overrides):
    """A complete ModelParams namespace with its defaults, for tests that build a Scene directly."""
    from argparse import ArgumentParser

    from arguments import ModelParams

    parser = ArgumentParser()
    params = ModelParams(parser)
    args = params.extract(parser.parse_args(['-s', source_path, '-m', model_path]))
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
