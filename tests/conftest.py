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
