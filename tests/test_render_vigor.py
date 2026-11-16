"""Module 10: inference and the image layout that metrics.py evaluates."""

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.gpu, pytest.mark.data]


@pytest.fixture(scope='module')
def rendered(trained_interface, trained_reflection):
    from render_vigor import render_sets

    label = render_sets(trained_interface.dataset, trained_interface.pipe, skip_train=True)
    return trained_interface.root / 'test' / label, label


def read(path):
    return np.asarray(Image.open(path)).astype(np.float32) / 255.0


def test_layout_matches_what_metrics_reads(rendered, trained_reflection):
    out_dir, label = rendered
    assert label == f'ours_{trained_reflection.iterations}'
    names = sorted(p.name for p in (out_dir / 'renders').iterdir())
    assert len(names) == 34 and names[0] == '00000.png'
    for folder in ('gt', 'transparent_masks'):
        assert sorted(p.name for p in (out_dir / folder).iterdir()) == names
    for label in ('direct', 'reflection', 'reflection_contribution', 'alpha', 'glass_map'):
        assert (out_dir / 'vis' / f'{label}_00000.png').exists()
    assert not (out_dir.parent.parent / 'train').exists()  # --skip_train


def test_images_are_consistent(rendered):
    out_dir, _ = rendered
    render, gt = read(out_dir / 'renders' / '00003.png'), read(out_dir / 'gt' / '00003.png')
    mask = read(out_dir / 'transparent_masks' / '00003.png')
    direct, alpha = read(out_dir / 'vis' / 'direct_00003.png'), read(out_dir / 'vis' / 'alpha_00003.png')

    assert render.shape == gt.shape and render.shape[2] == 3
    assert mask.shape == render.shape[:2] and set(np.unique(mask)) <= {0.0, 1.0}
    assert alpha.shape == render.shape[:2]

    # Where the reflection weight is zero, the composite is the direct image.
    off_glass = alpha == 0
    assert off_glass.any()
    assert np.abs(render[off_glass] - direct[off_glass]).max() <= 1 / 255 + 1e-6
    # The short run is rough, but the render must resemble the ground truth.
    assert np.mean((render - gt) ** 2) < 0.05


def test_metrics_runs_on_the_rendered_set(rendered, trained_interface):
    import json

    from metrics import evaluate

    evaluate([str(trained_interface.root)])
    with open(trained_interface.root / 'results.json') as f:
        results = json.load(f)
    method = rendered[1]
    for key in ('SSIM', 'PSNR', 'LPIPS', 'Masked SSIM', 'Masked PSNR', 'Masked LPIPS'):
        assert np.isfinite(results[method][key]), key
    assert results[method]['PSNR'] > 13.0
    with open(trained_interface.root / 'per_view.json') as f:
        assert len(json.load(f)[method]['PSNR']) == 34
