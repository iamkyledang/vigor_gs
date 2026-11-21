"""eval_masked_baselines.py: scoring another method's test renders on the glass region of the aligned scene data."""

import os
import random

import numpy as np
import pytest
import torch
from PIL import Image

from eval_masked_baselines import eval_size, evaluate, load_reference, match_views, split_test_frames
from utils.image_utils import psnr
from utils.loss_utils import ssim

pytestmark = pytest.mark.data


class NoLPIPS:
    """Stand-in for the VGG LPIPS so the tests need neither the weights download nor a GPU."""

    def __call__(self, x, y):
        return (x - y).abs().mean()


def fake_method(scene_dir, root, size=None, noise=4.0, transform=None, seed=0):
    """
    A method folder like Deformable 3DGS writes it: test views in a shuffled order as 00000.png ..., gt/ holding the
    COLMAP image (resized to `size` if given, optionally transformed), renders/ the same plus Gaussian noise.
    Returns {file: frame name}.
    """
    rng = np.random.default_rng(seed)
    names = split_test_frames(scene_dir)
    random.Random(seed).shuffle(names)
    os.makedirs(os.path.join(root, 'gt'))
    os.makedirs(os.path.join(root, 'renders'))
    truth = {}
    for i, name in enumerate(names):
        image = Image.open(os.path.join(scene_dir, 'colmap', 'images', name)).convert('RGB')
        if size is not None:
            image = image.resize(size)
        if transform is not None:
            image = transform(image)
        gt = np.asarray(image).astype(np.float32)
        render = np.clip(gt + rng.normal(0.0, noise, gt.shape), 0, 255).astype(np.uint8)
        f = f'{i:05d}.png'
        image.save(os.path.join(root, 'gt', f))
        Image.fromarray(render).save(os.path.join(root, 'renders', f))
        truth[f] = name
    return truth


def test_eval_size_follows_loadcam():
    assert eval_size(1927, 1080, 2) == (964, 540)
    assert eval_size(3271, 1224, 2) == (1636, 612)
    assert eval_size(960, 540, 1) == (960, 540)
    with pytest.raises(ValueError):
        eval_size(960, 540, 7)


def test_reference_mask_matches_vigor_loader(scene_dir):
    """Same size and mask rule as utils/camera_utils.py:loadCam (PIL resize, then > 0)."""
    reference, size = load_reference(scene_dir, 2)
    name = next(iter(reference))
    image, mask = reference[name]
    assert image.shape == (size[1], size[0], 3) and mask.shape == (size[1], size[0])
    raw = Image.open(os.path.join(scene_dir, 'colmap', 'masks', os.path.splitext(name)[0] + '.png')).convert('L')
    assert np.array_equal(mask, np.asarray(raw.resize(size)) > 0)


def test_views_are_matched_whatever_their_order(scene_dir, tmp_path):
    truth = fake_method(scene_dir, str(tmp_path))
    reference, _ = load_reference(scene_dir, 1)
    assert match_views(str(tmp_path / 'gt'), reference) == truth


def test_scores_and_masked_definition(scene_dir, tmp_path):
    truth = fake_method(scene_dir, str(tmp_path), noise=4.0)
    result = evaluate(str(tmp_path / 'renders'), str(tmp_path / 'gt'), scene_dir, 1, lpips_model=NoLPIPS(), device='cpu')
    assert result['n_views'] == len(truth) and not result['warnings']
    assert result['mean']['GT check PSNR'] == pytest.approx(100.0)  # PNG copies of the reference images

    # sigma 4/255 noise: PSNR = -20 log10(4/255) = 36.1 dB, everywhere in the frame (clipping only raises it)
    for key in ('PSNR', 'PSNR in-mask', 'PSNR out-of-mask'):
        assert 35.5 < result['mean'][key] < 37.5, key
    assert result['mean']['Masked PSNR'] > result['mean']['PSNR in-mask']  # the zeroed outside pixels inflate it

    # Masked SSIM / PSNR follow metrics.py exactly: both images times the mask, then the whole-frame metric.
    f, name = next(iter(truth.items()))
    view = result['per_view'][name]
    reference, _ = load_reference(scene_dir, 1)
    g = torch.from_numpy(reference[name][0]).permute(2, 0, 1).contiguous().float().div(255)[None]
    r = torch.from_numpy(np.asarray(Image.open(tmp_path / 'renders' / f))).permute(2, 0, 1).contiguous().float().div(255)[None]
    m = torch.from_numpy(reference[name][1]).float()[None, None]
    assert view['Masked PSNR'] == pytest.approx(psnr(r * m, g * m).item(), abs=1e-4)
    assert view['Masked SSIM'] == pytest.approx(ssim(r * m, g * m).item(), abs=1e-5)
    assert view['mask fraction'] == pytest.approx(m.mean().item())


def test_renders_at_another_resolution_are_resized(scene_dir, tmp_path):
    """A method rendering at full size is scored at the -r 2 size of VIGOR-GS."""
    width, height = Image.open(os.path.join(scene_dir, 'colmap', 'images', split_test_frames(scene_dir)[0])).size
    fake_method(scene_dir, str(tmp_path), noise=2.0)
    result = evaluate(str(tmp_path / 'renders'), str(tmp_path / 'gt'), scene_dir, 2, lpips_model=NoLPIPS(), device='cpu')
    assert result['eval_size'] == list(eval_size(width, height, 2))
    assert result['render_sizes'] == [[width, height]]
    assert result['mean']['GT check PSNR'] > 40.0 and not result['warnings']


def test_other_images_are_flagged(scene_dir, tmp_path):
    """A method trained on other images (here shifted by 6 px, like unrectified frames) is reported, not hidden."""
    shift = lambda image: Image.fromarray(np.roll(np.asarray(image), 6, axis=1))  # noqa: E731
    fake_method(scene_dir, str(tmp_path), transform=shift)
    result = evaluate(str(tmp_path / 'renders'), str(tmp_path / 'gt'), scene_dir, 1, lpips_model=NoLPIPS(), device='cpu')
    assert result['mean']['GT check PSNR'] < 35.0
    assert any('not comparable' in w for w in result['warnings'])
