"""
Score the test renders of any method on the glass region, against the current aligned data of a scene.

Every method (Deformable 3DGS, SpectroMotion, RT-Splatting, VIGOR-GS) saves its test split as
`renders/00000.png, 00001.png, ...` next to `gt/` with the same names, but the methods do not all order their test
views the same way, render at the same resolution, or read the same copy of the data. So this script does not trust
the file names or the method's own ground truth:

1. The reference is rebuilt from the scene's aligned data: colmap/images and colmap/masks of the frames listed under
   "test" in colmap/split.json, resized to the resolution VIGOR-GS is evaluated at, the same way VIGOR-GS resizes them
   (`utils/camera_utils.py:loadCam` for the size, PIL's default filter as in `utils/memory_utils.py:image_to_tensor`,
   and mask > 0 after the resize).
2. Each `gt/<i>.png` of the method is matched to one test frame by comparing thumbnails (one-to-one assignment).
   The PSNR between the method's own ground truth and the reference (`GT check PSNR`) then says whether the method was
   trained and tested on the same images. A low value (say below 35 dB) means it used other images (e.g. the
   distorted source frames) and its scores are not comparable.
3. A render whose size differs from the evaluation size is resized with the same filter as the reference.

Metrics per view, all on sRGB PNGs in [0, 1]:
    PSNR, SSIM, LPIPS (VGG)               full image (as metrics.py)
    Masked PSNR, Masked SSIM, Masked LPIPS  as metrics.py: both images multiplied by the glass mask, then the metric
                                            on the whole frame. Comparable with the existing VIGOR-GS / RT-Splatting
                                            tables, but inflated, because the zeroed pixels outside the mask match.
    PSNR in-mask                          the mean squared error over the glass pixels only (the true PSNR there)
    PSNR out-of-mask                      the same over the pixels outside the mask
    mask fraction                         share of the frame covered by the mask
Views whose mask is empty are left out of the masked metrics (as metrics.py does), and are recorded by name so they
cannot shift the other views, which metrics.py's per-view dict does.

Usage:
    python eval_masked_baselines.py --scene_dir <scene with colmap/> --resolution 2 \
        --renders <.../test/ours_40000/renders> --gt <.../test/ours_40000/gt> --out result.json --label "Deformable 3DGS"
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from utils.image_utils import psnr
from utils.loss_utils import ssim

THUMB_WIDTH = 128
GT_CHECK_MIN_PSNR = 35.0  # below this the method's ground truth is not our reference image
MAX_PSNR = 100.0  # PSNR of identical images
SCALAR_KEYS = ('PSNR', 'SSIM', 'LPIPS', 'Masked PSNR', 'Masked SSIM', 'Masked LPIPS', 'PSNR in-mask', 'PSNR out-of-mask', 'mask fraction', 'GT check PSNR')


def eval_size(width, height, resolution):
    """The image size VIGOR-GS / RT-Splatting train and test at for `-r resolution` (utils/camera_utils.py:loadCam)."""
    if resolution not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError(f'resolution must be one of 1, 2, 3, 4, 5, 6, 8 (got {resolution})')
    return round(width / resolution), round(height / resolution)


def split_test_frames(scene_dir):
    """Names (e.g. '00000002.jpg') of the test frames of the scene, in frames.json order."""
    colmap = os.path.join(scene_dir, 'colmap')
    with open(os.path.join(colmap, 'split.json')) as f:
        test = set(json.load(f)['test'])
    with open(os.path.join(colmap, 'frames.json')) as f:
        names = [frame['name'] for frame in json.load(f)['frames']]
    names = [name for name in names if name in test]
    if not names:
        raise RuntimeError(f'No test frame in {colmap}/split.json')
    return names


def load_reference(scene_dir, resolution):
    """{name: (image uint8 HxWx3, mask bool HxW)} of the test frames at the evaluation size, and that size (w, h)."""
    colmap = os.path.join(scene_dir, 'colmap')
    reference, size = {}, None
    for name in split_test_frames(scene_dir):
        image = Image.open(os.path.join(colmap, 'images', name)).convert('RGB')
        mask_path = os.path.join(colmap, 'masks', os.path.splitext(name)[0] + '.png')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f'No aligned glass mask for {name}: {mask_path}')
        mask = Image.open(mask_path).convert('L')
        if mask.size != image.size:
            raise RuntimeError(f'{mask_path} is {mask.size}, its image {image.size}: the mask is not aligned with colmap/images')
        frame_size = eval_size(*image.size, resolution)
        if size is None:
            size = frame_size
        elif frame_size != size:
            raise RuntimeError(f'Test frames of different sizes: {frame_size} and {size}')
        reference[name] = (np.asarray(image.resize(size)), np.asarray(mask.resize(size)) > 0)
    return reference, size


def thumbnail(image):
    """Small grey float image used to match a method's ground truth to a test frame."""
    width, height = image.size
    small = image.convert('L').resize((THUMB_WIDTH, max(1, round(THUMB_WIDTH * height / width))), Image.BILINEAR)
    return np.asarray(small, np.float32) / 255.0


def assign(cost):
    """One-to-one assignment minimising the summed cost (rows <= columns): list of column indices per row."""
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [int(c) for _, c in sorted(zip(rows, cols))]
    except ImportError:  # greedy fallback: cheapest pairs first
        row_to_col, used = {}, set()
        for flat in np.argsort(cost, axis=None):
            r, c = np.unravel_index(flat, cost.shape)
            if r not in row_to_col and c not in used:
                row_to_col[r] = c
                used.add(c)
        return [int(row_to_col[r]) for r in range(cost.shape[0])]


def match_views(gt_dir, reference):
    """{file in gt_dir: test frame name} by thumbnail MSE, one-to-one."""
    files = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not files:
        raise RuntimeError(f'No image in {gt_dir}')
    names = list(reference)
    if len(files) > len(names):
        raise RuntimeError(f'{gt_dir} has {len(files)} views but the scene has {len(names)} test frames: not the same test split')
    ref_thumbs = np.stack([thumbnail(Image.fromarray(reference[name][0])) for name in names])
    gt_thumbs = []
    for f in files:
        thumb = thumbnail(Image.open(os.path.join(gt_dir, f)).convert('RGB'))
        if thumb.shape != ref_thumbs.shape[1:]:
            raise RuntimeError(f'{gt_dir}/{f} has another aspect ratio than colmap/images (thumbnail {thumb.shape} vs {ref_thumbs.shape[1:]})')
        gt_thumbs.append(thumb)
    gt_thumbs = np.stack(gt_thumbs)
    cost = ((gt_thumbs[:, None] - ref_thumbs[None]) ** 2).mean(axis=(2, 3))
    cols = assign(cost)
    return {f: names[c] for f, c in zip(files, cols)}


def to_tensor(array, device):
    return torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0).to(device)


def region_psnr(render, gt, region):
    """PSNR from the squared error averaged over the pixels where `region` (1x1xHxW, 0/1) is set."""
    count = region.sum() * render.shape[1]
    if count == 0:
        return None
    mse = (((render - gt) ** 2) * region).sum() / count
    return capped_psnr(mse)


def capped_psnr(mse):
    """PSNR of a mean squared error, capped at MAX_PSNR so that identical images give a finite number."""
    return min(MAX_PSNR, (-10 * torch.log10(mse.clamp_min(1e-20))).item())


def load_render(path, size):
    image = Image.open(path).convert('RGB')
    original = image.size
    if image.size != size:
        image = image.resize(size)  # same filter as the reference
    return np.asarray(image), original


@torch.no_grad()
def evaluate(renders_dir, gt_dir, scene_dir, resolution, label='', lpips_model=None, device='cuda'):
    """Score one method; returns the result dict (see the module docstring)."""
    if lpips_model is None:
        from lpipsPyTorch import LPIPS

        lpips_model = LPIPS(net_type='vgg').to(device)
    reference, size = load_reference(scene_dir, resolution)
    matching = match_views(gt_dir, reference)
    warnings = []
    if len(matching) < len(reference):
        warnings.append(f'{len(matching)} views for {len(reference)} test frames')

    per_view, render_sizes, gt_sizes = {}, set(), set()
    for f, name in matching.items():
        render_path = os.path.join(renders_dir, f)
        if not os.path.exists(render_path):
            raise FileNotFoundError(f'{render_path} missing (its ground truth {gt_dir}/{f} exists)')
        ref_image, ref_mask = reference[name]
        render, render_size = load_render(render_path, size)
        own_gt, gt_size = load_render(os.path.join(gt_dir, f), size)
        render_sizes.add(render_size)
        gt_sizes.add(gt_size)

        r, g = to_tensor(render, device), to_tensor(ref_image, device)
        m = torch.from_numpy(ref_mask).float()[None, None].to(device)
        view = {
            'file': f,
            'PSNR': psnr(r, g).item(),
            'SSIM': ssim(r, g).item(),
            'LPIPS': lpips_model(r, g).item(),
            'GT check PSNR': capped_psnr(((to_tensor(own_gt, device) - g) ** 2).mean()),
            'mask fraction': m.mean().item(),
            'PSNR out-of-mask': region_psnr(r, g, 1.0 - m),
        }
        if ref_mask.any():
            view.update(
                {
                    'Masked PSNR': psnr(r * m, g * m).item(),
                    'Masked SSIM': ssim(r * m, g * m).item(),
                    'Masked LPIPS': lpips_model(r * m, g * m).item(),
                    'PSNR in-mask': region_psnr(r, g, m),
                }
            )
        per_view[name] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in view.items()}

    means = {}
    for key in SCALAR_KEYS:
        values = [v[key] for v in per_view.values() if v.get(key) is not None and np.isfinite(v[key])]
        if values:
            means[key] = float(np.mean(values))
    empty = sorted(name for name, (_, mask) in reference.items() if name in per_view and not mask.any())
    low_gt = sorted(name for name, v in per_view.items() if v['GT check PSNR'] < GT_CHECK_MIN_PSNR)
    if low_gt:
        warnings.append(
            f'{len(low_gt)}/{len(per_view)} views have GT check PSNR < {GT_CHECK_MIN_PSNR} dB: the method was not tested on '
            'the aligned colmap/images of this scene, so its scores are not comparable'
        )
    return {
        'label': label,
        'renders': os.path.abspath(renders_dir),
        'scene': os.path.abspath(scene_dir),
        'resolution': resolution,
        'eval_size': list(size),
        'render_sizes': sorted(list(s) for s in render_sizes),
        'gt_sizes': sorted(list(s) for s in gt_sizes),
        'n_views': len(per_view),
        'n_test_frames': len(reference),
        'views_with_empty_mask': empty,
        'warnings': warnings,
        'mean': means,
        'per_view': per_view,
    }


if __name__ == '__main__':
    parser = ArgumentParser(description='Score test renders on the glass region against the aligned scene data')
    parser.add_argument('--scene_dir', required=True, help='scene folder holding colmap/{images,masks,split.json,frames.json}')
    parser.add_argument('--resolution', '-r', type=int, required=True, help='the -r VIGOR-GS is evaluated at for this scene')
    parser.add_argument('--renders', required=True)
    parser.add_argument('--gt', required=True, help="the method's own gt/ folder, used only to match views and check its data")
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    result = evaluate(args.renders, args.gt, args.scene_dir, args.resolution, args.label)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ('label', 'eval_size', 'render_sizes', 'n_views', 'warnings', 'mean')}, indent=1))
