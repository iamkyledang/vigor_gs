"""
Image metrics of the renders written by render.py, streamed one view at a time, in sRGB.

Per method folder (<model>/<split>/<method>/):

    PSNR, SSIM, LPIPS              full image against the original image (LPIPS VGG, inputs in [-1, 1])
    Valid PSNR                     valid pixels only (undistortion borders excluded)
    Glass PSNR, Opaque PSNR        annotated glass / non-glass valid pixels, normalized by their own count
    Masked PSNR/SSIM/LPIPS (RT)    both images multiplied by the glass mask, full-image average: the RT-Splatting
                                   protocol, kept for comparison (the zero background inflates it)
    LPIPS (RT input)               LPIPS on [0, 1] inputs, the RT-Splatting convention
    Transmission PSNR / Glass ...  the direct image B against the clean target, when render.py --components saved
                                   direct/ and clean/: how well the reflection-free layer is recovered

plus performance: FPS (render_stats.json) and the training time of the optimization loops of both phases
(<phase>/training.jsonl, summed over resumed sessions).

    python metrics.py -m <model> [<model> ...] [--skip_lpips] [--performance_only]

Writes <model>/results.json (means) and <model>/per_view.json.
"""

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.image_utils import masked_psnr
from utils.loss_utils import ssim


def read_image(path, device, gray=False):
    array = np.asarray(Image.open(path).convert('L' if gray else 'RGB')).copy()
    value = torch.from_numpy(array).float().div_(255)
    return (value[None, None] if gray else value.permute(2, 0, 1)[None]).to(device)


def phase_training_seconds(path, iteration=None):
    """Sum over sessions of the last elapsed time of each session (records up to `iteration`)."""
    path = Path(path)
    if not path.exists():
        return None
    sessions = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        elapsed = record.get('elapsed_seconds')
        if elapsed is None or not math.isfinite(elapsed) or elapsed < 0 or (iteration is not None and record['iteration'] > iteration):
            continue
        key = record.get('session_id', 'legacy')
        sessions[key] = max(sessions.get(key, 0.0), elapsed)
    return sum(sessions.values()) if sessions else None


def performance_summary(root, method, split, frames):
    root = Path(root)
    stats_path = root / 'render_stats.json'
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    timing = stats.get(split, {}).get(method)
    fps = timing['fps'] if timing and timing.get('frames') == frames else None
    iteration = int(method.rsplit('_', 1)[-1]) if method.rsplit('_', 1)[-1].isdigit() else None
    direct = method.startswith('direct_')
    phases = ('interface',) if direct else ('interface', 'reflection')
    seconds = {phase: phase_training_seconds(root / phase / 'training.jsonl', iteration if phase == phases[-1] else None) for phase in phases}
    total = sum(seconds.values()) if all(v is not None for v in seconds.values()) else None
    readable = None
    if total is not None:
        hours, minutes = divmod(int(total // 60), 60)
        readable = f'{hours} hours and {minutes} mins'
    return {'FPS': fps, 'Training Time': readable, 'Training Time (seconds)': total, 'Training Time by Phase (seconds)': seconds}


def mean_of(values, key):
    present = [v[key] for v in values if v.get(key) is not None]
    return float(np.mean(present)) if present else None


@torch.no_grad()
def evaluate_view(folder, name, device, perceptual):
    prediction = read_image(folder / 'renders' / name, device)
    target = read_image(folder / 'gt' / name, device)
    if prediction.shape != target.shape:
        raise ValueError(f'Size mismatch for {folder / name}')
    mask = read_image(folder / 'glass_masks' / name, device, True)[0] > 0.5
    valid_path = folder / 'valid_masks' / name
    valid = read_image(valid_path, device, True)[0] > 0.5 if valid_path.exists() else torch.ones_like(mask)
    p, t = prediction[0], target[0]
    record = {
        'PSNR': masked_psnr(p, t, torch.ones_like(mask)),
        'SSIM': float(ssim(prediction, target)),
        'Valid PSNR': masked_psnr(p, t, valid),
        'Glass PSNR': masked_psnr(p, t, mask & valid),
        'Opaque PSNR': masked_psnr(p, t, ~mask & valid),
        'Masked PSNR (RT)': masked_psnr(p * mask, t * mask, torch.ones_like(mask)),
        'Masked SSIM (RT)': float(ssim(prediction * mask, target * mask)) if mask.any() else None,
    }
    if perceptual is not None:
        record['LPIPS'] = float(perceptual(prediction * 2 - 1, target * 2 - 1))
        record['LPIPS (RT input)'] = float(perceptual(prediction, target))
        record['Masked LPIPS (RT)'] = float(perceptual(prediction * mask, target * mask)) if mask.any() else None
    if (folder / 'direct' / name).exists() and (folder / 'clean' / name).exists():
        direct, clean = read_image(folder / 'direct' / name, device)[0], read_image(folder / 'clean' / name, device)[0]
        record['Transmission PSNR'] = masked_psnr(direct, clean, valid)
        record['Glass Transmission PSNR'] = masked_psnr(direct, clean, mask & valid)
    return record


def evaluate(model_paths, device='cuda', skip_lpips=False, split='test'):
    perceptual = None
    if not skip_lpips:
        from lpipsPyTorch import LPIPS

        perceptual = LPIPS(net_type='vgg').to(device).eval()
    for model_path in model_paths:
        root = Path(model_path)
        parent = root / split
        if not parent.exists():
            raise FileNotFoundError(f'No {split} renders under {root}: run render.py first')
        results, per_view = {}, {}
        for folder in sorted(p for p in parent.iterdir() if (p / 'renders').is_dir()):
            names = sorted(p.name for p in (folder / 'renders').glob('*.png'))
            if not names:
                raise ValueError(f'No renders in {folder}')
            values = {name: evaluate_view(folder, name, device, perceptual) for name in tqdm(names, desc=folder.name)}
            keys = list(dict.fromkeys(k for v in values.values() for k in v))
            summary = {key: mean_of(values.values(), key) for key in keys}
            summary.update(performance_summary(root, folder.name, split, len(names)))
            results[folder.name], per_view[folder.name] = summary, values
            print(folder.name, json.dumps(summary, indent=2))
        if not results:
            raise ValueError(f'No method folders under {parent}')
        (root / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
        (root / 'per_view.json').write_text(json.dumps(per_view, indent=2, allow_nan=False))
    return results


def update_performance(model_paths, split='test'):
    """Refresh the performance fields of an existing results.json without recomputing image metrics."""
    for model_path in model_paths:
        root = Path(model_path)
        results = json.loads((root / 'results.json').read_text())
        for method in results:
            folder = root / split / method / 'renders'
            if folder.is_dir():
                results[method].update(performance_summary(root, method, split, sum(1 for _ in folder.glob('*.png'))))
        (root / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-m', '--model_paths', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip_lpips', action='store_true', help='omit LPIPS (smoke tests)')
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--performance_only', action='store_true')
    args = parser.parse_args()
    if args.performance_only:
        update_performance(args.model_paths, args.split)
    else:
        evaluate(args.model_paths, args.device, args.skip_lpips, args.split)
