"""
VIGOR-GS inference (Sec. 4, Eq. 12-13): render the train/test views of a trained model.

    B_* = R_2D(D_I(G_I^0, t_*), C_*)
    R_* = R_3D^refl(D_R(G_R^0, t_*), C_*, P; V_dyn)
    I_hat_* = (1 - alpha_*) B_* + alpha_* R_*

As at training, each pixel selects the first visible glass patch (glass_geometry.json, with the same occlusion rule);
test glass masks are never used for blending.

    python render.py -m <model> [--skip_train] [--components] [--direct_only]

Writes <model>/<split>/<method>/{renders, gt, glass_masks, valid_masks}/<view>.png, method = ours_<iteration> or,
with --direct_only (Phase I only, e.g. the no-separation ablation), direct_<iteration>. --components adds direct/,
reflection/, alpha/, patches/ (selected patch per pixel), clean/ (the clean target, for the transmission metric)
and virtual_cameras.json (the Eq. 9b pose of every visible patch). Also comparison.mp4 (ours | GT) and
<model>/render_stats.json (CUDA-event render time per frame, excluding loading and saving).
"""

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, load_model_args
from gaussian_renderer.composition import render_composite
from gaussian_renderer.interface import render_interface
from scene import Scene
from scene.deformation import Deformation
from scene.gaussian_model import ReflectionField
from utils.checkpoint_utils import load_checkpoint, restore_phase
from utils.color_utils import linear2srgb
from utils.model_io import direct_pass_for, load_glass, load_interface

PATCH_COLORS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]], np.uint8)


def save_image(tensor, path, linear=False):
    if linear:
        tensor = linear2srgb(tensor)
    array = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).save(path)


def patch_image(patch):
    """Selected patch per pixel as colors, black off the glass."""
    ids = patch.cpu().numpy()
    image = np.zeros(ids.shape + (3,), np.uint8)
    image[ids >= 0] = PATCH_COLORS[ids[ids >= 0] % len(PATCH_COLORS)]
    return image


def create_comparison_video(folder, names, fps=30.0):
    """MP4 of the saved sRGB pairs: ours on the left, GT on the right, native sizes centered on a common canvas."""
    import cv2

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video FPS must be positive and finite.')
    folder, names = Path(folder), list(names)
    if not names:
        raise ValueError('No frames for the comparison video.')
    width = height = 0
    for name in names:
        with Image.open(folder / 'renders' / (name + '.png')) as ours, Image.open(folder / 'gt' / (name + '.png')) as gt:
            if ours.size != gt.size:
                raise ValueError(f'Comparison size mismatch for {name}')
            width, height = max(width, ours.width), max(height, ours.height)
    header = 40
    canvas_height = height + header + (height + header) % 2  # even dimensions for the codec
    path = folder / 'comparison.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (2 * width, canvas_height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f'Cannot open an MPEG-4 encoder for {path}')
    try:
        for name in names:
            frame = np.zeros((canvas_height, 2 * width, 3), np.uint8)
            for side, sub in enumerate(('renders', 'gt')):
                with Image.open(folder / sub / (name + '.png')) as source:
                    image = np.asarray(source.convert('RGB'))
                h, w = image.shape[:2]
                x, y = side * width + (width - w) // 2, header + (height - h) // 2
                frame[y:y + h, x:x + w] = image
            cv2.putText(frame, 'Ours', (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'GT', (width + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f'No comparison video was written to {path}')
    return path


def load_reflection(model_args, scene):
    """Phase II of a model: (field, deformation, optical, constant, config, iteration); config holds its params and pipeline."""
    from gaussian_renderer.composition import build_blend

    folder = os.path.join(model_args.model_path, 'reflection')
    with open(os.path.join(folder, 'config.json')) as f:
        config = Namespace(**json.load(f))
    state = load_checkpoint(os.path.join(folder, 'latest.pth'))
    field = ReflectionField(model_args.sh_degree, config.max_reflection_points)
    deformation = Deformation(3, model_args, scene.center, scene.extent, static=config.static_reflection)
    restore_phase(state, field, deformation)
    optical, constant = build_blend(config, scene.center, scene.extent)
    for name, module in (('optical', optical), ('constant', constant)):
        if module is not None:
            module.load_state_dict(state['modules'][name])
            module.cuda().eval().requires_grad_(False)
    return field, deformation, optical, constant, config, state['iteration']


@torch.no_grad()
def render_sets(model_args, skip_train=False, skip_test=False, components=False, direct_only=False, time=None, video_fps=30.0, video=True):
    root = Path(model_args.model_path)
    scene = Scene(model_args, load_clean=components)
    interface, interface_deformation, iteration = load_interface(model_args, scene)
    if direct_only:
        with open(root / 'interface' / 'config.json') as f:
            pipe = Namespace(depth_ratio=json.load(f)['depth_ratio'], debug=False)
        name = f'direct_{iteration}'
    else:
        reflection, reflection_deformation, optical, constant, config, iteration = load_reflection(model_args, scene)
        patches, glass_record = load_glass(model_args)
        settings = glass_record.get('settings', {'ablate_occlusion': True})
        pipe = config
        name = f'ours_{iteration}'
        direct_pass = direct_pass_for(interface, interface_deformation, pipe, patches, settings, capacity=1)  # every frame rendered, not cached

    stats_path = root / 'render_stats.json'
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for split, cameras, skip in (('train', scene.getTrainCameras(), skip_train), ('test', scene.getTestCameras(), skip_test)):
        if skip or not cameras:
            continue
        folder = root / split / name
        subfolders = ['renders', 'gt', 'glass_masks', 'valid_masks'] + (['direct', 'reflection', 'alpha', 'patches', 'clean'] if components else [])
        for sub in subfolders:
            (folder / sub).mkdir(parents=True, exist_ok=True)
        milliseconds, poses = [], {}
        for cam in tqdm(cameras, desc=f'Rendering {split}'):
            if time is not None:
                cam.time = time
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            if direct_only:
                out = render_interface(cam, interface, interface_deformation, pipe)
                image = out['direct']
            else:
                out = render_composite(cam, direct_pass(cam), reflection, reflection_deformation, patches, settings, config, pipe, optical, constant)
                image = out['composite']
            end.record()
            torch.cuda.synchronize()
            milliseconds.append(begin.elapsed_time(end))
            filename = cam.image_name + '.png'
            save_image(image, folder / 'renders' / filename, linear=True)
            save_image(cam.original_srgb, folder / 'gt' / filename)
            save_image(cam.glass_mask.float(), folder / 'glass_masks' / filename)
            save_image(cam.valid_mask.float(), folder / 'valid_masks' / filename)
            if components:
                save_image(out['direct'], folder / 'direct' / filename, linear=True)
                if cam.has_clean:
                    save_image(cam.clean_linear, folder / 'clean' / filename, linear=True)
                if not direct_only:
                    save_image(out['reflection'], folder / 'reflection' / filename, linear=True)
                    save_image(out['alpha'], folder / 'alpha' / filename)
                    Image.fromarray(patch_image(out['geometry']['patch'])).save(folder / 'patches' / filename)
                    poses[cam.image_name] = {str(k): {'center': p['center'].tolist(), 'rotation': p['rotation'].tolist(), 'reference_pixel': p['reference_pixel']}
                                             for k, p in enumerate(out['poses']) if p is not None}
        if poses:
            (folder / 'virtual_cameras.json').write_text(json.dumps(poses, indent=1))
        timing = {'frames': len(cameras), 'mean_ms': float(np.mean(milliseconds)), 'fps': 1000.0 / float(np.mean(milliseconds)),
                  'peak_vram_mb': torch.cuda.max_memory_allocated() / 2**20}
        stats.setdefault(split, {})[name] = timing
        if video:
            ordered = sorted(cameras, key=lambda c: (c.sequence, c.sensor_id, c.time, c.image_name))
            create_comparison_video(folder, [c.image_name for c in ordered], video_fps)
        print(f'[ INFO ] {split}/{name}: {len(cameras)} views, {timing["mean_ms"]:.1f} ms per frame')
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def main(argv=None):
    parser = ArgumentParser(description='VIGOR-GS: render the train/test views of a trained model')
    model = ModelParams(parser, sentinel=True)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--components', action='store_true', help='also save direct, reflection, alpha, patches, clean, virtual cameras')
    parser.add_argument('--direct_only', action='store_true', help='render B_t of Phase I only')
    parser.add_argument('--time', type=float, default=None, help='render every view at this normalized time')
    parser.add_argument('--video_fps', type=float, default=30.0)
    parser.add_argument('--no_video', action='store_true')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.time is not None and not 0 <= args.time <= 1:
        parser.error('--time must lie in [0, 1]')
    model_args = model.extract(args)
    if not model_args.model_path:
        parser.error('-m is required')
    model_args.model_path = os.path.abspath(model_args.model_path)
    model_args = load_model_args(model_args)
    render_sets(model_args, args.skip_train, args.skip_test, args.components, args.direct_only or model_args.ignore_clean_images,
                args.time, args.video_fps, not args.no_video)


if __name__ == '__main__':
    main()
