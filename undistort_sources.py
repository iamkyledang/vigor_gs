"""
VIGOR-GS: bring the glass masks and clean images of a scene onto the undistorted COLMAP images.

Masks (frames.json 'source_mask') and clean images (<scene>/clean_image/<split>/images/) are made on the source
images <scene>/<split>/images/, which for VSGD and Wayve are the distorted originals. COLMAP's undistorted images
differ in size and geometry, and the distorted camera model was not kept. For every COLMAP camera this script fits
that model from SIFT matches between source and COLMAP images (utils/undistort_utils.py; radial for mildly distorted
lenses such as VSGD, fisheye for wide-angle ones such as WayveScenes101, chosen automatically), checks that the warped
source images line up with the COLMAP images, and writes:

    <out>/colmap/masks/<stem>.png                         undistorted glass masks (read first by readVigorSceneInfo)
    <out>/colmap/<clean_images>/<split>/images/<stem>.png  undistorted clean images (read before <scene>/<clean_images>)
    <out>/colmap/source_warp.json                          fitted distortion and alignment checks, per camera
    <out>/source_warp_preview/<stem>.jpg                   COLMAP image with mask contours: resized (red), warped (green)

`--out` defaults to the scene itself. Write to another folder first to look at the result:

    python undistort_sources.py -s data/datasets/datasets/VSGD/0758 --out data/undistort_tmp/VSGD/0758
    python undistort_sources.py -s data/datasets/datasets/VSGD/0758 --preview 0          # then into the scene
"""

import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.dataset_readers import _find_clean_image
from utils.undistort_utils import MODELS, build_remap, fit_distortion, local_shifts, match_points, psnr, warp


def undistorted_camera(intr):
    if intr.model == 'PINHOLE':
        fx, fy, cx, cy = intr.params
    elif intr.model == 'SIMPLE_PINHOLE':
        fx, cx, cy = intr.params
        fy = fx
    else:
        raise ValueError(f'COLMAP camera {intr.id} is {intr.model}; only undistorted (PINHOLE / SIMPLE_PINHOLE) models are expected')
    return {'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy), 'width': int(intr.width), 'height': int(intr.height)}


def read_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def evenly_spaced(items, count):
    if len(items) <= count:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, count).round().astype(int)]


def check_alignment(scene, frames, names, colmap_dir, undistorted, maps):
    """PSNR and local shifts against the COLMAP image: plain resize of the source image vs. the fitted warp."""
    size = (undistorted['width'], undistorted['height'])
    rows = []
    for name in names:
        src = read_gray(os.path.join(scene, frames[name]['source_image']))
        col = read_gray(os.path.join(colmap_dir, 'images', name))
        resized = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
        warped = warp(src, maps)
        rows.append(
            {
                'name': name,
                'psnr_resize': psnr(resized, col),
                'psnr_warp': psnr(warped, col),
                'max_shift_resize_px': float(np.abs(local_shifts(resized, col)).max()),
                'max_shift_warp_px': float(np.abs(local_shifts(warped, col)).max()),
            }
        )
    return rows


def mask_overlay(colmap_rgb, mask_resized, mask_warped):
    out = colmap_rgb.copy()
    for mask, color in ((mask_resized, (255, 0, 0)), (mask_warped, (0, 255, 0))):
        binary = (mask.reshape(mask.shape[0], mask.shape[1], -1).max(axis=2) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def main():
    parser = ArgumentParser(description='Undistort the glass masks and clean images of a VIGOR-GS scene onto the COLMAP images')
    parser.add_argument('-s', '--source_path', required=True, help='scene folder (holds colmap/, train/, test/)')
    parser.add_argument('--out', default=None, help='output root, default the scene itself')
    parser.add_argument('--clean_images', default='clean_image', help='clean image root, relative to the scene')
    parser.add_argument('--fit_frames', type=int, default=10, help='frames per camera used to fit the distortion')
    parser.add_argument('--check_frames', type=int, default=6, help='frames per camera used to check the alignment')
    parser.add_argument('--max_residual', type=float, default=1.0, help='refuse to write when the median match residual (px) is larger')
    parser.add_argument('--model', default='auto', choices=['auto', *MODELS], help='distortion model; auto fits each one and keeps the best (radial for mild lenses, fisheye for wide-angle ones)')
    parser.add_argument('--preview', type=int, default=3, help='mask overlays to save per camera (0 for none)')
    parser.add_argument('--overwrite', action='store_true', help='replace existing output files')
    args = parser.parse_args()

    scene = os.path.abspath(args.source_path)
    out = os.path.abspath(args.out) if args.out else scene
    colmap_dir = os.path.join(scene, 'colmap')
    sparse_dir = os.path.join(colmap_dir, 'sparse', '0')
    clean_root = os.path.join(scene, args.clean_images)

    with open(os.path.join(colmap_dir, 'frames.json')) as f:
        frames = {frame['name']: frame for frame in json.load(f)['frames']}
    with open(os.path.join(colmap_dir, 'split.json')) as f:
        split = json.load(f)
    split_of = {name: s for s in ('train', 'test') for name in split.get(s, [])}
    extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, 'images.bin'))
    intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, 'cameras.bin'))

    by_camera = {}
    for extr in extrinsics.values():
        name = os.path.basename(extr.name)
        if name in split_of:
            by_camera.setdefault(extr.camera_id, []).append(name)
    if not by_camera:
        sys.exit('No image of split.json is in the COLMAP model.')

    # Plan every output first, so that nothing is written when a fit fails or a file would be overwritten.
    jobs = []  # (name, kind, source path, target path)
    for names in by_camera.values():
        for name in names:
            frame, stem = frames[name], os.path.splitext(name)[0]
            if frame.get('source_mask') and os.path.exists(os.path.join(scene, frame['source_mask'])):
                jobs.append((name, 'mask', os.path.join(scene, frame['source_mask']), os.path.join(out, 'colmap', 'masks', stem + '.png')))
            clean = _find_clean_image(clean_root, frame, split_of[name], stem) if os.path.isdir(clean_root) else None
            if clean is not None:
                rel = os.path.splitext(os.path.relpath(clean, clean_root))[0] + '.png'
                jobs.append((name, 'clean', clean, os.path.join(out, 'colmap', args.clean_images, rel)))
    existing = [target for *_, target in jobs if os.path.exists(target)]
    if existing and not args.overwrite:
        sys.exit(f'{len(existing)} output files already exist, e.g. {existing[0]}. Pass --overwrite to replace them.')

    report = {'scene': scene, 'convention': 'COLMAP pixel centers at +0.5', 'cameras': {}}
    maps_of = {}
    for camera_id, names in sorted(by_camera.items()):
        names = sorted(names)
        undistorted = undistorted_camera(intrinsics[camera_id])
        source_size = Image.open(os.path.join(scene, frames[names[0]]['source_image'])).size

        undist_pts, source_pts = [], []
        for name in evenly_spaced(names, args.fit_frames):
            pc, ps = match_points(read_gray(os.path.join(scene, frames[name]['source_image'])), read_gray(os.path.join(colmap_dir, 'images', name)))
            undist_pts.append(pc)
            source_pts.append(ps)
        undist_pts, source_pts = np.concatenate(undist_pts), np.concatenate(source_pts)
        fits = {}
        for model in MODELS if args.model == 'auto' else [args.model]:
            try:
                fits[model] = fit_distortion(undist_pts, source_pts, undistorted, source_size, model)
            except ValueError as e:
                print(f'[camera {camera_id}] {model}: {e}')
        if not fits:
            sys.exit(f'Camera {camera_id}: no distortion model fits the matches; nothing written.')
        # The simpler radial model, unless a later (more flexible) model lowers the median residual by more than 10%.
        def median_residual(m):
            return float(np.median(fits[m][1][fits[m][1] < 3.0]))

        model = next(iter(fits))
        for other in list(fits)[1:]:
            if median_residual(other) < 0.9 * median_residual(model):
                model = other
        params, dist = fits[model]
        maps = build_remap(undistorted, params, model)
        maps_of[camera_id] = maps

        inside = (maps[0] >= -0.5) & (maps[0] <= source_size[0] - 0.5) & (maps[1] >= -0.5) & (maps[1] <= source_size[1] - 0.5)
        checks = check_alignment(scene, frames, evenly_spaced(names[1:-1] or names, args.check_frames), colmap_dir, undistorted, maps)
        entry = {
            'undistorted': undistorted,
            'source_size': list(source_size),
            'model': model,
            'params': dict(zip(MODELS[model], map(float, params))),
            'other_models': {m: {'inliers_3px': int((d < 3.0).sum()), 'residual_median_px': float(np.median(d[d < 3.0]))} for m, (_, d) in fits.items() if m != model},
            'matches': int(len(dist)),
            'inliers_3px': int((dist < 3.0).sum()),
            'residual_median_px': float(np.median(dist[dist < 3.0])),
            'residual_p95_px': float(np.percentile(dist[dist < 3.0], 95)),
            'outside_source_fraction': float(1.0 - inside.mean()),
            'frames': len(names),
            'checks': checks,
        }
        report['cameras'][str(camera_id)] = entry
        p = entry['params']
        print(
            f'[camera {camera_id}] {len(names)} frames, {entry["inliers_3px"]}/{entry["matches"]} matches within 3 px, residual median {entry["residual_median_px"]:.3f} px, p95 {entry["residual_p95_px"]:.3f} px'
        )
        print(f'    model {model}: ' + ' '.join(f'{k}={v:.5g}' for k, v in p.items()) + f'; {100 * entry["outside_source_fraction"]:.2f}% of COLMAP pixels fall outside the source image')
        for m, o in entry['other_models'].items():
            print(f'    ({m}: {o["inliers_3px"]} matches within 3 px, residual median {o["residual_median_px"]:.3f} px)')
        for c in checks:
            print(f'    {c["name"]}: PSNR resize {c["psnr_resize"]:.2f} -> warp {c["psnr_warp"]:.2f} dB, max shift {c["max_shift_resize_px"]:.2f} -> {c["max_shift_warp_px"]:.2f} px')
        if entry['residual_median_px'] > args.max_residual:
            sys.exit(f'Camera {camera_id}: median residual {entry["residual_median_px"]:.3f} px exceeds --max_residual {args.max_residual}; nothing written.')

    camera_of = {os.path.basename(e.name): e.camera_id for e in extrinsics.values()}
    counts = {'mask': 0, 'clean': 0}
    for name, kind, source, target in jobs:
        image = Image.open(source)
        warped = warp(np.array(image), maps_of[camera_of[name]], nearest=kind == 'mask')
        os.makedirs(os.path.dirname(target), exist_ok=True)
        Image.fromarray(warped, mode=image.mode).save(target)
        counts[kind] += 1
    report['written'] = {'masks': counts['mask'], 'clean_images': counts['clean'], 'clean_root': os.path.join('colmap', args.clean_images)}
    os.makedirs(os.path.join(out, 'colmap'), exist_ok=True)
    with open(os.path.join(out, 'colmap', 'source_warp.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Wrote {counts["mask"]} masks and {counts["clean"]} clean images under {os.path.join(out, "colmap")}')

    if args.preview > 0:
        preview_dir = os.path.join(out, 'source_warp_preview')
        os.makedirs(preview_dir, exist_ok=True)
        for camera_id, names in by_camera.items():
            u = report['cameras'][str(camera_id)]['undistorted']
            for name in evenly_spaced(sorted(n for n in names if frames[n].get('source_mask')), args.preview):
                mask = np.array(Image.open(os.path.join(scene, frames[name]['source_mask'])))
                colmap_rgb = np.array(Image.open(os.path.join(colmap_dir, 'images', name)).convert('RGB'))
                resized = cv2.resize(mask, (u['width'], u['height']), interpolation=cv2.INTER_NEAREST)
                overlay = mask_overlay(colmap_rgb, resized, warp(mask, maps_of[camera_id], nearest=True))
                Image.fromarray(overlay).save(os.path.join(preview_dir, os.path.splitext(name)[0] + '.jpg'), quality=90)
        print(f'Previews in {preview_dir}')


if __name__ == '__main__':
    main()
