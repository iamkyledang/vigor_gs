"""
VIGOR-GS: fix the glass patches P = {(n_k, d_k, S_k)} between the two phases.

Known geometry (--glass_geometry PATH, JSON {"patches": [{"normal": [nx, ny, nz], "offset": d, "vertices": [[x, y, z],
...]}]}, n^T x + d = 0, vertices an ordered convex polygon on the plane) is validated and used as is. Otherwise the
patches are extracted from the frozen Phase I geometry (utils/patch_extraction.py) and checked against the training
masks: below --plane_min_coverage / --plane_min_precision the geometry is written with "accepted": false and
Phase II refuses it.

    python extract_glass.py -m <model> [--glass_geometry patches.json] [--max_patches 4] [--ablate_occlusion]

Writes <model>/glass_geometry.json: patches, the first-hit settings of Eq. (9) (occlusion test), quality, source.
Also <model>/glass_preview/<view>.png: annotated glass (green), patch hits (red), both (yellow), occluded hits (blue).
"""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from arguments import GlassParams, ModelParams, PipelineParams, load_model_args
from gaussian_renderer.interface import render_interface
from scene import Scene
from utils.glass_geometry import first_hit, load_glass_geometry, patches_to_records, validate_patches
from utils.model_io import GLASS_GEOMETRY, direct_pass_for, load_interface, occlusion_settings
from utils.patch_extraction import extract_patches, patch_quality, sample_views


def settings_of(glass, extent):
    """The first-hit rule stored with the patches. glass_margin: disks this close to a patch are the glass layer."""
    return {'occlusion_tolerance': glass.occlusion_tolerance, 'occlusion_alpha': glass.occlusion_alpha, 'ablate_occlusion': glass.ablate_occlusion,
            'glass_margin': glass.plane_threshold * extent}


@torch.no_grad()
def write_previews(folder, views, patches, settings, direct_pass, epsilon):
    os.makedirs(folder, exist_ok=True)
    for cam in views:
        geometry = first_hit(cam, patches, epsilon, occlusion_settings(settings, direct_pass(cam)))
        base = cam.original_srgb.permute(1, 2, 0).cpu().numpy() * 0.5
        hits, mask, occluded = geometry['visible'].cpu().numpy(), cam.glass_mask[0].cpu().numpy(), geometry['occluded'].cpu().numpy()
        overlay = base.copy()
        overlay[mask & ~hits] = [0, 1, 0]
        overlay[hits & ~mask] = [1, 0, 0]
        overlay[hits & mask] = [1, 1, 0]
        overlay[occluded] = [0, 0.4, 1]
        image = 0.6 * overlay + 0.4 * base
        Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(os.path.join(folder, cam.image_name + '.png'))


def extract(model_args, pipe, glass):
    scene = Scene(model_args, load_clean=False)
    field, deformation, iteration = load_interface(model_args, scene)
    settings = settings_of(glass, scene.extent)
    cameras = scene.getTrainCameras()
    quality = {}
    if glass.glass_geometry:
        patches, _ = load_glass_geometry(glass.glass_geometry)
        records = patches_to_records(patches)
        source = os.path.abspath(glass.glass_geometry)
    else:
        records, quality['samples'] = extract_patches(cameras, field, deformation, pipe, glass, render_interface, scene.extent, model_args.seed)
        patches = validate_patches(records)
        source = f'phase_i_iteration_{iteration}'
    direct_pass = direct_pass_for(field, deformation, pipe, patches, settings)
    quality.update(patch_quality(sample_views(cameras), patches, pipe.ray_epsilon, lambda cam: occlusion_settings(settings, direct_pass(cam))))
    # Known geometry is trusted (its quality is reported only); extracted geometry must pass the checks.
    accepted = bool(glass.glass_geometry) or (quality['coverage'] >= glass.plane_min_coverage and quality['precision'] >= glass.plane_min_precision)
    record = {'patches': records, 'settings': settings, 'quality': quality, 'source': source, 'accepted': bool(accepted)}
    with open(os.path.join(model_args.model_path, GLASS_GEOMETRY), 'w') as f:
        json.dump(record, f, indent=2)
    write_previews(os.path.join(model_args.model_path, 'glass_preview'), sample_views(cameras, 6), patches, settings, direct_pass, pipe.ray_epsilon)
    return record


def parse(argv, known=False):
    """Parsed settings of this step; with `known`, arguments of other steps are left in args.unknown (train.py)."""
    parser = ArgumentParser(description='VIGOR-GS: fixed glass patches from known geometry or the Phase I field')
    model, pipeline, glass = ModelParams(parser, sentinel=True), PipelineParams(parser), GlassParams(parser)
    args, unknown = parser.parse_known_args(argv) if known else (parser.parse_args(argv), [])
    if not args.model_path:
        parser.error('-m is required')
    model_args = model.extract(args)
    model_args.model_path = os.path.abspath(model_args.model_path)
    glass_args = glass.extract(args)
    glass_args.unknown = unknown
    return load_model_args(model_args), pipeline.extract(args), glass_args


def main(argv=None, known=False):
    model_args, pipe, glass = parse(sys.argv[1:] if argv is None else argv, known)
    if model_args.ignore_clean_images:
        raise SystemExit('This model is the no-separation ablation (--ignore_clean_images): it has no Phase II and needs no glass patches.')
    record = extract(model_args, pipe, glass)
    print(f'[ INFO ] {len(record["patches"])} glass patches, quality {record["quality"]}, accepted: {record["accepted"]}')
    if not record['accepted']:
        raise SystemExit(f'The extracted glass geometry fails the coverage/precision checks (see {os.path.join(model_args.model_path, GLASS_GEOMETRY)} '
                         'and glass_preview/). Improve Phase I or supply known geometry with --glass_geometry; Phase II will refuse it.')


if __name__ == '__main__':
    main()
