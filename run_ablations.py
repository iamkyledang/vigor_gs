"""
The VIGOR-GS ablation suite on one scene (docs/ablations.md explains what each variant tests).

    python run_ablations.py -s <scene> --root <runs folder> [--variants full,pure_fresnel,...] [shared flags ...]

Without --variants, the main suite runs: the five ablations of MAIN_SUITE (`full` is not rerun; see below). The other
variants in VARIANTS run only when named. Variants that copy Phase I from `full` use <root>/full/interface: put an
existing full run at <root>/full, otherwise only its Phase I is trained there. If <root>/full/results.json exists it
is the reference row of summary.csv, whether or not `full` is among the variants.

Resolution. Unless -r / --resolution is passed, it is chosen from the scene path: -r 1 for VSGD and 3D-FRONT-T,
-r 2 for WayveScene101 (SCENE_RESOLUTIONS). The same value goes to every variant of the scene, so the Phase I
copied from `full` always matches it.

Tile budget. Unless --max_tile_entries is passed, every variant gets --max_tile_entries 2000000000 (MAX_TILE_ENTRIES):
the default cap of the reflected-ray renderer (64M (Gaussian, tile) pairs) is hit in Phase II on all three scenes, and
raising it changes no result. render.py reads it back from reflection/config.json.

Each variant is a model folder <root>/<variant> trained with train.py. A variant that changes only Phase II or the
glass selection reuses the Phase I of its `source` (copied, so the difference comes from that change alone); the
source is trained first when missing. Finished steps are skipped, so an interrupted suite is rerun with the same
command. Shared flags (e.g. -r 2 --iterations 30000) go to every variant. The table of the main metrics, with
the difference to `full`, is printed and written to <root>/summary.csv.
"""

import csv
import json
import os
import re
import shutil
import sys
from argparse import ArgumentParser

import train

# name: (Phase I source, model flags, glass flags, Phase II flags)
VARIANTS = {
    'full': ('full', [], [], []),
    # Main suite: one ablation per claim of the framework.
    'no_separation': ('no_separation', ['--ignore_clean_images'], None, None),  # two fields (Sec. 1, Eq. 2-8)
    'no_factorization': ('no_factorization', ['--ablate_factorization'], [], []),  # p * o interface (Sec. 1, Eq. 2)
    'no_virtual_camera': ('full', [], [], ['--ablate_virtual_camera']),  # reflected rays / V_dyn (Eq. 5, 9a, 9b)
    'pure_fresnel': ('full', [], [], ['--pure_fresnel']),  # the h_psi correction (Eq. 11)
    'static_reflection': ('full', [], [], ['--static_reflection']),  # D_R for dynamic scenes (Eq. 1)
    # Additional variants, run only when named in --variants.
    'static_interface': ('static_interface', ['--static_interface'], [], []),
    'no_occlusion': ('full', [], ['--ablate_occlusion'], []),
    'single_patch': ('full', [], ['--max_patches', '1'], []),
    'pinhole_virtual_camera': ('full', [], [], ['--pinhole_virtual_camera']),
    'no_fresnel': ('full', [], [], ['--ablate_fresnel']),
    'no_motion_loss': ('full', [], [], ['--lambda_motion', '0']),
}
MAIN_SUITE = ('no_separation', 'no_factorization', 'no_virtual_camera', 'pure_fresnel', 'static_reflection')
MAX_TILE_ENTRIES = 2000000000  # effectively no cap on (Gaussian, tile) pairs of Phase II and rendering
# Default downsampling factor per dataset, matched against the lower-cased scene path (first match wins).
SCENE_RESOLUTIONS = (('vsgd', 1), ('3d-front-t', 1), ('wayve', 2))
COLUMNS = ('PSNR', 'SSIM', 'LPIPS', 'Glass PSNR', 'Opaque PSNR', 'Transmission PSNR', 'Glass Transmission PSNR', 'FPS', 'Training Time (seconds)')


def default_resolution(scene, shared):
    """['-r', N] for the dataset of `scene` unless the command line already sets the resolution; [] if unknown."""
    if any(a in ('-r', '--resolution') or a.startswith('--resolution=') or re.fullmatch(r'-r\d+', a) for a in shared):
        return []
    path = scene.lower()
    for key, factor in SCENE_RESOLUTIONS:
        if key in path:
            return ['-r', str(factor)]
    return []


def default_tile_budget(shared):
    """['--max_tile_entries', N] unless the command line already sets it."""
    if any(a == '--max_tile_entries' or a.startswith('--max_tile_entries=') for a in shared):
        return []
    return ['--max_tile_entries', str(MAX_TILE_ENTRIES)]


def copy_phase_one(root, source, target):
    os.makedirs(target, exist_ok=True)
    if not os.path.exists(os.path.join(target, 'interface', 'latest.pth')):
        shutil.copytree(os.path.join(root, source, 'interface'), os.path.join(target, 'interface'), dirs_exist_ok=True)
        shutil.copy(os.path.join(root, source, 'config.json'), os.path.join(target, 'config.json'))


def run_variant(name, scene, root, shared):
    source, model_flags, glass_flags, reflection_flags = VARIANTS[name]
    target = os.path.join(root, name)
    if source != name:
        if not os.path.exists(os.path.join(root, source, 'interface', 'latest.pth')):
            print(f'===== {source}: Phase I only, as the source of {name}', flush=True)
            train.main(['-s', scene, '-m', os.path.join(root, source), '--only_interface'] + shared + VARIANTS[source][1])
        copy_phase_one(root, source, target)
    argv = ['-s', scene, '-m', target] + shared + model_flags
    if glass_flags:  # '=' keeps argparse from reading a value that starts with '--' as a flag
        argv.append('--glass_args=' + ' '.join(glass_flags))
    if reflection_flags:
        argv.append('--reflection_args=' + ' '.join(reflection_flags))
    print(f'===== {name}: {" ".join(argv)}', flush=True)
    train.main(argv)


def summarize(root, names):
    rows = {}
    for name in dict.fromkeys(['full', *names]):  # `full` is the reference even when it was not rerun
        path = os.path.join(root, name, 'results.json')
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)
            method = max(results, key=lambda m: int(m.rsplit('_', 1)[-1]))
            rows[name] = {column: results[method].get(column) for column in COLUMNS}
    reference = rows.get('full', {})
    with open(os.path.join(root, 'summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', *COLUMNS, 'PSNR - full', 'Glass PSNR - full'])
        for name, row in rows.items():
            delta = [None if row.get(k) is None or reference.get(k) is None else row[k] - reference[k] for k in ('PSNR', 'Glass PSNR')]
            writer.writerow([name, *[row[c] for c in COLUMNS], *delta])
            print(f'{name:24s} ' + ' '.join(f'{c}={row[c]:.3f}' if isinstance(row[c], float) else f'{c}={row[c]}' for c in COLUMNS[:7]))
    return rows


def main(argv=None):
    parser = ArgumentParser(description=__doc__.split('\n\n')[0], add_help=True)
    parser.add_argument('-s', '--source_path', required=True)
    parser.add_argument('--root', required=True, help='folder of the variant model folders')
    parser.add_argument('--variants', default=','.join(MAIN_SUITE),
                        help=f'comma-separated, from {", ".join(VARIANTS)} (default: the main suite {", ".join(MAIN_SUITE)})')
    args, shared = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    names = [n for n in args.variants.split(',') if n]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        parser.error(f'unknown variants {unknown}')
    scene = os.path.abspath(args.source_path)
    shared = default_resolution(scene, shared) + default_tile_budget(shared) + shared
    print(f'shared flags: {" ".join(shared)}', flush=True)
    for name in names:
        run_variant(name, scene, os.path.abspath(args.root), shared)
    summarize(os.path.abspath(args.root), names)


if __name__ == '__main__':
    main()
