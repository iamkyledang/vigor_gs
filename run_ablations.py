"""
The VIGOR-GS ablation suite on one scene (docs/ablations.md explains what each variant tests).

    python run_ablations.py -s <scene> --root <runs folder> [--variants full,pure_fresnel,...] [shared flags ...]

Each variant is a model folder <root>/<variant> trained with train.py. A variant that changes only Phase II or the
glass selection reuses the Phase I of its `source` (copied, so the difference comes from that change alone); the
source is trained first when missing. Finished steps are skipped, so an interrupted suite is rerun with the same
command. Shared flags (e.g. -r 2 --iterations 30000) go to every variant. The table of the main metrics, with
the difference to `full`, is printed and written to <root>/summary.csv.
"""

import csv
import json
import os
import shutil
import sys
from argparse import ArgumentParser

import train

# name: (Phase I source, model flags, glass flags, Phase II flags)
VARIANTS = {
    'full': ('full', [], [], []),
    # Phase I: the representation of the interface.
    'no_separation': ('no_separation', ['--ignore_clean_images'], None, None),
    'no_factorization': ('no_factorization', ['--ablate_factorization'], [], []),
    'static_interface': ('static_interface', ['--static_interface'], [], []),
    # Glass selection (Eq. 9).
    'no_occlusion': ('full', [], ['--ablate_occlusion'], []),
    'single_patch': ('full', [], ['--max_patches', '1'], []),
    # Phase II: virtual camera and rays (Eq. 5, 9a, 9b).
    'no_virtual_camera': ('full', [], [], ['--ablate_virtual_camera']),
    'pinhole_virtual_camera': ('full', [], [], ['--pinhole_virtual_camera']),
    # Phase II: reflection weight (Eq. 10-11).
    'pure_fresnel': ('full', [], [], ['--pure_fresnel']),
    'no_fresnel': ('full', [], [], ['--ablate_fresnel']),
    # Phase II: dynamics and the motion loss (Eq. 1, 7b).
    'static_reflection': ('full', [], [], ['--static_reflection']),
    'no_motion_loss': ('full', [], [], ['--lambda_motion', '0']),
}
COLUMNS = ('PSNR', 'SSIM', 'LPIPS', 'Glass PSNR', 'Opaque PSNR', 'Transmission PSNR', 'Glass Transmission PSNR', 'FPS', 'Training Time (seconds)')


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
    for name in names:
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
    parser.add_argument('--variants', default=','.join(VARIANTS), help=f'comma-separated, from {", ".join(VARIANTS)}')
    args, shared = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    names = [n for n in args.variants.split(',') if n]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        parser.error(f'unknown variants {unknown}')
    for name in names:
        run_variant(name, os.path.abspath(args.source_path), os.path.abspath(args.root), shared)
    summarize(os.path.abspath(args.root), names)


if __name__ == '__main__':
    main()
