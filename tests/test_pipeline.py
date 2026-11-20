"""train.py (the whole chain) and run_ablations.py on the synthetic scene."""

import json
import os

import pytest

from tests.conftest import SMALL

TINY = ['--iterations', '60', '--warm_up', '20', '--surface_from_iter', '20', '--densify_from_iter', '20', '--densification_interval', '20',
        '--test_interval', '60', '--checkpoint_interval', '60', '--presence_reset_interval', '0', '--reflection_init_points', '2000',
        '--skip_lpips'] + SMALL


def known_geometry(tmp_path, synthetic_scene):
    path = tmp_path / 'known.json'
    path.write_text(json.dumps({'patches': synthetic_scene.patches}))
    return str(path)


def test_unknown_flags_are_refused(tmp_path):
    import train

    with pytest.raises(SystemExit, match='--lamda_motion'):
        train.main(['-s', str(tmp_path), '-m', str(tmp_path / 'm'), '--lamda_motion', '0.1'])
    assert train.without_source(['-s', 'a', '-r', '2', '--source_path=b', '--x']) == ['-r', '2', '--x']


@pytest.mark.gpu
@pytest.mark.slow
def test_train_runs_the_chain_and_skips_finished_steps(synthetic_scene, tmp_path):
    import train

    model = str(tmp_path / 'model')
    argv = ['-s', synthetic_scene.path, '-m', model, '--glass_geometry', known_geometry(tmp_path, synthetic_scene),
            '--reflection_args', '--lambda_motion 0.05'] + TINY
    train.main(argv)
    for path in ('config.json', 'interface/latest.pth', 'glass_geometry.json', 'reflection/latest.pth', 'results.json', 'test/ours_60/renders'):
        assert os.path.exists(os.path.join(model, path)), path
    assert json.load(open(os.path.join(model, 'reflection', 'config.json')))['lambda_motion'] == 0.05
    assert json.load(open(os.path.join(model, 'interface', 'config.json')))['iterations'] == 60
    sessions = len(open(os.path.join(model, 'interface', 'training.jsonl')).readlines())
    train.main(argv + ['--skip_render', '--skip_metrics'])  # nothing left to train
    assert len(open(os.path.join(model, 'interface', 'training.jsonl')).readlines()) == sessions


@pytest.mark.gpu
@pytest.mark.slow
def test_ablation_suite_reuses_phase_one_and_summarizes(synthetic_scene, tmp_path):
    import run_ablations

    root = tmp_path / 'runs'
    run_ablations.main(['-s', synthetic_scene.path, '--root', str(root), '--variants', 'pure_fresnel,no_separation,no_occlusion',
                        '--glass_geometry', known_geometry(tmp_path, synthetic_scene)] + TINY)
    assert sorted(os.listdir(root)) == ['full', 'no_occlusion', 'no_separation', 'pure_fresnel', 'summary.csv']
    import torch

    full = torch.load(root / 'full' / 'interface' / 'latest.pth', weights_only=False)['field']['params']['xyz']
    copied = torch.load(root / 'pure_fresnel' / 'interface' / 'latest.pth', weights_only=False)['field']['params']['xyz']
    assert torch.equal(full, copied)  # the Phase II variant reuses full's Phase I
    assert not os.path.exists(root / 'full' / 'reflection')  # full itself was only trained as a source here
    assert not os.path.exists(root / 'no_separation' / 'glass_geometry.json')
    assert os.path.isdir(root / 'no_separation' / 'test' / 'direct_60')
    assert json.load(open(root / 'no_occlusion' / 'glass_geometry.json'))['settings']['ablate_occlusion'] is True
    lines = open(root / 'summary.csv').read().splitlines()
    assert lines[0].startswith('variant,PSNR') and len(lines) == 4
