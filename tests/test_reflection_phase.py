"""Phase II: neighbors for L_motion, a short run on the synthetic scene, frozen Phase I, resume and the ablations."""

import json
import os
import shutil
from types import SimpleNamespace

import pytest
import torch

from tests.conftest import SMALL
from train_reflection import previous_frames

SHORT = ['--iterations', '200', '--warm_up', '50', '--densify_from_iter', '50', '--densification_interval', '50', '--test_interval', '200',
         '--report_interval', '50', '--checkpoint_interval', '100', '--presence_reset_interval', '0', '--reflection_init_points', '3000',
         '--max_tile_entries', '20000000', '--quiet']


def test_previous_frames_stay_within_a_sequence_and_camera():
    def cam(uid, time, sequence='a', sensor=1):
        return SimpleNamespace(uid=uid, time=time, sequence=sequence, sensor_id=sensor, image_width=4, image_height=3)

    cameras = [cam(1, 0.2), cam(2, 0.0), cam(3, 0.5), cam(4, 0.1, 'b'), cam(5, 0.3, 'b'), cam(6, 0.25, 'a', 2)]
    previous = previous_frames(cameras)
    assert previous[2] is None and previous[1].uid == 2 and previous[3].uid == 1
    assert previous[4] is None and previous[5].uid == 4
    assert previous[6] is None  # another sensor


def copy_phase_one(trained_interface, synthetic_scene, tmp_path, ablate_occlusion=False):
    """A model folder holding the shared Phase I and the true glass patch."""
    import extract_glass

    model = tmp_path / 'model'
    model.mkdir()
    shutil.copytree(os.path.join(trained_interface, 'interface'), model / 'interface')
    shutil.copy(os.path.join(trained_interface, 'config.json'), model / 'config.json')
    known = tmp_path / 'known.json'
    known.write_text(json.dumps({'patches': synthetic_scene.patches}))
    extract_glass.main(['-m', str(model), '--glass_geometry', str(known)] + (['--ablate_occlusion'] if ablate_occlusion else []))
    return str(model)


@pytest.mark.gpu
@pytest.mark.slow
def test_short_phase_two_explains_the_reflection_and_keeps_phase_one(trained_interface, synthetic_scene, tmp_path):
    import train_reflection

    model = copy_phase_one(trained_interface, synthetic_scene, tmp_path)
    before = torch.load(os.path.join(model, 'interface', 'latest.pth'), weights_only=False)['field']['params']
    train_reflection.main(['-m', model] + SHORT)
    after = torch.load(os.path.join(model, 'interface', 'latest.pth'), weights_only=False)['field']['params']
    assert all(torch.equal(before[k], after[k]) for k in before)
    log = [json.loads(line) for line in open(os.path.join(model, 'reflection', 'training.jsonl'))]
    final = log[-1]
    assert final['iteration'] == 200 and 'motion' in log[0] and 'optical' in log[0]
    assert final['test_glass_psnr'] > final['test_glass_psnr_direct']  # the reflection explains part of the glass
    assert final['test_psnr'] > final['test_psnr_direct']  # off the glass alpha = 0, so the gain comes from the glass
    config = json.load(open(os.path.join(model, 'reflection', 'config.json')))
    assert config['pure_fresnel'] is False and config['lambda_motion'] == 0.1
    state = torch.load(os.path.join(model, 'reflection', 'latest.pth'), weights_only=False)
    assert 'optical' in state['modules'] and state['optimizers']['blend'] is not None
    train_reflection.main(['-m', model, '--resume'] + SHORT[:1] + ['230'] + SHORT[2:])
    log = [json.loads(line) for line in open(os.path.join(model, 'reflection', 'training.jsonl'))]
    assert log[-1]['iteration'] == 230


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('flags', [['--pure_fresnel'], ['--ablate_fresnel'], ['--ablate_virtual_camera'], ['--pinhole_virtual_camera'],
                                   ['--static_reflection', '--lambda_motion', '0'], ['--ablate_occlusion']])
def test_each_phase_two_ablation_trains(trained_interface, synthetic_scene, tmp_path, flags):
    import train_reflection

    occlusion = flags == ['--ablate_occlusion']
    model = copy_phase_one(trained_interface, synthetic_scene, tmp_path, ablate_occlusion=occlusion)
    run = ['-m', model, '--iterations', '30', '--warm_up', '10', '--test_interval', '30', '--checkpoint_interval', '30',
           '--reflection_init_points', '2000', '--densify_until_iter', '0', '--quiet'] + ([] if occlusion else flags)
    train_reflection.main(run)
    log = [json.loads(line) for line in open(os.path.join(model, 'reflection', 'training.jsonl'))]
    assert log[-1]['iteration'] == 30 and log[-1]['test_psnr'] > 10
    state = torch.load(os.path.join(model, 'reflection', 'latest.pth'), weights_only=False)
    assert ('optical' in state['modules']) == ('--pure_fresnel' not in flags)
    assert ('constant' in state['modules']) == ('--ablate_fresnel' in flags)
    if occlusion:
        assert json.load(open(os.path.join(model, 'glass_geometry.json')))['settings']['ablate_occlusion'] is True
