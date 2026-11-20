"""Inference outputs, the comparison video and the metrics, including timing across resumed sessions."""

import json
import os

import numpy as np
import pytest
from PIL import Image

from metrics import performance_summary, phase_training_seconds


def write_log(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def test_training_time_sums_the_last_elapsed_time_of_each_session(tmp_path):
    log = tmp_path / 'interface' / 'training.jsonl'
    write_log(str(log), [{'iteration': 100, 'session_id': 'a', 'elapsed_seconds': 10.0}, {'iteration': 200, 'session_id': 'a', 'elapsed_seconds': 20.0},
                         {'iteration': 300, 'session_id': 'b', 'elapsed_seconds': 25.0}])
    assert phase_training_seconds(log) == pytest.approx(45.0)
    assert phase_training_seconds(log, iteration=200) == pytest.approx(20.0)
    assert phase_training_seconds(tmp_path / 'missing.jsonl') is None
    write_log(str(tmp_path / 'reflection' / 'training.jsonl'), [{'iteration': 50, 'session_id': 'c', 'elapsed_seconds': 3600.0 + 120}])
    (tmp_path / 'render_stats.json').write_text(json.dumps({'test': {'ours_50': {'frames': 4, 'fps': 12.5}}}))
    summary = performance_summary(tmp_path, 'ours_50', 'test', 4)
    assert summary['FPS'] == 12.5 and summary['Training Time'] == '1 hours and 2 mins'
    assert summary['Training Time by Phase (seconds)'] == {'interface': 45.0, 'reflection': 3720.0}
    assert performance_summary(tmp_path, 'ours_50', 'test', 5)['FPS'] is None  # another frame count: not this render
    assert performance_summary(tmp_path, 'direct_300', 'test', 4)['Training Time (seconds)'] == pytest.approx(45.0)


def test_comparison_video_is_ours_left_gt_right_in_the_given_order(tmp_path):
    import cv2

    from render import create_comparison_video

    for sub in ('renders', 'gt'):
        (tmp_path / sub).mkdir()
    for name, size, ours, gt in [('later', (25, 33), (0, 255, 0), (255, 255, 0)), ('earlier', (21, 29), (255, 0, 0), (0, 0, 255))]:
        Image.fromarray(np.full((*size, 3), ours, np.uint8)).save(tmp_path / 'renders' / f'{name}.png')
        Image.fromarray(np.full((*size, 3), gt, np.uint8)).save(tmp_path / 'gt' / f'{name}.png')
    capture = cv2.VideoCapture(str(create_comparison_video(tmp_path, ['earlier', 'later'], fps=12.0)))
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0, abs=0.01)
        for left, right in [((0, 0, 255), (255, 0, 0)), ((0, 255, 0), (0, 255, 255))]:  # BGR
            ok, frame = capture.read()
            assert ok and frame.shape == (66, 66, 3)
            np.testing.assert_allclose(frame[52, 16], left, atol=15)
            np.testing.assert_allclose(frame[52, 49], right, atol=15)
        assert not capture.read()[0]
    finally:
        capture.release()
    with pytest.raises(ValueError):
        create_comparison_video(tmp_path, ['earlier'], fps=0.0)
    with pytest.raises(ValueError):
        create_comparison_video(tmp_path, [])


@pytest.mark.gpu
@pytest.mark.slow
def test_render_and_metrics_of_a_trained_model(trained_interface, synthetic_scene, tmp_path):
    import metrics
    import render
    import train_reflection
    from tests.test_reflection_phase import copy_phase_one

    model = copy_phase_one(trained_interface, synthetic_scene, tmp_path)
    train_reflection.main(['-m', model, '--iterations', '60', '--warm_up', '20', '--test_interval', '60', '--checkpoint_interval', '60',
                           '--reflection_init_points', '2000', '--densify_until_iter', '0', '--quiet'])
    render.main(['-m', model, '--components'])
    render.main(['-m', model, '--direct_only', '--skip_train', '--no_video'])
    folder = os.path.join(model, 'test', 'ours_60')
    for sub in ('renders', 'gt', 'glass_masks', 'valid_masks', 'direct', 'reflection', 'alpha', 'patches', 'clean'):
        assert len(os.listdir(os.path.join(folder, sub))) == 3, sub
    assert os.path.exists(os.path.join(folder, 'comparison.mp4'))
    poses = json.load(open(os.path.join(folder, 'virtual_cameras.json')))
    assert len(poses) == 3 and all('0' in p for p in poses.values())
    # Off the glass the composite is the direct image: alpha is zero there.
    alpha = np.asarray(Image.open(os.path.join(folder, 'alpha', sorted(os.listdir(os.path.join(folder, 'alpha')))[0])))
    patches = np.asarray(Image.open(os.path.join(folder, 'patches', sorted(os.listdir(os.path.join(folder, 'patches')))[0])))
    assert np.all(alpha[patches.sum(-1) == 0] == 0) and alpha[patches.sum(-1) > 0].max() > 0

    results = metrics.evaluate([model], skip_lpips=True)
    ours, direct = results['ours_60'], results['direct_300']
    for key in ('PSNR', 'SSIM', 'Valid PSNR', 'Glass PSNR', 'Opaque PSNR', 'Masked PSNR (RT)', 'Transmission PSNR', 'Glass Transmission PSNR', 'FPS'):
        assert ours[key] is not None, key
    assert ours['Glass PSNR'] > direct['Glass PSNR']
    assert ours['Training Time by Phase (seconds)']['reflection'] > 0
    assert os.path.exists(os.path.join(model, 'per_view.json'))
