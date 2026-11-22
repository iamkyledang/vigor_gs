"""Colab T4 workflow for VSGD 0758, invoked by setup.sh through colab exec."""

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import tarfile


DRIVE = Path('/content/drive/MyDrive')
DRIVE_ROOT = DRIVE / 'DACN_HK261/results'
WORKSPACE = Path('/content/Lighting_GS')
REPO = WORKSPACE / 'vigor_gs'
SCENE = WORKSPACE / 'datasets/VSGD/0758'
RESULTS = DRIVE_ROOT / 'vigor_gs'
MODEL = REPO / 'output/vsgd_0758_r1'


def require_drive():
    if not DRIVE.is_dir() or not os.path.ismount(DRIVE.parent):
        raise RuntimeError('Mount Google Drive first with colab drivemount.')


def check_layout():
    for name in ('train.py', 'render.py', 'metrics.py', 'setup_colab_t4.sh'):
        if not (REPO / name).is_file():
            raise FileNotFoundError(REPO / name)
    for name in ('images', 'sparse/0', 'frames.json', 'split.json', 'masks', 'clean_image', 'source_warp.json'):
        if not (SCENE / 'colmap' / name).exists():
            raise FileNotFoundError(SCENE / 'colmap' / name)


def link_output():
    require_drive()
    target = RESULTS / 'output'
    target.mkdir(parents=True, exist_ok=True)
    output = REPO / 'output'
    if output.is_symlink():
        if output.resolve() != target.resolve():
            raise RuntimeError(f'Unexpected output symlink: {output} -> {output.resolve()}')
    elif output.exists():
        raise RuntimeError(f'Preserve or move the VM-local output directory before proceeding: {output}')
    else:
        output.symlink_to(target, target_is_directory=True)
    print(f'Output: {output} -> {target}', flush=True)


def check_model():
    """Validate persisted inputs before installing the environment or changing training state."""
    checkpoints = {
        phase: [p for p in (MODEL / phase).glob('iteration_*.pth')
                if p.is_file() and p.stem.removeprefix('iteration_').isdigit()]
        for phase in ('interface', 'reflection')
    }
    for phase in checkpoints:
        latest = MODEL / phase / 'latest.pth'
        if latest.is_file():
            checkpoints[phase].append(latest)
    if os.environ.get('VIGOR_REQUIRE_CHECKPOINT') == '1' and not checkpoints['interface']:
        raise RuntimeError(f'Resume requested, but no Phase I checkpoint exists in {MODEL / "interface"}.')
    if checkpoints['reflection'] and not checkpoints['interface']:
        raise RuntimeError('Phase II checkpoint exists without its required Phase I checkpoint.')
    config = MODEL / 'config.json'
    if any(checkpoints.values()) and not config.is_file():
        raise FileNotFoundError(f'Saved checkpoints require the original model config: {config}')
    if config.is_file():
        saved = json.loads(config.read_text())
        if saved.get('resolution') != 1 or Path(saved.get('source_path', '')).resolve() != SCENE.resolve():
            raise RuntimeError(f'Existing model uses different inputs or resolution: {config}')
    return checkpoints


def checkpoint_status():
    """Read saved iterations using the installed torch environment; restore missing latest files."""
    from utils.checkpoint_utils import load_checkpoint

    require_drive()
    link_output()
    checkpoints = check_model()
    for phase, label in (('interface', 'Phase I'), ('reflection', 'Phase II')):
        latest = MODEL / phase / 'latest.pth'
        if not checkpoints[phase]:
            print(f'{label}: no checkpoint yet; this phase will start after its prerequisites.', flush=True)
            continue
        source = latest if latest.is_file() else max(
            checkpoints[phase], key=lambda p: int(p.stem.removeprefix('iteration_')))
        # A corrupt existing latest file fails explicitly; it is never replaced silently.
        state = load_checkpoint(str(source))
        required = {'iteration', 'field', 'deformation', 'deformation_optimizer', 'modules', 'optimizers', 'rng'}
        if not required.issubset(state) or not isinstance(state['iteration'], int) or state['iteration'] < 0:
            raise ValueError(f'Incomplete resume state in {source}')
        if source != latest:
            temporary = latest.with_name('latest.pth.resume.tmp')
            shutil.copyfile(source, temporary)
            os.replace(temporary, latest)
            print(f'{label}: restored latest.pth from {source.name}', flush=True)
        iteration = state['iteration']
        action = f'resume at iteration {iteration + 1}' if iteration < 30000 else 'training already complete; skip'
        print(f'{label}: saved iteration {iteration}; {action}. Checkpoint: {latest}', flush=True)
        del state


def prepare():
    require_drive()
    archives = (DRIVE_ROOT / 'vigor_gs.tar.gz', DRIVE_ROOT / 'datasets/vsgd_0758.tar.gz')
    required = 0
    for path in archives:
        with tarfile.open(path, 'r:gz') as archive:
            required += sum(member.size for member in archive if member.isfile())
    if shutil.disk_usage('/content').free <= required + 25 * 1024**3:
        raise RuntimeError('Insufficient VM disk space, including 25 GiB for the CUDA environment.')
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for path in archives:
        print(f'Extracting {path.name}', flush=True)
        subprocess.run(['tar', '--no-same-owner', '-xzf', str(path), '-C', str(WORKSPACE)], check=True)
    check_layout()
    link_output()
    check_model()


def stream_command(command, label):
    require_drive()
    check_layout()
    link_output()
    logs = RESULTS / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    log = logs / f'{label}_0758_r1_T4_{stamp}.log'
    env = dict(os.environ, VIGOR_REPO=str(REPO), VIGOR_CONDA_ROOT='/content/miniforge3', PYTHONUNBUFFERED='1')
    print(f'Log: {log}', flush=True)
    with log.open('w', buffering=1) as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f'{label} failed (exit {code}). Log: {log}')
    print(f'{label.upper()} SUCCESSFUL', flush=True)


def setup():
    stream_command(['bash', str(REPO / 'setup_colab_t4.sh')], 'setup')


def run():
    check_model()
    stream_command(['bash', '-c', r'''
set -euo pipefail
source "$VIGOR_REPO/setup_colab_t4.sh" --activate-only
VIGOR_COLAB_STEP=checkpoints python -u colab_t4.py
python -u train.py -s /content/Lighting_GS/datasets/VSGD/0758 \
    -m output/vsgd_0758_r1 -r 1 --checkpoint_interval 1000 --skip_render --skip_metrics
python -u render.py -m output/vsgd_0758_r1 -r 1 --skip_train --components
python -u metrics.py -m output/vsgd_0758_r1 --split test
'''], 'train_render_metrics')


if __name__ == '__main__':
    actions = {'prepare': prepare, 'setup': setup, 'run': run, 'checkpoints': checkpoint_status}
    action = os.environ.get('VIGOR_COLAB_STEP', '')
    if action not in actions:
        raise ValueError('Set VIGOR_COLAB_STEP=prepare|setup|run|checkpoints.')
    actions[action]()
