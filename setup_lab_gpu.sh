#!/usr/bin/env bash
# VIGOR-GS on the lab workstation: Ubuntu 24.04, NVIDIA RTX PRO 6000 Blackwell (SM 12.0),
# driver 595 / CUDA 13.2, Python 3.11, torch 2.9.1 cu128, conda toolkit 12.9.
# Replaces setup.sh + colab_t4.py's Colab path for a plain SSH machine (no sudo needed).
# setup_colab_t4.sh pins torch 2.1.2/CUDA 12.1/sm_75 (T4-only); Blackwell (sm_120) needs
# newer wheels, so this script builds its own conda env instead of reusing that one.
#
# Trains all three scenes (VSGD 0758, WayveScene101 scene_001, 3D-FRONT-T scene_1) back to
# back; each resumes independently from its own saved checkpoint.
#
#   bash setup_lab_gpu.sh                # rclone + prepare + install
#   bash setup_lab_gpu.sh rclone         # only install rclone (then run `rclone config`)
#   bash setup_lab_gpu.sh prepare|install|check
#   bash setup_lab_gpu.sh resume-status  # pull + report saved checkpoints, change nothing
#   bash setup_lab_gpu.sh pull           # Drive -> local results (existing checkpoints)
#   bash setup_lab_gpu.sh push           # local results -> Drive, then verify
#   bash setup_lab_gpu.sh run [--scenes vsgd_0758,3dfrontt_scene_1] [train.py args]
#   source setup_lab_gpu.sh --activate-only
#
# Run long jobs detached:  nohup bash setup_lab_gpu.sh run > ~/vigor_gs_run.out 2>&1 &
#
# Why results survive: training writes to a local volume (VIGOR_RESULTS_LOCAL) and is
# mirrored to Google Drive (VIGOR_RCLONE_RESULTS) every VIGOR_SYNC_SECONDS and on exit.
# Colab mounted Drive directly; rclone copy replaces that because this host has no FUSE mount.
#
# Checkpoint "signature": colab_t4.py's check_model() rejects a saved model whenever its
# config.json source_path is not exactly equal (Path.resolve()) to the current scene path.
# Checkpoints trained on Colab store /content/Lighting_GS/datasets/<scene>; this host's
# absolute path is different (and could change again if VIGOR_WORK moves), so that exact
# check would always reject a perfectly good checkpoint here. _vigor_checkpoint_status below
# compares scenes by their last path components instead of the full absolute path, so a
# Colab checkpoint -- or one made under an earlier VIGOR_WORK layout on this host -- still
# resumes correctly.

# ---------------------------------------------------------------- configuration
_vigor_self="${BASH_SOURCE[0]}"
export VIGOR_WORK="${VIGOR_WORK:-/user-data}"
export VIGOR_WORKSPACE="${VIGOR_WORKSPACE:-$VIGOR_WORK/Lighting_GS}"
export VIGOR_REPO="${VIGOR_REPO:-$VIGOR_WORKSPACE/vigor_gs_quan_dang}"
export VIGOR_CONDA_ROOT="${VIGOR_CONDA_ROOT:-$VIGOR_WORK/miniforge3}"
export VIGOR_ENV_PREFIX="$VIGOR_REPO/.conda"
# vigor_gs_data sits next to the repo (same layout as the local vigor_gs_quan_dang / vigor_gs_data checkout):
#   $VIGOR_DATA_ROOT/VSGD/0758/colmap, $VIGOR_DATA_ROOT/WayveScene101/scene_001/colmap, ...
export VIGOR_DATA_ROOT="${VIGOR_DATA_ROOT:-$VIGOR_WORKSPACE/vigor_gs_data}"
export VIGOR_RESULTS_LOCAL="${VIGOR_RESULTS_LOCAL:-$VIGOR_WORK/vigor_gs_results/output}"
export VIGOR_LOGS_LOCAL="${VIGOR_LOGS_LOCAL:-$VIGOR_WORK/vigor_gs_results/logs}"
export VIGOR_RCLONE_REMOTE="${VIGOR_RCLONE_REMOTE:-gdrive}"
export VIGOR_RCLONE_ROOT="${VIGOR_RCLONE_ROOT:-$VIGOR_RCLONE_REMOTE:DACN_HK261/results}"
export VIGOR_RCLONE_RESULTS="${VIGOR_RCLONE_RESULTS:-$VIGOR_RCLONE_ROOT/vigor_gs/output}"
export VIGOR_RCLONE_LOGS="${VIGOR_RCLONE_LOGS:-$VIGOR_RCLONE_ROOT/vigor_gs/logs}"
if [[ -z "${VIGOR_CODE_ARCHIVE:-}" ]]; then   # wherever scp put it; falls back to Drive in _vigor_prepare
    VIGOR_CODE_ARCHIVE="$VIGOR_WORK/vigor_gs.tar.gz"
    for _candidate in /home/vigor_gs.tar.gz "$HOME/vigor_gs.tar.gz" "$VIGOR_WORK/vigor_gs.tar.gz"; do
        [[ -f "$_candidate" ]] && { VIGOR_CODE_ARCHIVE="$_candidate"; break; }
    done
fi
if [[ -z "${VIGOR_DATA_ARCHIVE:-}" ]]; then
    VIGOR_DATA_ARCHIVE="$VIGOR_WORK/vigor_gs_datasets.tar.gz"
    for _candidate in /home/vigor_gs_datasets.tar.gz "$HOME/vigor_gs_datasets.tar.gz" "$VIGOR_WORK/vigor_gs_datasets.tar.gz"; do
        [[ -f "$_candidate" ]] && { VIGOR_DATA_ARCHIVE="$_candidate"; break; }
    done
fi
VIGOR_SYNC_SECONDS="${VIGOR_SYNC_SECONDS:-600}"
VIGOR_MIN_FREE_GB="${VIGOR_MIN_FREE_GB:-40}"
export PATH="$HOME/.local/bin:$PATH"

# name|<scene source, relative to VIGOR_DATA_ROOT>|<model dir, relative to VIGOR_REPO>|resolution|extra train.py args
# 3D-FRONT-T runs at -r 1 (no downsample; the documented default is -r 2) -- its model dir is "_r1"
# so this doesn't collide with the already-completed -r 2 run in "output/3dfrontt_scene_1_r2".
# vsgd_0758_r1_fresh is a from-scratch Phase I retrain (the original Colab-derived vsgd_0758_r1, kept
# untouched locally, only covered ~30% of the window; the retrain reaches 0.54, above the default 0.5).
# --max_tile_entries 2e9 effectively removes the memory-safety cap on (Gaussian, tile) pairs of the reflected-ray
# renderer (default 64M, hit by all three scenes in Phase II); it does not change any result, and
# render.py reads it back from reflection/config.json so the final render uses it too.
VIGOR_SCENES=(
    "vsgd_0758_fresh|VSGD/0758|output/vsgd_0758_r1_fresh|1|--max_tile_entries 2000000000"
    "wayvescene101_scene_001|WayveScene101/scene_001|output/wayvescene101_scene_001_r2|2|--max_tile_entries 2000000000"
    "3dfrontt_scene_1|3D-FRONT-T/scene_1|output/3dfrontt_scene_1_r1|1|--max_tile_entries 2000000000"
)

_vigor_die() { echo "ERROR: $*" >&2; return 1; }

_vigor_activate() {
    [[ -x "$VIGOR_ENV_PREFIX/bin/python" && -x "$VIGOR_ENV_PREFIX/bin/nvcc" ]] || {
        echo 'Missing VIGOR-GS environment/toolkit; run setup first.' >&2; return 1;
    }
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]] || cap="12.0"
    export PATH="$VIGOR_ENV_PREFIX/bin:$PATH"
    export CUDA_HOME="$VIGOR_ENV_PREFIX"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    export CC="$VIGOR_ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$VIGOR_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$CXX"
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LD_LIBRARY_PATH="$VIGOR_ENV_PREFIX/lib:$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CMAKE_PREFIX_PATH="$VIGOR_ENV_PREFIX/lib/python3.11/site-packages/torch/share/cmake:$VIGOR_ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export TORCH_CUDA_ARCH_LIST="$cap" CUDAARCHS="${cap/./}"
    export TORCH_EXTENSIONS_DIR="$VIGOR_REPO/.cache/torch_extensions_lab"
    export TORCH_HOME="$VIGOR_REPO/.cache/torch"          # VGG weights for LPIPS persist here
    export MAX_JOBS="${MAX_JOBS:-8}" CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS:-8}"
    export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
}

# ---------------------------------------------------------------- preflight
_vigor_preflight() {
    local tool
    for tool in curl tar gzip sha256sum nvidia-smi; do
        command -v "$tool" >/dev/null || { _vigor_die "Missing command: $tool"; return 1; }
    done
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]] || cap="12.0"
    if [[ "${cap%%.*}" -lt 12 ]]; then
        echo "WARNING: GPU compute capability $cap is not Blackwell (12.x); the cu128 build still works." >&2
    fi
    mkdir -p "$VIGOR_WORK" 2>/dev/null || { _vigor_die "Cannot create $VIGOR_WORK"; return 1; }
    local free_gb
    free_gb="$(df -BG --output=avail "$VIGOR_WORK" | tail -1 | tr -dc '0-9')"
    echo "Free on $VIGOR_WORK: ${free_gb} GiB"
    if [[ "$free_gb" -lt "$VIGOR_MIN_FREE_GB" ]]; then
        _vigor_die "Need >= ${VIGOR_MIN_FREE_GB} GiB free on $VIGOR_WORK (datasets ~2G packed, ~15G env/CUDA toolkit)."
        return 1
    fi
}

# ---------------------------------------------------------------- rclone
_vigor_install_rclone() (
    set -euo pipefail
    if command -v rclone >/dev/null; then rclone version | head -1; return 0; fi
    local version zip tmp
    version="$(curl -fsSL https://downloads.rclone.org/version.txt | awk '{print $2}')"
    [[ "$version" =~ ^v[0-9.]+$ ]]
    zip="rclone-$version-linux-amd64.zip"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    curl -fL --retry 3 -o "$tmp/$zip" "https://downloads.rclone.org/$version/$zip"
    curl -fsSL -o "$tmp/SHA256SUMS" "https://downloads.rclone.org/$version/SHA256SUMS"
    (cd "$tmp" && grep " $zip\$" SHA256SUMS | sha256sum -c -)
    mkdir -p "$HOME/.local/bin"
    if command -v unzip >/dev/null; then
        unzip -q -j -o "$tmp/$zip" "rclone-$version-linux-amd64/rclone" -d "$HOME/.local/bin"
    else
        python3 - "$tmp/$zip" "rclone-$version-linux-amd64/rclone" "$HOME/.local/bin/rclone" <<'PY'
import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z, z.open(sys.argv[2]) as s, open(sys.argv[3], "wb") as d:
    shutil.copyfileobj(s, d)
PY
    fi
    chmod +x "$HOME/.local/bin/rclone"
    "$HOME/.local/bin/rclone" version | head -1
)

_vigor_need_remote() {
    command -v rclone >/dev/null || { _vigor_die "rclone missing; run: bash $_vigor_self rclone"; return 1; }
    rclone listremotes | grep -qx "$VIGOR_RCLONE_REMOTE:" || {
        cat >&2 <<EOF
ERROR: rclone remote '$VIGOR_RCLONE_REMOTE' is not configured.
  On this headless server:  rclone config   (n -> name '$VIGOR_RCLONE_REMOTE' -> drive -> scope 1 -> answer 'n' to
  auto config -> on your own PC with rclone run:  rclone authorize "drive"  -> paste the token back here)
  Then verify:  rclone lsd $VIGOR_RCLONE_ROOT
EOF
        return 1
    }
    rclone lsd "$VIGOR_RCLONE_ROOT" >/dev/null 2>&1 || rclone mkdir "$VIGOR_RCLONE_ROOT"
}

# ---------------------------------------------------------------- archives
_vigor_extract() {
    local archive="$1" marker="$2" description="$3" destination="$4" stamp
    [[ -f "$archive" ]] || { _vigor_die "Missing $description archive: $archive"; return 1; }
    stamp="$(stat -c '%s-%Y' "$archive")"
    if [[ -f "$marker" && "$(cat "$marker")" == "$stamp" ]]; then
        echo "$description already extracted."; return 0
    fi
    echo "Testing $archive ..."
    gzip -t "$archive"
    mkdir -p "$destination"
    echo "Extracting $description into $destination ..."
    tar --no-same-owner -xzf "$archive" -C "$destination"
    echo "$stamp" > "$marker"
}

_vigor_link_output() {
    mkdir -p "$VIGOR_RESULTS_LOCAL" || return 1
    local output="$VIGOR_REPO/output"
    if [[ -L "$output" ]]; then
        [[ "$(readlink -f "$output")" == "$(readlink -f "$VIGOR_RESULTS_LOCAL")" ]] || {
            _vigor_die "Unexpected output symlink: $output -> $(readlink -f "$output")"; return 1;
        }
    elif [[ -e "$output" ]]; then
        _vigor_die "Preserve or move the VM-local output directory before proceeding: $output"; return 1
    else
        ln -s "$VIGOR_RESULTS_LOCAL" "$output" || return 1
    fi
    echo "Output: $output -> $VIGOR_RESULTS_LOCAL"
}

_vigor_prepare() {
    if [[ ! -f "$VIGOR_CODE_ARCHIVE" ]]; then
        _vigor_need_remote || return 1
        echo "Downloading vigor_gs.tar.gz from Drive ..."
        rclone copyto "$VIGOR_RCLONE_ROOT/vigor_gs.tar.gz" "$VIGOR_CODE_ARCHIVE" \
            --progress --transfers 1 --checksum || return 1
    fi
    _vigor_extract "$VIGOR_CODE_ARCHIVE" "$VIGOR_WORKSPACE/.vigor_code_extracted" "VIGOR-GS code" "$VIGOR_WORKSPACE" || return 1
    if [[ ! -f "$VIGOR_DATA_ARCHIVE" ]]; then
        _vigor_need_remote || return 1
        echo "Downloading vigor_gs_datasets.tar.gz from Drive ..."
        rclone copyto "$VIGOR_RCLONE_ROOT/datasets/vigor_gs_datasets.tar.gz" "$VIGOR_DATA_ARCHIVE" \
            --progress --transfers 1 --checksum || return 1
    fi
    _vigor_extract "$VIGOR_DATA_ARCHIVE" "$VIGOR_WORKSPACE/.vigor_data_extracted" "VIGOR-GS datasets" "$VIGOR_DATA_ROOT" || return 1
    [[ -f "$VIGOR_REPO/train.py" && -f "$VIGOR_REPO/submodules/simple-knn/setup.py" \
       && -f "$VIGOR_REPO/submodules/diff-surfel-anych/setup.py" ]] \
        || { _vigor_die "vigor_gs layout incomplete under $VIGOR_REPO"; return 1; }
    local entry name source_rel model_rel resolution scene path
    for entry in "${VIGOR_SCENES[@]}"; do
        IFS='|' read -r name source_rel model_rel resolution _ <<< "$entry"
        scene="$VIGOR_DATA_ROOT/$source_rel"
        # split.json/source_warp.json are optional in scene/dataset_readers.py (e.g.
        # 3D-FRONT-T has no source_warp.json); only the pieces the reader requires are checked.
        for path in images sparse/0 frames.json masks clean_image; do
            [[ -e "$scene/colmap/$path" ]] || { _vigor_die "$name: missing scene input $scene/colmap/$path"; return 1; }
        done
    done
    _vigor_link_output || return 1
    mkdir -p "$VIGOR_LOGS_LOCAL"
    echo "PREPARE SUCCESSFUL"
}

# ---------------------------------------------------------------- install
_vigor_install_miniforge() {
    [[ -x "$VIGOR_CONDA_ROOT/bin/conda" ]] && return 0
    if [[ -e "$VIGOR_CONDA_ROOT" ]]; then
        _vigor_die "Incomplete Miniforge at $VIGOR_CONDA_ROOT; choose a new VIGOR_CONDA_ROOT."; return 1
    fi
    local dir="$VIGOR_REPO/.cache/install" tag base
    mkdir -p "$dir"
    tag="$(curl -fsSIL -o /dev/null -w '%{url_effective}' \
        https://github.com/conda-forge/miniforge/releases/latest | awk -F/ '{print $NF}')"
    [[ "$tag" =~ ^[0-9][0-9A-Za-z.-]*$ ]] || { _vigor_die "Cannot resolve Miniforge release"; return 1; }
    base="Miniforge3-$tag-Linux-x86_64.sh"
    curl -fL --retry 3 -o "$dir/$base" "https://github.com/conda-forge/miniforge/releases/download/$tag/$base"
    curl -fsSL --retry 3 -o "$dir/$base.sha256" "https://github.com/conda-forge/miniforge/releases/download/$tag/$base.sha256"
    (cd "$dir" && sha256sum -c "$base.sha256")
    bash "$dir/$base" -b -p "$VIGOR_CONDA_ROOT"
}

_vigor_install() (
    set -euo pipefail
    cd "$VIGOR_REPO"
    [[ -f submodules/simple-knn/setup.py && -f submodules/diff-surfel-anych/setup.py ]]
    mkdir -p "$VIGOR_REPO/.cache/install" "$VIGOR_REPO/.cache/tmp"
    export TMPDIR="$VIGOR_REPO/.cache/tmp"
    export PIP_CACHE_DIR="$VIGOR_REPO/.cache/pip"
    _vigor_install_miniforge
    local conda_command="$VIGOR_CONDA_ROOT/bin/conda"
    if [[ ! -f "$VIGOR_ENV_PREFIX/conda-meta/history" ]]; then
        if [[ -e "$VIGOR_ENV_PREFIX" ]]; then
            echo "Incomplete environment at $VIGOR_ENV_PREFIX; preserve/rename it before retrying." >&2
            exit 1
        fi
        "$conda_command" create -y --override-channels -c conda-forge \
            -p "$VIGOR_ENV_PREFIX" python=3.11 pip 'setuptools<82' wheel \
            ninja make 'gcc_linux-64=11' 'gxx_linux-64=11'
    fi
    # Toolkit 12.9 is the first conda series with solid sm_120 (Blackwell) support.
    "$conda_command" install -y --override-channels \
        -c nvidia/label/cuda-12.9.1 -c conda-forge \
        -p "$VIGOR_ENV_PREFIX" 'cuda-toolkit=12.9' ninja make 'gcc_linux-64=11' 'gxx_linux-64=11'
    _vigor_activate
    # cu128 wheels carry sm_120 kernels; the cu121 wheels setup_colab_t4.sh installs do not run on Blackwell.
    .conda/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu128
    .conda/bin/python -m pip install -r requirements.txt
    # opencv-python needs libGL.so.1, often absent in containers without root.
    if ! .conda/bin/python -c 'import cv2' 2>/dev/null; then
        echo 'cv2 import failed; switching to opencv-python-headless'
        cv_version="$(.conda/bin/python -m pip show opencv-python | awk '/^Version:/{print $2}')"
        .conda/bin/python -m pip uninstall -y opencv-python opencv-python-headless
        .conda/bin/python -m pip install --no-deps --force-reinstall "opencv-python-headless==${cv_version}"
        .conda/bin/python -c 'import cv2; print("cv2", cv2.__version__)'
    fi
    local extension build stamp
    stamp="$(date -u +%Y%m%d_%H%M%S_%N)"
    for extension in simple-knn diff-surfel-anych; do
        build="$VIGOR_REPO/submodules/$extension/build"
        if [[ -e "$build" ]]; then
            mv -- "$build" "${build}.before-lab-${stamp}"
        fi
    done
    .conda/bin/python -m pip install -v --no-cache-dir --no-build-isolation --no-deps \
        --force-reinstall "$VIGOR_REPO/submodules/simple-knn"
    # VIGOR-GS must load its own renderer, including reflected-ray CUDA kernels.
    (cd submodules/diff-surfel-anych && "$VIGOR_ENV_PREFIX/bin/python" setup.py build_ext --inplace --force)
    .conda/bin/python -m pip check
    _vigor_verify
    .conda/bin/python - <<'PY'
import torch
from lpipsPyTorch import LPIPS
with torch.no_grad():
    metric = LPIPS(net_type='vgg').cuda().eval()
    image = torch.zeros(1, 3, 64, 64, device='cuda')
    assert torch.isfinite(metric(image, image)).all()
print('LPIPS VGG ready.', flush=True)
PY
    .conda/bin/python -m pip freeze > "$VIGOR_REPO/.cache/lab_gpu_packages.txt"
    mkdir -p "$VIGOR_LOGS_LOCAL"
    cp "$VIGOR_REPO/.cache/lab_gpu_packages.txt" "$VIGOR_LOGS_LOCAL/packages_lab_$(date -u +%Y%m%d_%H%M%S).txt"
    echo 'SETUP SUCCESSFUL'
)

_vigor_verify() (
    set -euo pipefail
    cd "$VIGOR_REPO"
    _vigor_activate
    local first_source first_scene
    IFS='|' read -r _ first_source _ _ _ <<< "${VIGOR_SCENES[0]}"
    first_scene="$VIGOR_DATA_ROOT/$first_source"
    VIGOR_VERIFY_SCENE="$first_scene" .conda/bin/python - <<'PY'
from pathlib import Path
import os
import torch
import torchvision
assert torch.__version__.split('+')[0] == '2.9.1', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.24.1'
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA unavailable'
capability = torch.cuda.get_device_capability()
arch = 'sm_%d%d' % capability
assert arch in torch.cuda.get_arch_list(), (arch, torch.cuda.get_arch_list())
x = torch.randn(64, 64, device='cuda', requires_grad=True)
x.square().mean().backward()
torch.cuda.synchronize()
import gaussian_renderer  # Puts this checkout's in-place renderer first on sys.path.
import diff_surfel_anych
from diff_surfel_anych import GaussianRasterizer, render_reflected_rays
assert Path(diff_surfel_anych.__file__).resolve().is_relative_to(Path('submodules/diff-surfel-anych').resolve())
from simple_knn._C import distCUDA2
assert torch.isfinite(distCUDA2(torch.rand(32, 3, device='cuda'))).all()
import train, render, metrics
from scene.dataset_readers import readVigorSceneInfo
scene = readVigorSceneInfo(os.environ['VIGOR_VERIFY_SCENE'], init_points=32)
assert len(scene.train_cameras) > 0 and len(scene.test_cameras) > 0
print('GPU:', torch.cuda.get_device_name(), arch, 'torch:', torch.__version__)
print('VIGOR-GS and', os.environ['VIGOR_VERIFY_SCENE'], 'inputs ready.', flush=True)
PY
)

# ---------------------------------------------------------------- checkpoint signature fix
# VIGOR_CKPT_MODEL_PATH / VIGOR_CKPT_SCENE / VIGOR_CKPT_RESOLUTION are set by the caller
# (one scene at a time) so this same check runs for all three scenes.
_vigor_checkpoint_status() (
    set -euo pipefail
    cd "$VIGOR_REPO"
    _vigor_activate
    .conda/bin/python - <<'PY'
import json
import os
import shutil
from pathlib import Path

from utils.checkpoint_utils import load_checkpoint

MODEL = Path(os.environ['VIGOR_CKPT_MODEL_PATH'])
SCENE = Path(os.environ['VIGOR_CKPT_SCENE']).resolve()
RESOLUTION = int(os.environ['VIGOR_CKPT_RESOLUTION'])


def scene_identity(path_str):
    """Compare scenes by <dataset>/<scene id> only, not the absolute prefix: the same scene is
    unpacked under 'datasets/...' on Colab but 'vigor_gs_data/...' on this lab host -- even the
    wrapping dataset-root folder name legitimately differs, so only the last two components
    (e.g. 'VSGD/0758') identify the scene; a checkpoint trained on one must still resume on
    the other."""
    parts = Path(path_str).parts
    return parts[-2:] if len(parts) >= 2 else tuple(parts)


checkpoints = {
    phase: [p for p in (MODEL / phase).glob('iteration_*.pth')
            if p.is_file() and p.stem.removeprefix('iteration_').isdigit()]
    for phase in ('interface', 'reflection')
}
for phase in checkpoints:
    latest = MODEL / phase / 'latest.pth'
    if latest.is_file():
        checkpoints[phase].append(latest)
if checkpoints['reflection'] and not checkpoints['interface']:
    raise RuntimeError('Phase II checkpoint exists without its required Phase I checkpoint.')
config = MODEL / 'config.json'
if any(checkpoints.values()) and not config.is_file():
    raise FileNotFoundError(f'Saved checkpoints require the original model config: {config}')
if config.is_file():
    saved = json.loads(config.read_text())
    if saved.get('resolution') != RESOLUTION or scene_identity(saved.get('source_path', '')) != scene_identity(str(SCENE)):
        raise RuntimeError(f'Existing model uses different inputs or resolution: {config}')

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
PY
)

_vigor_scene_checkpoint_status() {
    local source_rel="$1" model_rel="$2" resolution="$3"
    VIGOR_CKPT_MODEL_PATH="$VIGOR_REPO/$model_rel" \
    VIGOR_CKPT_SCENE="$VIGOR_DATA_ROOT/$source_rel" \
    VIGOR_CKPT_RESOLUTION="$resolution" \
        _vigor_checkpoint_status
}

# ---------------------------------------------------------------- Drive sync
_vigor_filters=(--exclude '*.tmp')

_vigor_pull() {
    _vigor_need_remote || return 1
    mkdir -p "$VIGOR_RESULTS_LOCAL" || return 1
    echo "Drive -> local: $VIGOR_RCLONE_RESULTS -> $VIGOR_RESULTS_LOCAL"
    if rclone lsf "$VIGOR_RCLONE_RESULTS" >/dev/null 2>&1; then
        rclone copy "$VIGOR_RCLONE_RESULTS" "$VIGOR_RESULTS_LOCAL" --update --transfers 4 \
            --drive-chunk-size 64M "${_vigor_filters[@]}" --stats 30s --stats-one-line || return 1
    else
        echo "No existing results on Drive yet."
    fi
}

_vigor_push() {
    _vigor_need_remote || return 1
    mkdir -p "$VIGOR_RESULTS_LOCAL" "$VIGOR_LOGS_LOCAL" || return 1
    rclone copy "$VIGOR_RESULTS_LOCAL" "$VIGOR_RCLONE_RESULTS" --update --transfers 4 \
        --drive-chunk-size 64M --retries 5 "${_vigor_filters[@]}" --stats 60s --stats-one-line || return 1
    rclone copy "$VIGOR_LOGS_LOCAL" "$VIGOR_RCLONE_LOGS" --update --transfers 2 --retries 5
}

_vigor_verify_drive() {
    echo "Verifying Drive copy (size comparison) ..."
    rclone check "$VIGOR_RESULTS_LOCAL" "$VIGOR_RCLONE_RESULTS" --one-way --size-only \
        "${_vigor_filters[@]}" && echo "DRIVE COPY VERIFIED"
}

_vigor_resume_status_all() {
    _vigor_activate || return 1
    _vigor_pull || return 1
    _vigor_link_output || return 1
    local entry name source_rel model_rel resolution rc=0
    for entry in "${VIGOR_SCENES[@]}"; do
        IFS='|' read -r name source_rel model_rel resolution _ <<< "$entry"
        echo "=== $name ==="
        _vigor_scene_checkpoint_status "$source_rel" "$model_rel" "$resolution" || rc=1
    done
    return $rc
}

_vigor_run() (
    set -uo pipefail
    cd "$VIGOR_REPO"
    local only_scenes=""
    if [[ "${1:-}" == --scenes ]]; then only_scenes=",$2,"; shift 2; fi
    _vigor_need_remote || exit 1
    _vigor_activate
    mkdir -p "$VIGOR_LOGS_LOCAL"
    _vigor_pull || exit 1
    _vigor_link_output || exit 1
    (while sleep "$VIGOR_SYNC_SECONDS"; do _vigor_push >/dev/null 2>&1 || echo "sync failed; will retry"; done) &
    local sync_pid=$!
    trap 'kill "$sync_pid" 2>/dev/null; wait "$sync_pid" 2>/dev/null; _vigor_push; _vigor_verify_drive' EXIT

    local entry name source_rel model_rel resolution extra_args scene log failures=0
    for entry in "${VIGOR_SCENES[@]}"; do
        IFS='|' read -r name source_rel model_rel resolution extra_args <<< "$entry"
        if [[ -n "$only_scenes" && "$only_scenes" != *",$name,"* ]]; then
            echo "Skipping $name (not in --scenes list)"; continue
        fi
        scene="$VIGOR_DATA_ROOT/$source_rel"
        log="$VIGOR_LOGS_LOCAL/train_lab_${name}_$(date -u +%Y%m%d_%H%M%S).log"
        echo "=== $name: $scene -r $resolution -> $model_rel ==="
        echo "Log: $log"
        if ! _vigor_scene_checkpoint_status "$source_rel" "$model_rel" "$resolution"; then
            echo "$name: checkpoint check failed, skipping" >&2
            failures=$((failures + 1))
            continue
        fi
        # train.py chains Phase I, glass extraction, Phase II, test renders and metrics itself,
        # and resumes any phase whose latest.pth has not yet reached its --iterations.
        .conda/bin/python -u train.py -s "$scene" -m "$model_rel" -r "$resolution" \
            --checkpoint_interval 1000 $extra_args "$@" 2>&1 | tee "$log"
        [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "$name: train.py failed" >&2; failures=$((failures + 1)); }
    done

    # train.py renders the test views and writes results.json only after BOTH phases reach their
    # iterations, so a scene that crashed earlier has no render/metrics yet: report it instead of
    # letting it look finished (a re-run resumes from the checkpoints and then renders/evaluates).
    local incomplete=0 m p1 p2 rd mt
    echo "=== summary: render = test/ views, metrics = results.json ==="
    for entry in "${VIGOR_SCENES[@]}"; do
        IFS='|' read -r name source_rel model_rel resolution extra_args <<< "$entry"
        if [[ -n "$only_scenes" && "$only_scenes" != *",$name,"* ]]; then continue; fi
        m="$VIGOR_REPO/$model_rel"
        p1=-; p2=-; rd=-; mt=-
        [[ -f "$m/interface/latest.pth" ]] && p1=ckpt
        [[ -f "$m/reflection/latest.pth" ]] && p2=ckpt
        [[ -d "$m/test" ]] && rd=done
        [[ -f "$m/results.json" ]] && mt=done
        printf '%-26s PhaseI=%-5s PhaseII=%-5s render=%-5s metrics=%s\n' "$name" "$p1" "$p2" "$rd" "$mt"
        [[ "$mt" == done ]] || incomplete=$((incomplete + 1))
    done
    [[ "$failures" -eq 0 && "$incomplete" -eq 0 ]]
    exit $?
)

# ---------------------------------------------------------------- dispatch
# Subshell steps run as plain statements: errexit is disabled inside any `||`/`&&` context.
_vigor_main() {
    local step="${1:-all}" rc
    case "$step" in
        --activate-only) _vigor_activate ;;
        --check|check) _vigor_verify ;;
        resume-status) _vigor_resume_status_all ;;
        rclone) _vigor_install_rclone ;;
        prepare) _vigor_preflight || return $?; _vigor_prepare ;;
        install) _vigor_preflight || return $?; _vigor_install ;;
        pull) _vigor_pull ;;
        push) _vigor_push || return $?; _vigor_verify_drive ;;
        run) shift; _vigor_run "$@" ;;
        all)
            _vigor_preflight || return $?
            _vigor_install_rclone; rc=$?; [[ $rc -eq 0 ]] || return $rc
            _vigor_prepare || return $?
            _vigor_install ;;
        *) echo "Usage: bash setup_lab_gpu.sh [all|rclone|prepare|install|check|resume-status|pull|push|run [--scenes a,b] [args]]" >&2
           return 2 ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    _vigor_main "$@"
    exit $?
elif [[ $# -gt 0 ]]; then
    _vigor_main "$@"
fi
