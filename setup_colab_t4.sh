#!/usr/bin/env bash
# Colab-only environment, adapted from RT-Splatting_v0/setup_colab_t4.sh.
# bash setup_colab_t4.sh; source setup_colab_t4.sh --activate-only

VIGOR_REPO="${VIGOR_REPO:-/content/Lighting_GS/vigor_gs}"
VIGOR_CONDA_ROOT="${VIGOR_CONDA_ROOT:-/content/miniforge3}"
VIGOR_ENV_PREFIX="$VIGOR_CONDA_ROOT/envs/vigor_gs"

_vigor_prepend() {
  local name="$1" directory="$2" current
  [[ -d "$directory" ]] || return 0
  current="${!name:-}"
  case ":$current:" in
    *":$directory:"*) ;;
    *) printf -v "$name" '%s' "$directory${current:+:$current}"; export "$name" ;;
  esac
}

_vigor_build_env() {
  export VIGOR_REPO VIGOR_CONDA_ROOT VIGOR_ENV_PREFIX
  export CUDA_HOME="$VIGOR_ENV_PREFIX" CUDACXX="$VIGOR_ENV_PREFIX/bin/nvcc"
  export CC="$VIGOR_ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$VIGOR_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX" CUDAARCHS=75 TORCH_CUDA_ARCH_LIST=7.5
  export MAX_JOBS="${MAX_JOBS:-2}" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
  export TORCH_EXTENSIONS_DIR="$VIGOR_REPO/.cache/torch_extensions_t4_gcc11"
  _vigor_prepend PATH "$VIGOR_ENV_PREFIX/bin"
  _vigor_prepend LD_LIBRARY_PATH "$VIGOR_ENV_PREFIX/lib/python3.10/site-packages/torch/lib"
  _vigor_prepend LD_LIBRARY_PATH "$VIGOR_ENV_PREFIX/lib"
  _vigor_prepend LD_LIBRARY_PATH "$CUDA_HOME/targets/x86_64-linux/lib"
  _vigor_prepend CPATH "$CUDA_HOME/targets/x86_64-linux/include"
  _vigor_prepend LIBRARY_PATH "$CUDA_HOME/targets/x86_64-linux/lib"
  _vigor_prepend LIBRARY_PATH "$CUDA_HOME/targets/x86_64-linux/lib/stubs"
  _vigor_prepend LIBRARY_PATH "$CUDA_HOME/lib/stubs"
}

_vigor_verify() (
  set -euo pipefail
  _vigor_build_env
  cd "$VIGOR_REPO"
  python - <<'PY'
from pathlib import Path
import sys
import torch
import torchvision
import numpy
import cv2, scipy, PIL, plyfile, tqdm, matplotlib
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split('+')[0] == '2.1.2', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.16.2'
assert torch.version.cuda == '12.1', torch.version.cuda
assert numpy.__version__ == '1.26.4'
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.get_device_capability(0) == (7, 5)
assert 'T4' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
import gaussian_renderer  # Puts this checkout's in-place renderer first on sys.path.
import diff_surfel_anych
from diff_surfel_anych import GaussianRasterizer, render_reflected_rays
assert Path(diff_surfel_anych.__file__).resolve().is_relative_to(Path('submodules/diff-surfel-anych').resolve())
from simple_knn._C import distCUDA2
assert torch.isfinite(distCUDA2(torch.rand(32, 3, device='cuda'))).all()
import train, render, metrics
from scene.dataset_readers import readVigorSceneInfo
scene = readVigorSceneInfo('/content/Lighting_GS/datasets/VSGD/0758', init_points=32)
assert len(scene.train_cameras) > 0 and len(scene.test_cameras) > 0
print('VIGOR-GS CUDA and VSGD 0758 inputs ready:', torch.cuda.get_device_name(0), flush=True)
PY
)

_vigor_install() (
  set -euo pipefail
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]
  cd "$VIGOR_REPO"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q T4 || {
    echo 'This installer requires a Colab NVIDIA T4 session.' >&2; exit 1;
  }
  mkdir -p "$VIGOR_REPO/.tmp" "$VIGOR_REPO/.cache"
  export TMPDIR="$VIGOR_REPO/.tmp"
  if [[ ! -x "$VIGOR_CONDA_ROOT/bin/conda" ]]; then
    [[ ! -e "$VIGOR_CONDA_ROOT" ]] || {
      echo "Incomplete Miniforge: $VIGOR_CONDA_ROOT; preserve/rename it before retrying." >&2; exit 1;
    }
    curl -fsSL --retry 3 -o "$TMPDIR/miniforge.sh" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$TMPDIR/miniforge.sh" -b -p "$VIGOR_CONDA_ROOT"
  fi
  export CONDA_PKGS_DIRS="$VIGOR_CONDA_ROOT/pkgs"
  local conda_bin="$VIGOR_CONDA_ROOT/bin/conda"
  if [[ ! -f "$VIGOR_ENV_PREFIX/conda-meta/history" ]]; then
    [[ ! -e "$VIGOR_ENV_PREFIX" ]] || {
      echo "Incomplete env: $VIGOR_ENV_PREFIX; preserve/rename it before retrying." >&2; exit 1;
    }
    "$conda_bin" create -y --override-channels -c conda-forge \
      -p "$VIGOR_ENV_PREFIX" python=3.10 pip
  fi
  "$conda_bin" install -y --override-channels -c conda-forge \
    -p "$VIGOR_ENV_PREFIX" python=3.10 pip 'gcc_linux-64=11' 'gxx_linux-64=11' ninja
  if ! "$VIGOR_ENV_PREFIX/bin/nvcc" --version 2>/dev/null | grep -q 'release 12\.1'; then
    "$conda_bin" install -y --override-channels \
      -c nvidia/label/cuda-12.1.0 -c nvidia -c conda-forge \
      -p "$VIGOR_ENV_PREFIX" cuda-toolkit=12.1
  fi
  _vigor_build_env
  python -m pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
  python -m pip install --no-cache-dir 'numpy==1.26.4' 'setuptools==69.5.1' wheel \
    -r "$VIGOR_REPO/requirements.txt"
  local extension build stamp
  stamp="$(date -u +%Y%m%d_%H%M%S_%N)"
  for extension in simple-knn diff-surfel-anych; do
    build="$VIGOR_REPO/submodules/$extension/build"
    if [[ -e "$build" ]]; then
      mv -- "$build" "${build}.before-t4-${stamp}"
    fi
  done
  python -m pip install -v --no-cache-dir --no-build-isolation --no-deps \
    --force-reinstall "$VIGOR_REPO/submodules/simple-knn"
  # VIGOR-GS must load its own renderer, including reflected-ray CUDA kernels.
  (cd submodules/diff-surfel-anych && python setup.py build_ext --inplace --force)
  python -m pip check
  _vigor_verify
  # Fetch metric weights before the long training job starts.
  python - <<'PY'
from lpipsPyTorch import LPIPS
import torch
with torch.no_grad():
    metric = LPIPS(net_type='vgg').cuda().eval()
    image = torch.zeros(1, 3, 64, 64, device='cuda')
    assert torch.isfinite(metric(image, image)).all()
print('LPIPS VGG ready.', flush=True)
PY
  python -m pip freeze > "$VIGOR_REPO/.cache/colab_t4_packages.txt"
)

_vigor_main() {
  case "${1:-}" in
    --activate-only)
      [[ -x "$VIGOR_ENV_PREFIX/bin/python" && -x "$VIGOR_ENV_PREFIX/bin/nvcc" ]] || {
        echo 'Run setup_colab_t4.sh first.' >&2; return 1;
      }
      _vigor_build_env
      cd "$VIGOR_REPO"
      ;;
    '') _vigor_install ;;
    *) echo 'Usage: bash setup_colab_t4.sh; source setup_colab_t4.sh --activate-only' >&2; return 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  _vigor_main "$@"
else
  set -euo pipefail
  _vigor_main "$@"
fi
