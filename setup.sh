#!/usr/bin/env bash
# Run locally: package VSGD 0758 -> upload -> fresh Colab T4 -> resume/train/render/metrics.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd)"
UPLOAD_DIR="$WORKSPACE/colab_upload/vigor_gs_0758"
SCENE="$WORKSPACE/vigor_gs_data/redistort_data-20261120T145509Z-1-003/redistort_data/VSGD/0758"
DRIVE_ROOT="gdrive:DACN_HK261/results"
SESSION="${VIGOR_SESSION:-vigor-gs-0758-$(date -u +%Y%m%d-%H%M%S)-$RANDOM}"
REQUIRE_CHECKPOINT=0

case "${1:-}" in
  ''|--package-only) ;;
  --resume) REQUIRE_CHECKPOINT=1 ;;
  *) echo 'Usage: bash vigor_gs/setup.sh [--resume|--package-only]' >&2; exit 2 ;;
esac
for tool in tar gzip python3; do
  command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 1; }
done
if [[ "${1:-}" != --package-only ]]; then
  for tool in colab rclone; do
    command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 1; }
  done
fi
for path in train.py render.py metrics.py colab_t4.py setup_colab_t4.sh \
  submodules/simple-knn/setup.py submodules/diff-surfel-anych/setup.py \
  submodules/diff-surfel-anych/third_party/glm/glm/glm.hpp; do
  [[ -f "$SCRIPT_DIR/$path" ]] || { echo "Missing source: $path" >&2; exit 1; }
done
for path in images sparse/0 frames.json split.json masks clean_image source_warp.json; do
  [[ -e "$SCENE/colmap/$path" ]] || { echo "Missing scene input: $SCENE/colmap/$path" >&2; exit 1; }
done

if (( REQUIRE_CHECKPOINT )); then
  printf '\n==> VIGOR-GS: locate saved checkpoints on Drive\n'
  checkpoint_list="$(rclone lsf "$DRIVE_ROOT/vigor_gs/output/vsgd_0758_r1" --recursive --files-only)"
  if ! grep -Eq '^interface/(latest|iteration_[0-9]+)\.pth$' <<< "$checkpoint_list"; then
    echo 'No saved Phase I checkpoint found. Resume stopped before creating a Colab session.' >&2
    exit 1
  fi
  if ! grep -Fxq 'config.json' <<< "$checkpoint_list"; then
    echo 'Saved model config.json is missing. Resume requires the full model directory on Drive.' >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint_list" | grep -E '^(interface|reflection)/(latest|iteration_[0-9]+)\.pth$'
fi

mkdir -p "$UPLOAD_DIR"
printf '\n==> VIGOR-GS: package code\n'
tar -I 'gzip -1' -cf "$UPLOAD_DIR/vigor_gs.tar.gz" -C "$WORKSPACE" \
  --exclude='*/.git' --exclude='*/.cache' --exclude='*/.tmp' \
  --exclude='*/.conda' --exclude='*/.pytest_cache' --exclude='*/__pycache__' \
  --exclude='*.pyc' --exclude='*.so' --exclude='*.o' --exclude='*.a' \
  --exclude='*.egg-info' --exclude='*/build' --exclude='*/build.before-*' \
  --exclude='*/dist' --exclude='*.zip' --exclude='*.tar.gz' \
  --exclude='vigor_gs/data' --exclude='vigor_gs/output*' --exclude='vigor_gs/logs' \
  vigor_gs

# Only the aligned COLMAP inputs are needed by the scene reader. Raw videos,
# GLINT priors, other scenes, and the original distorted images stay local.
printf '\n==> VIGOR-GS: package VSGD 0758 aligned inputs\n'
tar -I 'gzip -1' -cf "$UPLOAD_DIR/vsgd_0758.tar.gz" -C "$SCENE" \
  --exclude='*/__pycache__' --exclude='*.pyc' \
  --transform='s,^colmap,datasets/VSGD/0758/colmap,' colmap
ls -lh "$UPLOAD_DIR/vigor_gs.tar.gz" "$UPLOAD_DIR/vsgd_0758.tar.gz"
if [[ "${1:-}" == --package-only ]]; then
  exit 0
fi

printf '\n==> VIGOR-GS: upload to Drive\n'
rclone copyto "$UPLOAD_DIR/vigor_gs.tar.gz" "$DRIVE_ROOT/vigor_gs.tar.gz" \
  --progress --drive-chunk-size 64M
rclone copyto "$UPLOAD_DIR/vsgd_0758.tar.gz" "$DRIVE_ROOT/datasets/vsgd_0758.tar.gz" \
  --progress --drive-chunk-size 64M

printf '\n==> VIGOR-GS: create T4 session\n'
printf 'Session: %s\n' "$SESSION"
colab --auth oauth2 new -s "$SESSION" --gpu T4
printf '\n==> VIGOR-GS: mount Drive (complete authorization if prompted)\n'
colab --auth oauth2 drivemount -s "$SESSION"

run_step() {
  local step="$1" timeout="$2"
  printf '\n==> VIGOR-GS: %s\n' "$step"
  colab --auth oauth2 exec -s "$SESSION" --timeout "$timeout" \
    --env "VIGOR_COLAB_STEP=$step" --env "VIGOR_REQUIRE_CHECKPOINT=$REQUIRE_CHECKPOINT" \
    -f "$SCRIPT_DIR/colab_t4.py"
}
run_step prepare 3600
run_step setup 7200
run_step run 86400
echo 'VIGOR-GS VSGD 0758 completed (resolution 1).'
echo 'Results: MyDrive/DACN_HK261/results/vigor_gs/output/vsgd_0758_r1/'
