# VIGOR-GS

Implementation of [framework.tex](framework.tex), replacing the inherited RT-Splatting mechanism. Read [specification.md](specification.md) for the equations, approved geometry-extraction prior, renderer approximations, module reading order and validation status.

The local environment is `.conda`. Commands below run from this folder and use its Python directly; activation is optional:

```bash
cd /mnt/passport/Lighting_GS/VIGOR_GS
# Optional: conda activate /mnt/passport/Lighting_GS/VIGOR_GS/.conda
```

## RTX 3060 6 GB quick test: train, render, metrics

This reduced configuration uses quarter resolution, smaller deformation MLPs and explicit point limits. It is designed to stay within a 1–2 hour test budget and may finish substantially sooner; final runtime depends on the learned Gaussian/tile counts, GPU power and disk speed. Confirm the measured iteration rate in `output/vsgd0758_quick/*/training.jsonl`. It is a shorter quality experiment, not a converged benchmark.

```bash
.conda/bin/python train.py -s ../datasets/VSGD/0758 -m output/vsgd0758_quick -r 4 --iterations 6000 --reflection_iterations 6000 --init_points 30000 --reflection_init_points 20000 --max_interface_points 100000 --max_reflection_points 50000 --deform_depth 3 --deform_width 64 --warm_up 500 --surface_from_iter 500 --densify_until_iter 4000 --densification_interval 200 --position_lr_max_steps 6000 --checkpoint_interval 2000 --cache_views 8 --max_tile_entries 12000000

.conda/bin/python render.py -m output/vsgd0758_quick --skip_train --components

.conda/bin/python metrics.py -m output/vsgd0758_quick
```

Training runs both sequential phases. If inferred glass geometry fails the coverage/precision checks, Phase II stops and the Phase I checkpoint plus `glass_geometry.json` remain available for inspection. Do not lower those checks merely to claim a successful reconstruction. Improve Phase I or supply reliable finite geometry with `--glass_geometry`.

Outputs include `interface/latest.pth`, `reflection/latest.pth`, `glass_geometry.json`, `data_alignment.json`, `config.json`, training JSONL logs, rendered test components, `results.json`, and `per_view.json`. Final images and quantitative metrics are in sRGB; fitting and composition use linear RGB.

Every rendered split also gets `comparison.mp4` inside its method folder, for example `output/vsgd0758_quick/test/ours_6000/comparison.mp4`. It shows **Ours on the left** and **GT on the right**, with labels above the images. GT is the original observed photograph used for composite evaluation. Playback defaults to 30 FPS; set `--video_fps 15` to slow it down. Frames are ordered by sequence, camera and timestamp. Video encoding runs after rendering and is excluded from the measured rendering FPS.

`results.json` also reports rendering `FPS` and total `Training Time` as `xx hours and xx mins`, with precise total/per-phase seconds. FPS excludes image saving; training time covers the optimization loops across both phases and resumed sessions. Unavailable timing is `null`. To update an existing result without recomputing PSNR/SSIM/LPIPS:

```bash
.conda/bin/python metrics.py -m output/vsgd0758_quick --performance_only
```

## Full-resolution quality defaults

Resolution defaults to **1**. The default configuration uses 30,000 iterations per phase, SH degree 3, 4-layer/128-wide deformation MLPs, up to 80,000 SfM initialization points, 50,000 initial reflection points, and caps of 500,000/250,000 points. The full-resolution tile budget is 64 million entries; the quick command lowers it to 12 million. These are quality-oriented starting defaults; their empirical optimum and comparative benchmark quality have not been established. Full-resolution runs may need more than 6 GB VRAM.

```bash
.conda/bin/python train.py -s ../datasets/VSGD/0758 -m output/vsgd0758_full
.conda/bin/python render.py -m output/vsgd0758_full --skip_train --components
.conda/bin/python metrics.py -m output/vsgd0758_full
```

## Resume and inspect

```bash
# Continue either phase; saved model configuration is reused.
.conda/bin/python train.py -s ../datasets/VSGD/0758 -m output/vsgd0758_quick --resume output/vsgd0758_quick/interface/latest.pth
.conda/bin/python train.py -s ../datasets/VSGD/0758 -m output/vsgd0758_quick --resume output/vsgd0758_quick/reflection/latest.pth

# Render Phase I without a reflection checkpoint.
.conda/bin/python render.py -m output/vsgd0758_quick --direct_only --skip_train

# Record implementation changes immediately, or continuously while editing.
bash eval.sh --snapshot
bash eval.sh --watch
```

The first resume command continues Phase I to its configured iteration count and then starts Phase II. The second resumes Phase II directly. Increase a phase length with `--iterations` or `--reflection_iterations`. Choose a fresh model directory for a separate experiment; existing checkpoints are protected from an accidental new training run.

## Rebuild the CUDA extension

The local compiled extension includes the existing 2DGS rasterizer and the new routed 3D Gaussian ray renderer. Rebuild after changing CUDA source:

```bash
cd submodules/diff-surfel-anych
CUDA_HOME=../../.conda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 ../../.conda/bin/python setup.py build_ext --inplace
cd ../..
```

Use an absolute `CUDA_HOME` if your compiler requires it. CUDA 11.8, Python 3.10, PyTorch 2.0.1+cu118 and a compatible C++ compiler are required by the prepared environment. The source submodule is selected at import time, leaving RT-Splatting's installed renderer intact.

## Read and test the implementation

The [module reading order](specification.md#module-responsibilities-and-reading-order) follows training through field storage, deformation, glass geometry, rendering, calibration and checkpoints. `train.py` retains phase scheduling and losses; `gaussian_renderer/__init__.py` exposes the rendering API. Checkpoint loading lives in `utils/checkpoint_utils.py` and is shared by training and rendering.

```bash
.conda/bin/python -m unittest discover -s tests -v
```

The tests use the standard library's `unittest`; no additional test dependency is needed. CPU checks cover geometry, tile coverage, calibration conventions and composition. CUDA checks compare reflected-ray outputs/gradients with an independent PyTorch reference and verify exact optimizer continuation, RNG restoration, frozen fields and point-cap behavior. GPU cases report skips if CUDA is unavailable. The refactor keeps the training/rendering/metrics commands and existing version-1 checkpoints compatible.

## Evaluation conventions

`PSNR`, `SSIM` and standard-range `LPIPS` are full-image metrics. `Glass PSNR` normalizes by actual valid glass pixels. Separately labeled padded-mask metrics and `LPIPS (RT input)` retain the inherited baseline conventions for comparison; padding can inflate a masked PSNR. For a fair study, run both methods at the same resolution, split, alignment and metric convention. `--skip_lpips` explicitly omits LPIPS for a dependency smoke check; normal evaluation includes it.

This is a research implementation. Refraction and recursive reflections are outside scope, and extracted glass depth is not guaranteed to be physically identifiable from clean RGB plus silhouettes.

## Verified on this computer

The local `.conda` environment, CUDA forward/backward checks, geometric ray-routing checks, both training phases, point caps, checkpoint resume, 29 test renders and PSNR/SSIM/VGG-LPIPS evaluation passed.

A quarter-resolution run with 100 iterations per phase took 6.29 s in Phase I and 13.17 s in Phase II (training-loop timings, excluding startup, loading and plane extraction). It ended with 15,028 interface and 25,321 reflection Gaussians, and a peak PyTorch allocation of 290.3 MiB. Mean test render GPU time was 26.45 ms. Driver/context/desktop memory is additional.

That deliberately short smoke run reached 16.40 dB test PSNR; this verifies the evaluation path and is **not** a converged quality result or a comparison against RT-Splatting. The 6,000-iteration quick experiment above has not been run to completion during implementation.
