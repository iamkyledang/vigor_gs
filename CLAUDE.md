# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Two code paths in one repo

1. **VIGOR-GS** (the project): two Gaussian fields trained in two phases, each with its own deformation MLP. See "VIGOR-GS architecture" below.
2. **RT-Splatting baseline** (CVPR 2026, what the repo started from, see `git show 1a3c021:README.md`): a single hybrid surface/volume 2D-Gaussian model with a learned specular branch. Its files (`train.py`, `render.py`, `scene/gaussian_model.py`, `gaussian_renderer/__init__.py`, `eval.sh`) are kept untouched for comparison. See "Baseline architecture".

VIGOR-GS is built module by module, one commit per module (`NN-name: ...` commit messages, written in Vietnamese). The spec is `docs/VIGOR_GS_Academic_Framework (2).pdf` plus `reference.md`; **where they disagree, `reference.md` wins** (the glass plane is extracted from the interface field, and glass masks are used). `guildance.md` is an older commit plan and is not authoritative. `docs/implementation_notes.md` (Vietnamese) records what was built, the design choices still up for review, and the trial-run numbers; update it when results or decisions change.

Open issues that are not code bugs:
- **No clean images yet.** Phase I should train on reflection-removed targets; the datasets have none, so runs use `--clean_fallback_original` and the reflections get baked into the direct image, leaving Phase II nothing to explain.
- **Single-plane assumption vs. the data.** VIGOR-GS assumes one glass plane. The glass in `3D-FRONT-T/scene_1` is a multi-pane glass box with cameras orbiting it, so one plane fits only one face (about 40% inliers).
- Pure Schlick gives a reflection weight of about 0.04 at normal incidence, so the reflection field must be very bright to matter, and its gradients are scaled down by the same factor.

## Environment & commands

**On this machine everything runs in WSL** (Windows has no Python or CUDA toolkit). The conda env `rtsplat` already has torch 2.1.2+cu121 and the built extensions; the GPU is an RTX 3050 with 4 GB, so only low-resolution, short runs fit. Run commands like this from the Windows shell:

```shell
wsl -d Ubuntu -e bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate rtsplat && cd /mnt/d/Research/ODE/Transparent/vigor_gs && python -m pytest'
```

Use **Windows git** for commits: the working tree has CRLF line endings (`core.autocrlf=true`), so git inside WSL reports every file as modified.

Tests live in `tests/` (`python -m pytest`, or `python -m pytest tests/test_env.py::test_surfel_rasterizer_smoke` for one). Tests marked `gpu` or `data` skip themselves when CUDA or the dev scene (`data/datasets/datasets/3D-FRONT-T/scene_1`, override with `VIGOR_TEST_SCENE`) is missing.

The project needs CUDA. Every model, rasterizer and loss calls `.cuda()` unconditionally. Fresh install (the README pins torch 2.0.1; the working env here uses 2.1.2+cu121 with the system CUDA 12 toolkit):

```shell
conda create -n rtsplat python=3.10 -y && conda activate rtsplat
pip install -r requirements.txt            # torch==2.0.1, numpy<2, setuptools==81.0.0 are pinned
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-surfel-anych
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
```

After editing any `.cu`/`.h` file under `submodules/`, reinstall that submodule with the same `pip install --no-build-isolation ...` command (in WSL, put `/usr/local/cuda/bin` on `PATH` first). Python code imports the compiled package (`diff_surfel_anych`, `diff_gaussian_rasterization`, `simple_knn._C`), not the source tree. GLM sits in `submodules/diff-surfel-anych/third_party/glm`; both rasterizers include it from there.

VIGOR-GS pipeline for one scene (`-m` is the model root; each step reads what the previous one wrote):

```shell
python train_interface.py -s data/datasets/datasets/3D-FRONT-T/scene_1 -m output/scene_1 -r 4 --eval --clean_fallback_original
python extract_plane.py -m output/scene_1          # or: --glass_plane NX NY NZ D to set it by hand
python train_reflection.py -m output/scene_1
python render_vigor.py -m output/scene_1           # --skip_train to render the test split only
python metrics.py -m output/scene_1
```

- On the 4 GB GPU use `-r 4`, a small deformation MLP (`--deform_width 64 --deform_depth 4`) and a few thousand iterations; the defaults are 30,000 iterations per phase.
- Scene settings (`-s`, `-r`, `--eval`, deformation sizes, `--linear_rgb`) are given once to `train_interface.py`. Later steps read them from `<model>/interface/cfg_args`, and `render_vigor.py` from `<model>/reflection/cfg_args`.
- `train_interface.py` trains on linear RGB unless `--srgb` is passed; saved images are converted back to sRGB.

Baseline pipeline (one scene; `eval.sh` lists the per-scene hyperparameters used for every benchmark scene):

```shell
python train.py -s data/rt-splatting/van -r 4 --eval --env_scope_center -0.5 -1 0.5 --env_scope_radius 2.5 --init_until_iter 700 --norm_loss_from_iter 700 --xyz_axis 2.0 1.0 0.0
python render.py -m output/van      # add --skip_mesh to skip TSDF meshing; --render_path for a video trajectory
python metrics.py -m output/van     # writes results.json / per_view.json into the model dir
```

- The output dir defaults to `./output/<basename of source path>`. `render.py` reloads training args from `<model_path>/cfg_args` (`get_combined_args`), so it only needs `-m`.
- `--test_iterations` defaults to every 1000 iterations, and each run evaluates the full test set plus LPIPS. For quick experiments, pass e.g. `--test_iterations 61000` or a shorter `--iterations` (default 61000).
- Lint and format with ruff (`ruff.toml`: line length 210, single quotes, isort rules): `ruff check .` and `ruff format .`.

## Data requirements

**VIGOR-GS layout** (`readVigorSceneInfo`, chosen when `<scene>/colmap/frames.json` exists; this is what `data/datasets/datasets/*` uses):
- Images come from `colmap/images/`, poses from `colmap/sparse/0/`, the train/test split from `colmap/split.json` (with `--eval`; `llffhold` is not used), and the per-frame `time` in [0, 1] from `colmap/frames.json`.
- Glass masks: `colmap/masks/<stem>.png`, else the frame's `source_mask` path. Only 3D-FRONT-T masks are aligned with the COLMAP images; VSGD/Wayve masks are in the original (pre-undistortion) image size, and Wayve masks are privacy masks, not glass.
- Clean (reflection-removed) targets: `<scene>/<--clean_images>/<stem>.*` (default folder `clean`). **The datasets ship none yet**, so loading fails unless `--clean_fallback_original` is passed, which makes `Camera.clean_image` alias the original image (`Camera.has_clean_image` tells which).
- `--linear_rgb` converts both targets from sRGB to linear at load time.

**Baseline layout** (plain COLMAP scene, used by `eval.sh`):
- Only the COLMAP loader works end to end. A scene needs `images/`, `sparse/0/` and **`transparent_masks/<image_stem>.png`** for every image; loading fails without the masks. The Blender (`transforms_train.json`) loader never sets `transparent_mask` on `CameraInfo`, so it crashes in `loadCam`.
- Cameras must be PINHOLE or SIMPLE_PINHOLE (undistorted). `convert.py` is the standard 3DGS COLMAP conversion script.
- With `--eval`, every 8th image (sorted by name) goes to the test split (`llffhold=8`).

## VIGOR-GS architecture

**Results layout.** `<model>/interface/` (Phase I: `cfg_args`, `point_cloud/iteration_N/point_cloud.ply`, `deform/iteration_N/deform.pth`), `<model>/glass_plane.json`, `<model>/reflection/` (Phase II, same structure, plus `blend/iteration_N/blend.pth` for the alpha parameters). `utils/vigor_utils.py` holds these names and `load_interface` / `load_reflection`.

**Two fields on one base class.** `scene/gaussian_field.py::GaussianField` owns parameters, the optimizer, clone/split/prune surgery, PLY I/O and checkpoints. Per-Gaussian parameters are declared once in the `PARAMS` tuple (parameter `name` lives in `self._<name>` and in the optimizer group `name`); a subclass that adds one extends `PARAMS`, `_initial_extra_params` and `_learning_rates`, nothing else.
- `InterfaceGaussianModel` (`scene/interface_model.py`): 2D disks with **occupancy `p`** (geometric presence) and **opacity `o`** (optical contribution). Densification and pruning follow `p`.
- `ReflectionGaussianModel` (`scene/reflection_model.py`): plain 3DGS ellipsoids, initialized from random points on the camera side of the glass (the reflected scene has no SfM points).

**Deformation** (`scene/deform_model.py`): one `DeformModel` per field maps (canonical center, time) to `(d_xyz, d_rotation, d_scaling)`. The output layer is zero-initialized, so a fresh network is the identity. `p`, `o` and colors are never deformed. Renderers take the offsets as an argument (`offsets=None` renders the canonical field), so the training scripts control warm-up and timestamp noise.

**Phase I** (`train_interface.py`, `gaussian_renderer/interface.py`). `render_interface` rasterizes the disks twice with `diff_surfel_anych`: a volume pass with `p*o` gives the direct image `B_t` (`'direct'`), and a surface pass with `p` alone gives the interface geometry and `'glass_map' = 1 - surface_opacity`, i.e. where the first surface is present but transparent. Losses: L1+D-SSIM of `B_t` against `Camera.clean_image`, normal consistency, BCE of the glass map against the glass mask, occupancy decay.

**Glass plane** (`extract_plane.py`, `utils/plane_utils.py`): median-depth points of the surface pass inside the glass masks, pooled over training views, fitted with RANSAC + SVD. Convention: `n^T x + d = 0`, `|n| = 1`, cameras on the positive side.

**Phase II** (`train_reflection.py`, `gaussian_renderer/vigor.py`). Phase I is frozen and its `B_t` and glass map are rendered once per training view (`DirectCache`). The reflection field stays in **real world coordinates** and is rendered through the virtual camera `reflect_camera` (`utils/virtual_camera.py`), whose view matrix is `W2C * M` with `M` the plane reflection; the result lines up pixel for pixel with the real camera, so nothing is flipped and the Gaussians are never mirrored. `utils/fresnel.py` gives `alpha = clip(Schlick(cos) + h_psi) * glass_map` and the blend `(1 - alpha) B + alpha R`, supervised by the original image. `BlendModel` bundles whatever decides alpha (no parameters by default) and is what gets saved and passed to `render_composite`.

**Inference** (`render_vigor.py`) writes the same `renders/`, `gt/`, `transparent_masks/`, `vis/` layout as the baseline, so `metrics.py` is shared.

**Ablations** are `ModelParams` flags, so they are saved in `cfg_args` and every later step of the same model follows them. Give each variant its own `-m` folder.

| Flag | Given to | What it removes |
|---|---|---|
| `--ablate_factorization` | `train_interface.py` | The `p*o` factorization: one opacity drives both passes (`InterfaceGaussianModel(factorized=False)`). |
| `--ignore_clean_images` | `train_interface.py` | Clean supervision: one field fits the original images. Evaluate with `render_vigor.py --direct_only`, which needs no Phase II and writes `direct_<iter>/`. |
| `--ablate_virtual_camera` | `train_reflection.py` | The mirrored camera: the reflection field is rendered from the real camera. |
| `--ablate_fresnel` | `train_reflection.py` | The Schlick weight: one learned constant per scene, still gated by the glass map. |
| `--shared_reflection_field` | `train_reflection.py` | The separate reflection field: the reflection is the frozen interface seen from the virtual camera. With pure Fresnel there is nothing to train and the result is saved as iteration 0. |

**Arguments.** `ModelParams` is shared by both code paths and saved to `cfg_args`; `InterfaceOptimizationParams` and `ReflectionOptimizationParams` belong to the two phases. `get_combined_args(parser, cfg_subdir=...)` drops command-line options left at `None`, so read optional script flags with `getattr`; parameters missing from an older `cfg_args` fall back to their defaults in `ParamGroup.extract`.

## Baseline architecture

**One `GaussianModel` (`scene/gaussian_model.py`), 2D surfels.** Scales have 2 components (2DGS). Each Gaussian carries SH color, `occupancy`, `opacity`, `transmissivity`, `reflectance`, `roughness` and a 4-dim `language_feature` (a misnomer: it is the specular feature, not language). The model also owns two global networks: `dir_encoding` (`SphMipEncoding`, a learnable equirectangular feature map sampled with `nvdiffrast.torch.texture` with mip level from roughness) and `light_mlp`.

**`env_scope_center` / `env_scope_radius` / `xyz_axis`.** `get_inside_mask` limits reflectance and transmissivity to Gaussians inside this sphere, i.e. the reflective/transparent object. Outside it they are masked to 0. `xyz_axis` is an axis permutation (cast to int) applied before `cart2sph` for the reflection direction. Every scene needs these values hand-tuned (see `eval.sh`).

**`render()` (`gaussian_renderer/__init__.py`) runs the rasterizer twice on the same Gaussians:**
1. Volume pass, `opacities = occupancy * opacity` → `render_tran` (the transmitted background seen through glass).
2. Surface pass, `opacities = occupancy` → `render_scat` (the diffuse surface), plus per-pixel `extras` channels that unpack, in order, as `[roughness(1), feature(4), inside_mask(1), reflectance(1), opacity(1), transmissivity(1)]`. Any new extras channel must be added to the `torch.cat` and to the `.split([...])` together.
3. Deferred specular: the view direction is reflected about the surface normal, the encoding is sampled at the roughness mip level, and the MLP gives specular light (exp) and attenuation (sigmoid). This runs only on pixels with `foreground > 0.05`.
4. Composite: `final = tran*T*att + scat*(1-T)*att + spec*reflectance`. While `pipe.init_stage` is true, spec and attenuation are left out.

`diff_surfel_anych` is a 2DGS rasterizer modified to accept an arbitrary number of `extras` channels. It returns `(color, extras, radii, allmap)`, where the `allmap` channels are: 0 = alpha-weighted expected depth, 1 = alpha, 2:5 = view-space normal, 5 = median depth, 6 = distortion.

**Training schedule (`train.py`) is driven by iteration thresholds**, which is why `eval.sh` varies them per scene:
- `iter < norm_loss_from_iter`: L1+SSIM on `render_tran` only.
- `iter < init_until_iter`: L1+SSIM on `final_rendering` without spec (`pipe.init_stage=True`).
- After that: the full composite loss. From `mask_loss_from_iter` on (`-1` means `init_until_iter`), `final_tran` is partly detached by a local-variance "spec complexity" weight. Further terms: LPIPS (from `lpips_loss_from_iter`), BCE mask loss (`1 - surface_opacity` vs. transparent mask), a transmissivity loss pushing T→0 outside the mask, a consistency loss on T and scatter inside the mask, occupancy decay, and a normal-consistency loss.
- Densification and pruning use **occupancy**, not opacity. `reset_occupancy` runs every `occupancy_reset_interval`, and `reset_opacity` runs half an interval later. `last_update` / `last_reset_iter` keep Gaussians from being pruned before they have been seen since the last reset.

**Adding a per-Gaussian attribute** means updating every one of: `create_from_pcd`, `training_setup` (optimizer group `name` must match), `construct_list_of_attributes` / `save_ply` / `load_ply`, `_prune_optimizer` / `prune_points`, `densification_postfix`, `densify_and_split`, `densify_and_clone`, and `capture` / `restore`.

**Persistence caveats.** `save_ply` writes `point_cloud.ply` plus `light_mlp.pt` and `dir_encoding.pt` (whole pickled modules) in `point_cloud/iteration_N/`. `capture()`/`restore()` checkpoints (`--checkpoint_iterations` / `--start_checkpoint`) leave out `_opacity`, `_roughness`, `_reflectance`, `_language_feature`, the MLP and the encoding, so resuming from a `.pth` checkpoint is incomplete.

**Render outputs → metrics.** `GaussianExtractor.reconstruction` (`utils/mesh_utils.py`) writes `renders/`, `gt/`, `transparent_masks/` and `vis/` (decomposition maps) under `<model>/test/ours_<iter>/`. `metrics.py` reads exactly that layout and reports full-image and masked (transparent-region) SSIM/PSNR/LPIPS (VGG). `gaussian_renderer/network_gui.py` is unused.
