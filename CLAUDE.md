# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status: README vs. code

The README describes **VIGOR-GS**: two Gaussian fields trained in two phases (a frozen 2DGS interface field used to fit a glass plane, then a 3DGS reflection field rendered from a camera mirrored across that plane, blended with a Fresnel weight), each with its own deformation MLP. The code this repo started from is the **RT-Splatting** (CVPR 2026) baseline (see `git show 1a3c021:README.md`): a single hybrid surface/volume 2D-Gaussian model with a learned specular branch.

VIGOR-GS is being built module by module on top of that baseline, one commit per module (`NN-name: ...` commit messages). The spec is `docs/VIGOR_GS_Academic_Framework (2).pdf` plus `reference.md`; **where they disagree, `reference.md` wins** (the glass plane is extracted from the interface field, and glass masks are used). `guildance.md` is an older commit plan and is not authoritative. The baseline files (`train.py`, `render.py`, `scene/gaussian_model.py`, `gaussian_renderer/__init__.py`) stay untouched for comparison; VIGOR-GS code goes in new files.

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
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
```

After editing any `.cu`/`.h` file under `submodules/`, reinstall that submodule with the same `pip install --no-build-isolation ...` command. Python code imports the compiled package (`diff_surfel_anych`, `simple_knn._C`), not the source tree. GLM sits in `submodules/diff-surfel-anych/third_party/glm` and is included via `-I` in `setup.py`.

Pipeline (one scene; `eval.sh` lists the per-scene hyperparameters used for every benchmark scene):

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

## Architecture

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
