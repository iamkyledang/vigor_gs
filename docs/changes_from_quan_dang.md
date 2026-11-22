# Changes in `f_refactor` relative to `quan_dang`

The `f_refactor` branch was forked from `quan_dang` (commit `9fce787`) and refactored based on `docs/framework.pdf`, with reference to the `viet_huynh` branch. This document lists what was retained, what was corrected because it was wrong or missing relative to the framework, and what was added. Each item corresponds to a commit on the branch (`git log quan_dang..f_refactor`).

## Retained ideas (and most of the code)

- **CUDA renderer using reflected rays** (`render_reflected_rays` in `submodules/diff-surfel-anych/rasterize_points.cu`), with tests comparing its forward pass and gradients against a PyTorch reference. This is the "physically correct" renderer described in Sec. 3.
- Tile candidate lists that find the pixel of each G_R Gaussian by projecting the mirror image of its center across the glass plane with the real camera. This is only a binning step: G_R stays in world coordinates on the camera side of the glass, and the virtual camera V_dyn sits on the glass on that same side (`implementation_details.md` §7.1, §7.4).
- Patch extraction using sequential RANSAC, connected components, and convex hulls, with coverage/precision thresholds.
- Motion loss based on image changes (Eq. 7b), with gradients flowing to both predictions.
- Pixel-normalized losses (sum the 3 channels, divide by the number of valid pixels), L1 L_surf (Eq. 3a), no SSIM.
- Images stored as uint8 in RAM, linear RGB, atomic checkpoint writes with RNG state, comparison videos, "Masked (RT)" metrics, FPS, and training time.

## Corrections for errors or omissions relative to the framework

| Issue in `quan_dang` | Consequence | Fixing commit |
|---|---|---|
| FirstHit (Eq. 9) considers only patches | Glass behind an opaque object is still selected; reflections are blended over that object | `f_glass_geometry`, `f_occlusion_glass_layer`: check occlusion using frozen G_I depth, excluding the layer of disks on the glass; `--ablate_occlusion` ablation available |
| RT-Splatting's 2DGS rasterizer has `DUAL_VISIABLE 0` | Disk normals do not face the camera (41% measured as facing it), so the L1 term in Eq. 3a penalizes them and forces the disks to rotate 180° | `d_phase_one_interface`: `DUAL_VISIABLE 1`, as in 2DGS [1] |
| h_ψ takes raw pixel coordinates (u, v), bounded to ±0.1 by tanh | The same pixel in two views corresponds to different glass locations, so the correction does not transfer correctly to test views; the bound is absent from Eq. 11 | `f_reflection_renderer`: inputs are the glass point x_t(u), time, and cos θ; unbounded by default |
| V_dyn (Eq. 9b) is computed but discarded; u_ref is the middle pixel in scan order | No way to compare against the pinhole approximation mentioned in the PDF | `f_glass_geometry`, `f_reflection_renderer`: u_ref is the pixel nearest the region center; the pose is used for the `--pinhole_virtual_camera` ablation and exported to `virtual_cameras.json` |
| Reader loads masks from `source_mask` (distorted source-image coordinates) and refits the distortion model on every load | On aligned data (`colmap/masks`, `colmap/clean_image`), masks are misaligned with images; loading is slow | `f_scene_reader`: read only the aligned layout; `undistort_sources.py` performs alignment once per scene |
| No presence reset, no pruning by screen size, incorrect point budget | Excess Gaussians are not removed; large scenes produce huge Gaussians that inflate the rasterizer buffer (OOM) | `f_gaussian_fields` |
| `max_radii2D` is not updated; views sampled with `random.choice` | Screen-size pruning does not work; some views are skipped in an epoch | `d_phase_one_interface` |
| Patch extraction subsamples with unseeded `torch.randperm` | Each extraction produces a different patch set (coverage 0.89 then 0.72 from the same Phase I) | `f_extraction_determinism` |
| No ablations | Individual components cannot be validated | `d_phase_two_reflection`, `d_pipeline_ablations`: 12 variants; the default run is five ablations compared against a separately trained `full`, at `-r 1` (VSGD, 3D-FRONT-T) or `-r 2` (Wayve) (`docs/ablations.md`) |
| Metrics compare only against the original images | Layer-separation quality is not measured | `f_render_metrics`: Transmission PSNR (B against clean images) |

## Cleanup

- Removed `log/` (nearly 1 MB of patches and source snapshots), `record_implementation_change`, and the watcher in `eval.sh`. Git already tracks this history.
- Removed unused RT-Splatting utilities: `network_gui.py`, `mesh_utils.py`, `mcube_utils.py`, `render_utils.py`, `easyvolcap_utils.py`, `sph_utils.py`, `camera_utils.py`, `calibration_utils.py`.
- Replaced `specification.md` (specifications and verification results for the old code) with `docs/implementation_details.md`, `docs/ablations.md`, and this document. Moved `framework.tex` into `docs/`.
- Rewrote the code for readability: one script per stage (`train_interface.py`, `extract_glass.py`, `train_reflection.py`, `render.py`, `metrics.py`, plus `train.py` to run the entire pipeline), docstrings citing equation numbers, and no more densely packed lines containing multiple statements.
- Migrated tests to pytest with a physically consistent synthetic scene (`tests/synthetic.py`): 50 tests covering geometry, CUDA, individual phases, ablations, and the pipeline.

## Backward-incompatible changes

- Checkpoint version 2 (version 1 from `quan_dang` cannot be read). Model directory structure: `config.json`, `interface/`, `glass_geometry.json`, `reflection/`.
- Command line: `train.py --phase ...` and `--resume <path>` are replaced by stage-specific scripts (each with `--resume`) and `train.py`, which automatically skips or resumes each stage.
- Some parameters were renamed: `--reflection_iterations` becomes Phase II's `--iterations` (passed via `--reflection_args` when running `train.py`); `--plane_*` and occlusion parameters are grouped under `GlassParams`.
