# VIGOR-GS implementation specification

This checkout implements `framework.tex` in the existing RT-Splatting source files. The original RT optical MLP mechanism has been replaced. Reference code from `../vigor_gs_viet_huynh` informed Gaussian storage, optimizer surgery and canonical deformation; no runtime imports depend on that checkout.

## Decisions agreed with the user

- Reuse existing implementation files. New source modules are not required.
- `specification.md`, dated records under `log/`, runtime training/render outputs, and a local `.conda` environment are authorized.
- Use **training** glass annotations to guide Phase I surface fitting. Extract finite 3D glass patches after Phase I and freeze them for Phase II. Test masks are used for reporting only.
- The masks add an explicit supervised geometry prior beyond the original clean-RGB-only Phase I equation. They do not become learned plane assignments or inference blending masks.

## Inputs and coordinate alignment

The copied scene is `../datasets/VSGD/0758`. All 444 files from `~/Downloads/VSGD/0758` have matching SHA-256 digests at the destination. The copy was merged into the existing scene, preserving existing original JPGs and COLMAP reconstruction files.

- `train/images/*.png`, `test/images/*.png`: downloaded reflection-removed supervision.
- `colmap/images/*.jpg`: undistorted **original** observed images.
- `source_metadata/colmap/sparse/0`: camera metadata. For 0758 this is also the undistorted model, so it cannot directly rectify the clean PNGs. The source lens model is recovered from matched original JPGs, using training views only and a held-out training-view check.
- `colmap/sparse/0`: target camera calibration, poses and SfM initialization points.
- `colmap/frames.json`: explicit timestamps, sequence identity, and train/test split. No random holdout or time renormalization is introduced.

When original source intrinsics are missing, radial/fisheye recovery follows the useful calibration method in the reference implementation. Its parameters and held-out pixel residual are saved in the model output as `data_alignment.json`; a median residual over one pixel or fewer than 20 supporting validation matches is an error. Rectification happens in memory at the requested training resolution. Missing clean training targets are errors: original images never silently substitute for clean images. Intrinsics include the principal point and use COLMAP pixel centers at +0.5. Undistortion padding is excluded from training losses. Images stay as byte tensors on CPU unless explicitly requested otherwise; only the selected target is decoded to floating-point RGB. This avoids retaining the entire full-resolution dataset as float32 images.

## Fields and deformation

`scene/gaussian_model.py` contains both independent Gaussian fields and their independent MLPs:

- Interface: 2D disks, position, quaternion, two log scales, SH appearance, occupancy `p`, opacity `o`. The direct pass always uses `p * o`.
- Reflection: 3D ellipsoids with three scales, independent appearance/opacity and independent optimizer state.
- Each deformation MLP takes encoded normalized canonical position and time, and predicts position, quaternion and log-scale offsets. The output heads start at zero. Occupancy, opacity and appearance remain canonical and time independent.
- Topology changes preserve Adam state for surviving points and initialize new moments to zero. Configured point caps limit growth. Both the canonical fields and deformation networks are checkpointed.

Occupancy and opacity cannot be uniquely separated using their photometric product alone. The training glass prior makes the distinction useful but does not prove physical identifiability.

## Phase I

`train.py --phase interface` fits the clean targets with the 2DGS rasterizer inherited from RT-Splatting.

The specified photometric objective is a sum of absolute RGB errors **per evaluated pixel**, averaged over valid pixels. It is evaluated in linear RGB. Surface loss compares unit rendered normals with unit normals derived from the rendered depth. Pixels with missing neighbors, missing coverage or invalid normals are excluded. Empty surface support contributes zero surface penalty.

For the authorized geometry prior, an occupancy-only surface pass also renders alpha and opacity. Its glass signal is `surface_alpha - rendered_opacity`, equivalent to accumulating conditional transparency with surface visibility. Binary cross-entropy against the training glass annotation guides the surface parameters. It is weighted by `lambda_glass`; the prior can be disabled when geometry is supplied externally. This signal is used for geometry estimation, not final optical blending.

SSIM is an optional ablation/quality experiment and defaults to zero, as the supplied equations omit it. No LPIPS training loss or RT appearance MLP remains.

## Extracted finite glass geometry

After Phase I, masked training views supply world-space surface samples. Robust plane hypotheses are refined by SVD. Coplanar supports are split into connected components, and each component receives a finite convex polygon in its plane. `max_patches` is an upper bound, not a mandatory number of planes.

The result is written to the model output as `glass_geometry.json`, including vertices, normal, offset, inlier counts and residual errors. Coverage and precision are measured by projecting the fixed patches into sampled **training** cameras and comparing their geometric ray hits with the annotations. Defaults require both to be at least 0.5. Low quality stops Phase II while retaining the Phase I checkpoint and the geometry report.

This is an estimated geometry extension to the supplied specification's assumption of known planes. Clean 2DGS and silhouettes do **not** uniquely determine transparent glass depth. Fit residuals and silhouette agreement are useful checks, not evidence of physical ground truth. Convex supports approximate boundaries; holes or complex nonplanar glass need externally supplied geometry or a more elaborate reconstruction method.

Alternatively, `--glass_geometry PATH` accepts JSON containing `patches`, each with `normal: [nx,ny,nz]`, `offset: d`, and ordered, coplanar, convex `vertices: [[x,y,z], ...]`. The equation is `n dot x + d = 0`. Unit normalization and polygon validation happen before use. No geometry file is added to the source tree.

## Phase II and reflected rays

The entire interface field and interface deformation network are frozen, including `p` and `o`. Direct images are cached on CPU with a bounded LRU cache. The reflection field and its deformation MLP train independently.

For each real pixel:

1. Construct its calibrated world ray and intersect finite glass patches.
2. Select the nearest positive glass hit. No hit means reflection weight zero.
3. Start the reflected ray at that hit plus a small forward epsilon; reflect its direction by the selected normal.
4. Query the shared 3D reflection field on that forward ray.

`dynamic_virtual_camera` implements the proper LookAt control pose of Eq. 9b and reuses one camera-state function. This pose is diagnostic/control state; it is not substituted for the per-pixel reflected rays.

The existing `submodules/diff-surfel-anych` extension now also contains a 3D Gaussian ray renderer. Conservative projected ellipsoid bounding spheres build sparse tile candidate lists. The renderer routes candidates by the selected geometric patch ID and writes **one** reflection image. It never renders independent per-plane images for blending. There are no trainable plane assignments or per-plane blend weights.

For an ellipsoid with world-to-local inverse axes `A`, the renderer finds the closest point along `origin + s * direction` in the Gaussian metric. It evaluates the Gaussian there, rejects closest points behind the hit, and composites front to back. CUDA backward differentiates positions, inverse axes, opacity and SH-derived color. PyTorch propagates inverse-axis gradients into rotation, scale and deformation. Ray origins/plane routing are fixed in Phase II.

As in 3DGS, visibility order uses Gaussian center depth within each tile. The reflected rays and finite-support intersections are geometrically correct; Gaussian center sorting, finite support cutoff, alpha thresholds and closest-approach opacity remain rendering approximations. This is not an analytic volume integral or an exact per-pixel sort of intersecting anisotropic ellipsoids. It does not model refraction, recursive reflections or multiple glass interactions.

## Optical composition and losses

All fitting and blending use linear RGB. Standard sRGB decoding is applied to targets; SfM initial colors are converted during interface initialization; outputs are encoded to sRGB for PNGs and benchmark metrics.

`alpha = clip(F0 + (1-F0)*(1-cos(theta))^5 + h(u,t), 0, 1)` on geometric glass hits and zero elsewhere. Refractive indices default to air 1.0 and glass 1.5. The correction is a small, zero-initialized pixel/time network bounded by `optical_bound`; `--pure_fresnel` removes it.

`prediction = (1-alpha)*direct + alpha*reflection`.

Reflection RGB loss is the per-pixel RGB L1 sum. Motion loss compares the predicted image difference with the observed difference at the same pixel coordinates in neighboring training timestamps from the same sequence. Gradients reach **both** predictions. It is not a deformation-magnitude penalty. Optical loss is the mean squared correction, with zero correction outside glass.

Reflection initialization samples points along valid reflected training rays. Initial nonnegative linear radiance comes from the original/direct residual divided by Fresnel weight, so appearance is allowed to exceed one before the final blend.

## Checkpoints, memory and metrics

`output/<run>/interface/latest.pth` and `reflection/latest.pth` contain field parameters, active SH degree, optimizers, deformation networks and RNG state. Reflection checkpoints also contain the optical network and its optimizer. Checkpoints are saved after optimizer updates and replaced atomically. Resume with `--resume PATH`; saved configuration is reused unless explicitly overridden. Architecture, resolution and optical-model settings must remain compatible. Outputs include `config.json`, training JSONL timing/memory records, and frozen geometry.

Point limits and tile-entry limits are enforced. Exceeding the tile budget produces an explicit error instead of silently dropping Gaussian contributions. Quick settings reduce resolution, MLP width, point counts, growth frequency and iteration counts. Full-resolution defaults prioritize model capacity and are not a promise to fit in 6 GB.

`render.py` writes sRGB test renders, original ground truth, aligned glass masks and valid-pixel masks. `--components` also exports direct, reflection and alpha images. Test rendering uses the frozen 3D patches, never test glass masks for blending.

`metrics.py` streams one image at a time. It reports full-image PSNR/SSIM/LPIPS (standard LPIPS input range -1 to 1), a separately labeled LPIPS using the inherited RT input convention, valid-pixel PSNR, actual glass-pixel PSNR, opaque-pixel PSNR, and separately labeled padded-mask metrics matching the RT-Splatting protocol. Padded-mask PSNR includes zero-valued background pixels and is not a glass-only PSNR. Missing LPIPS weights are not silently ignored; `--skip_lpips` is an explicit smoke-test option.

## Environment and implementation history

The project environment is `VIGOR_GS/.conda`, based on the existing Python 3.10 / PyTorch 2.0.1 CUDA 11.8 RT-Splatting environment. The project's own rebuilt CUDA extension is loaded from its existing submodule path, so installing it does not replace RT-Splatting's installed renderer. No tiny-cuda-nn or new 3D rasterizer submodule is needed.

`log/` holds dated implementation notes, unified patches and the last source snapshot. Training automatically records any source change since the previous snapshot. For continuous logging while editing, run `bash eval.sh --watch` in another terminal. Use `bash eval.sh --snapshot` for an immediate snapshot. The watcher is optional and is not installed as a system service.

## Commands and validation

See `README.md` for the three-command quick test and the full-resolution defaults. Timing and GPU validation results are recorded here after checks finish. Research quality and improvements over the RT-Splatting baseline require completed, comparable training runs; they are not established by implementation checks.

### Validation recorded during implementation

- All 444 downloaded files verified by SHA-256 after copying.
- CUDA extension built successfully for the RTX 3060 (SM 8.6) with the existing CUDA 11.8 / PyTorch 2.0.1 toolchain.
- CUDA ray forward and backward checked against an independently expressed PyTorch reference on mixed-patch rays, including a Gaussian behind the ray origin. Maximum absolute gradient discrepancy: 7.16e-7 (opacity); other parameter gradients below 2.39e-7.
- Empty tile lists and a nondefault CUDA stream passed.
- Source-camera recovery: radial model; held-out training-frame median residual 0.15971 pixels, 348 supporting validation matches. All 117 training and 29 test frames loaded with aligned clean targets/masks.
- CPU geometry checks passed for nearest finite-patch selection, reflected direction, proper virtual-camera rotation, conservative tile candidate coverage and rejection of zero-area polygons.
- The project `.conda` environment is complete: Python 3.10.22, PyTorch 2.0.1+cu118, working CUDA and the project-local extension. Package installation reused the existing shared Conda package cache.
- Both phases passed a 100-iteration quarter-resolution scene run in `.conda`; 15,028 interface and 25,321 reflection Gaussians at completion. Phase-loop times: 6.286 s and 13.172 s. Peak PyTorch allocation: 290.286 MiB, excluding driver/context/desktop use. Extracted geometry: coverage 0.7404, precision 0.9011 on sampled training views.
- A separate reflection checkpoint resumed from iteration 40 to 45, including optimizer and deformation state, with finite losses.
- All 29 held-out views rendered successfully; mean GPU rendering time 26.446 ms. Standard PSNR/SSIM/LPIPS, actual glass-pixel metrics and inherited comparison metrics completed and were written to `output/benchmark_r4/results.json`.
- Smoke-run test metrics after only 100 iterations per phase: PSNR 16.4001 dB, SSIM 0.62248, standard-range VGG LPIPS 0.60307. These are dependency/integration checks, not converged research results. No claim of improvement over RT-Splatting is made.
- Full-resolution defaults use a 64-million-entry tile budget. The quick profile explicitly limits it to 12 million. Long-run convergence and the complete 6,000+6,000 quick experiment remain for the user to run.
