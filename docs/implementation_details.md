# VIGOR-GS: implementation details (`f_refactor` branch)

This document describes the current code stage by stage, following the framework (`docs/framework.pdf`, LaTeX source `docs/framework.tex`). Each section identifies where equations are implemented, the data conventions, and where the code chooses an interpretation or adds something beyond the PDF. These choices are collected in Section 12. Ablations are documented separately in `docs/ablations.md`. Changes relative to the `quan_dang` branch are listed in `docs/changes_from_quan_dang.md`.

## 0. Code reading order

| Stage | File | Contents |
|---|---|---|
| Data | `scene/dataset_readers.py`, `scene/cameras.py`, `undistort_sources.py` | Aligned layout, cameras, linear RGB, valid region |
| Gaussian fields | `scene/gaussian_model.py`, `scene/deformation.py` | `G_I^0`, `G_R^0` (Sec. 1), `D_I`, `D_R` (Eq. 1) |
| Phase I | `gaussian_renderer/interface.py`, `utils/losses.py`, `train_interface.py` | Eq. 2–4, 3a |
| Glass surfaces | `utils/glass_geometry.py`, `utils/patch_extraction.py`, `extract_glass.py` | `P`, Eq. 9, 9a, 9b, 10 |
| Phase II | `gaussian_renderer/reflection.py`, `gaussian_renderer/composition.py`, `train_reflection.py` | Eq. 5–8, 11 |
| Inference, evaluation | `render.py`, `metrics.py` | Eq. 12–13 |
| Full pipeline | `train.py`, `run_ablations.py` | Sequential stages, ablation suite |
| Shared | `arguments/__init__.py`, `utils/model_io.py`, `utils/checkpoint_utils.py`, `utils/run_utils.py` | Parameters, model loading, checkpoints, logging |
| CUDA | `submodules/diff-surfel-anych` | 2DGS rasterizer and reflected-ray renderer `render_reflected_rays` |

## 1. Mapping equations to code

| PDF | Meaning | Code |
|---|---|---|
| Sec. 1, `G_I^0`, `G_R^0` | 2D disks `(μ, q, s⁽²⁾, p, o, a)`, 3D ellipsoids `(μ, q, s⁽³⁾, o, a)` | `InterfaceField`, `ReflectionField` in `scene/gaussian_model.py` |
| Eq. 1 | `G(t) = D_φ(G^0, t)`, p and o independent of t | `Deformation.forward` (`scene/deformation.py`) |
| Sec. 1, `P = {(n_k, d_k, S_k)}` | Fixed, finite glass patches | `validate_patches`, `glass_geometry.json` |
| Eq. 2 | `B_t = R_2D(G_I(t), C_t)` with opacity `p·o` | `render_interface` (direct pass) |
| Eq. 3 | `L_I = (1/|Ω|) Σ ‖B − I^clean‖₁ + λ_s L_surf` | `pixel_l1`, `interface_losses` |
| Eq. 3a | `L_surf = (1/|Ω_s|) Σ ‖N − N^depth‖₁` | `surface_loss`, `out['surface_valid']` |
| Eq. 4 | Optimize `{G_I^0, φ_I}` | `train_interface.training` |
| Eq. 5 | Virtual camera `V_dyn` and renderer `R_3D^refl` | `dynamic_virtual_camera`, `ray_bundle`, `render_reflection` |
| Eq. 6, 13 | `Î = (1 − α) B + α R` (linear RGB) | `render_composite` |
| Eq. 7 | `L_R = (1/|Ω|) Σ ‖Î − I‖₁ + λ_m L_motion + λ_o L_opt` | `reflection_losses` |
| Eq. 7a–7b | Compare predicted and observed changes between adjacent frames | `motion_loss`, `previous_frames` |
| Eq. 7c | `L_opt = (1/|Ω|) Σ h_ψ²` | `optical_loss` |
| Eq. 8 | Optimize `{G_R^0, φ_R}` and ψ, with Θ_I fixed | `train_reflection.training` |
| Eq. 9 | `Π_t(u) = FirstHit(...)`, `x_t(u)` | `first_hit` (with occlusion checks, Section 6.3) |
| Eq. 9a | `r = d − 2(nᵀd) n` | `reflect`, `first_hit()['reflected']` |
| Eq. 9b | `c_v = x(u_ref) + ε r(u_ref)`, `Q_v = LookAt(r, h)` | `dynamic_virtual_camera`, `look_at` |
| Eq. 10 | `cos θ = |nᵀd|`, `F0`, Schlick | `first_hit()['cosine']`, `schlick` |
| Eq. 11 | `α = clip(F + h_ψ)` on glass, 0 elsewhere | `reflection_weight` |
| Eq. 12 | Inference with fixed parameters | `render.render_sets` |

## 2. Data

### 2.1. Scene layout

```
<scene>/colmap/images/<name>          original images I_t, undistorted to COLMAP's PINHOLE camera
<scene>/colmap/sparse/0/              COLMAP model (bin or txt): cameras, poses, SfM points
<scene>/colmap/frames.json            per frame: time (normalized to [0, 1]), split, source_clip (sequence)
<scene>/colmap/split.json             optional, used when a frame has no split
<scene>/colmap/masks/<stem>.png       glass mask M_t, aligned with colmap/images
<scene>/colmap/<clean_images>/        aligned clean images I_t^clean: <split>/images/<stem>.* or <stem>.*
<scene>/colmap/source_warp.json       optional: source-image distortion model (written by undistort_sources.py)
```

Masks and clean images are created on the distorted source images, so they must be warped to the COLMAP grid **once per scene** using `undistort_sources.py` (taken from `viet_huynh`). The reader performs no warping; it only resizes according to `-r`. VSGD 0758, WayveScene101 scene_001, and 3D-FRONT-T scene_1 in `data/` all already use this layout.

- A training frame without a clean image is an error unless `--ignore_clean_images` is enabled (the no-separation ablation). Test frames may lack clean images: they are evaluated only against the original images.
- Images in the COLMAP model that are absent from `frames.json` (e.g., other cameras in a rig) are skipped, and their count is printed.
- Only PINHOLE / SIMPLE_PINHOLE cameras are accepted. Distorted cameras are rejected with instructions to run `undistort_sources.py`.
- `center` and `extent` follow 3DGS's `getNerfppNorm`: the center is the mean of the training camera centers, and `extent` is 1.1 times the maximum distance from that center.

### 2.2. Valid region Ω

Undistorting a wide-angle image leaves pixels with no corresponding source pixel (black borders). When `source_warp.json` is present, the reader reconstructs the `cv2.remap` map (`utils/undistort_utils.build_remap`). Pixels whose source coordinates fall outside the source image are marked invalid. Both file formats are supported: the old format with a top-level `model` and a single focal length `f`, and the new format with a per-camera `model` and `fx`, `fy`. When downsampling, a pixel is valid only if its entire area is valid. Ω in every loss and metric is this set of valid pixels. For the 3 existing scenes, Ω covers the whole image.

### 2.3. Cameras and color

`scene/cameras.py` decodes pixels once, at the training resolution. Images are stored as `uint8` and masks as `bool` on `data_device` (RAM by default). The following properties return CUDA tensors:

- `original_linear`, `clean_linear`: images in **linear RGB**, since all fitting and blending in the framework use linear RGB (Eq. 3, 6). Standard sRGB decoding is implemented in `utils/color_utils.srgb2linear`.
- `original_srgb`: images for saving GT and computing metrics.
- `glass_mask` (M_t), `valid_mask` (Ω).

Intrinsics are scaled according to `-r`, and the principal point follows COLMAP's convention (pixel centers at +0.5). The projection matrix supports an off-center principal point: `P[0,2] = 2cx/W − 1`. The test `test_camera_scales_intrinsics_and_decodes_linear_rgb` verifies that the rasterizer and the rays in `utils/point_utils.camera_rays` use the same pixel-center convention.

## 3. The two Gaussian fields

`GaussianField` (`scene/gaussian_model.py`) contains the parts independent of primitive type: parameter storage, optimizer state adjustments during clone/split/prune (Adam moments are retained for existing points; new points start at 0), the 3DGS densification schedule, `save_ply`, and `capture`/`restore`. Per-Gaussian parameters are listed once in `PARAMS`.

| | `InterfaceField` (G_I) | `ReflectionField` (G_R) |
|---|---|---|
| Primitive | 2D disk (`SCALE_DIM = 2`) | 3D ellipsoid (`SCALE_DIM = 3`) |
| Parameters | xyz, SH, opacity o, scaling, rotation, **occupancy p** | xyz, SH, opacity o, scaling, rotation |
| Rendering opacity | `p · o` (direct pass) | `o` |
| Initialization | SfM points, colors converted to linear; p = 0.1, o = 0.5, random quaternions, scale = 0.1 × neighbor distance | Points along reflected rays (Section 7.6) |
| "Presence" for pruning/reset | p (o without factorization) | o |
| Point limit | `--max_interface_points` 500k | `--max_reflection_points` 250k |

Occupancy, opacity, and color are canonical parameters, so they are independent of time (Sec. 1). `factorized=False` (`--ablate_factorization`) removes parameter p entirely (p ≡ 1).

**Densification schedule** (following 3DGS/2DGS):

- From `densify_from_iter` to `densify_until_iter`, every `densification_interval` iterations: clone small Gaussians and split large Gaussians whose average gradient exceeds the threshold. When the point limit would be exceeded, only Gaussians with the strongest gradients are replicated.
- Prune when presence < `min_presence` (0.005).
- Every `presence_reset_interval` (3000) iterations, clamp presence to at most 0.01, as in 3DGS's opacity reset.
- After the first reset, prune Gaussians larger than 10% of the extent; for G_I, also prune disks whose screen-space radius exceeds `max_screen_size` (20 px). Without this step, a few huge Gaussians in large scenes (Wayve) cover every tile and inflate the rasterizer buffer.

Gradients used for densification:
- G_I: the norm of the screen-space coordinate gradient (`viewspace_points.grad[:, :2]`), as in 2DGS.
- G_R: the ray-based renderer has no screen-space coordinates, so it uses the norm of the 3D gradient of the deformed center multiplied by the extent (to make the threshold independent of scene scale), with a default threshold of 2e-5.

## 4. Deformation (Eq. 1)

`scene/deformation.py`, following Yang et al. (Deformable 3D Gaussians):

```
(Δμ, Δq, Δs) = MLP(γ(μ̂), γ(t))
μ' = μ + extent · Δμ,   q' = normalize(q + Δq),   s' = exp(log s + Δs)
```

- `μ̂ = (μ − center) / extent`. γ is sin/cos positional encoding with 2^k·π: `deform_xyz_freqs` = 6, `deform_time_freqs` = 6.
- MLP: `deform_depth` layers × `deform_width` (default 4 × 128), ReLU. The output layer is initialized to 0, so a new network is the identity function.
- Canonical centers are detached at the input, following Yang et al.: gradients still reach μ through `μ + Δμ` but do not propagate backward through the encoding, separating the canonical field from deformation. The test `test_deformation_starts_as_identity_and_static_never_moves` checks this.
- The two fields use separate instances (same class, no shared weights). Only geometry changes with t.
- The initial `warm_up` (1000 iterations) of each phase uses only the canonical field.
- The learning rate decays exponentially to 1% by the end of the phase.
- `static=True` (`--static_interface`, `--static_reflection`) bypasses the network: the field stays fixed over time.

## 5. Phase I (Eq. 2–4)

### 5.1. Renderer (`gaussian_renderer/interface.py`)

The `diff_surfel_anych` 2DGS rasterizer is built in place within this branch's submodule. `gaussian_renderer/__init__.py` prepends that directory to `sys.path`, so the environment's installed copy of `diff_surfel_anych` (from RT-Splatting) is not replaced.

- **Direct pass:** disks are rendered with `p·o`, producing `B_t`, depth (expected, or median according to `depth_ratio`), alpha, rendered normals `N_t` (converted to world coordinates), and depth-derived normals `N_t^depth` (the cross product of row and column differences of back-projected 3D points). `Ω_s` in Eq. 3a consists of pixels with alpha > 0.1, whose entire 3×3 neighborhood is also covered, and where both normals are defined.
- **Surface pass** (only when needed): the same disks, rendered with opacity = p, carrying o as an auxiliary channel. `glass_map = surface_alpha − Σ w_i o_i = Σ w_i (1 − o_i)`: high where a surface is present (high p) but transmits light (low o). Used only for the mask prior and patch extraction; never for blending (Sec. 1: "There is no learned image-space plane mask").
- **`DUAL_VISIABLE 1`** (`cuda_rasterizer/auxiliary.h`): each disk's normal is flipped toward the camera, as in the original 2DGS [1]. RT-Splatting sets this to 0. With that setting, rendered normals face the camera at only about 41% of pixels, while depth-derived normals always face the camera, and the L1 L_surf term (Eq. 3a) penalizes each backward-facing disk by about 2 units.

### 5.2. Loss (`utils/losses.py`)

Pixel normalization follows the PDF's "Reading the losses" section: `‖·‖₁` sums the three channels of a pixel, then divides by the number of valid pixels `|Ω|`.

```
L_I = pixel_l1(B, I^clean, Ω) + λ_s · L_surf(N, N^depth, Ω_s ∩ Ω) [+ λ_glass · BCE(glass_map, M_t)] [+ λ_dssim · D-SSIM]
```

| Component | Default | Notes |
|---|---|---|
| `λ_s` (`--lambda_surface`) | 0.05, starting at iteration `surface_from_iter` = 1000 | Eq. 3a; contributes 0 if Ω_s is empty |
| `λ_glass` (`--lambda_glass`) | 0.05 | **Not in Eq. 3.** See Section 5.3 |
| `λ_dssim` | 0 | Optional; the PDF omits SSIM |

### 5.3. Why a glass-mask prior is used

Glass is invisible in clean images, so Eq. 3 alone provides no signal indicating where it is. The PDF's Scope section also states that p and o cannot be disentangled without a prior. Without known glass geometry, `S_k` must come "from Phase I geometry" (Sec. 1). The BCE prior between `glass_map` and the training masks gives G_I a layer of "present but transparent" disks at the glass locations, which `extract_glass.py` can sample. The prior uses only **training** masks. Test masks are never used except to evaluate glass-region metrics. With known glass geometry, `--lambda_glass 0` can be used.

### 5.4. Training loop and output

- Every training view is sampled once per epoch, with a new random order each epoch (`ViewSampler`).
- The SH degree increases by one every 1000 iterations. The xyz learning rate decays exponentially, as in 3DGS.
- Every `test_interval` iterations: evaluate PSNR of B against the target (linear, on Ω) and IoU of `glass_map > 0.5` against the mask on the test set.
- Output: `<model>/config.json` (`ModelParams`, read by all subsequent stages); `<model>/interface/{config.json, latest.pth, training.jsonl, point_cloud.ply}`.
- `--resume` continues from the checkpoint. Without `--resume`, existing checkpoints are not overwritten.

## 6. Glass surfaces (Sec. 1, Eq. 9)

### 6.1. Format

`<model>/glass_geometry.json`:

```json
{"patches": [{"normal": [nx, ny, nz], "offset": d, "vertices": [[x, y, z], ...]}],
 "settings": {"occlusion_tolerance": 0.02, "occlusion_alpha": 0.5, "ablate_occlusion": false, "glass_margin": 0.043},
 "quality": {"coverage": 0.93, "precision": 0.88, "views": 12, "samples": 9000},
 "source": "phase_i_iteration_30000",
 "accepted": true}
```

The plane is `nᵀx + d = 0`. `vertices` is an ordered convex polygon lying on the plane. `validate_patches` normalizes n and rejects normals equal to 0, non-coplanar vertices, polygons with area 0, and non-convex polygons. Known geometry is supplied with `--glass_geometry <file>` (a patch list, or a dict with a `patches` key).

### 6.2. Extraction from Phase I (`utils/patch_extraction.py`)

1. **Sampling.** Across about 12 training views, the surface pass of frozen G_I gives the depth of the first "present" surface. Pixels inside the training glass mask with both `surface_alpha > 0.3` and `glass_map > 0.1` are back-projected into 3D (up to `plane_max_points` = 12000 points).
2. **Planes.** Run sequential RANSAC (512 hypotheses, threshold `plane_threshold × extent`), refine using SVD on the inliers, remove the inliers, and repeat for up to `max_patches` (4) planes.
3. **Patches.** Split coplanar inliers into connected components: union-find on a radius graph with a radius of 8 times the median neighbor distance, in plane coordinates. Each sufficiently large component becomes a convex hull. This produces separate patches for disconnected panes on the same wall.
4. **Orientation.** Flip normals toward the side containing the majority of cameras. This sign does not affect the equations (`cos θ` uses an absolute value, and reflection is unchanged when n changes sign), but makes the file easier to read.
5. **Quality.** Intersect the patches with rays from the sampled views (with occlusion checks), then compare against the masks: `coverage = |hit ∩ mask| / |mask|`, `precision = |hit ∩ mask| / |hit|`. Below `plane_min_coverage` or `plane_min_precision` (0.5), the file is written with `"accepted": false`, `extract_glass.py` exits with an error, and Phase II refuses to run.

`<model>/glass_preview/<view>.png`: green indicates mask pixels not covered by any patch, red indicates hits outside the mask, yellow indicates hits inside the mask, and blue indicates occluded hits.

Limitations: clean images and silhouettes do not uniquely determine the depth of transparent glass (PDF Scope). Therefore, residuals and agreement with the mask are only checks, not ground truth. A convex hull only approximates the boundary. Glass with holes or curved glass requires known geometry.

### 6.3. FirstHit and occlusion checks (Eq. 9)

`first_hit(view, patches, epsilon, occlusion)` performs the following steps for every pixel:

1. Cast a camera ray (unit direction d), intersect it with each plane, and keep only intersections at a distance > ε inside polygon `S_k`. Select the nearest intersection, i.e., "the nearest hit if projections overlap".
2. **Occlusion check** (you selected option A). Eq. 9 considers only patches. A glass pane behind an opaque object (a person in front of a glass door) is still selected, causing reflections to be blended over that object. Therefore, a hit is rejected when the **occlusion pass** of frozen G_I is **opaque** at that pixel (`alpha ≥ occlusion_alpha` = 0.5) **and** its depth is **in front of** the glass: `depth < z_hit · (1 − occlusion_tolerance)`, with relative `occlusion_tolerance` = 0.02.
   - The **occlusion pass** (`utils/model_io.DirectPass`) is G_I's direct pass after **excluding disks lying on the glass patches themselves**: disks within `glass_margin` (= the RANSAC threshold `plane_threshold × extent`) of a patch plane whose projections lie inside the polygon (`on_patches`). These disks form the glass layer. When clean images retain residual reflections (DAI), Phase I "paints" that residual onto this layer, making it slightly opaque. If this layer is included, even a plane-fitting error can place its depth in front of the patch, falsely marking real glass as occluded. The trial run on 3D-FRONT-T scene_1 showed exactly this behavior (Section 13). B used for blending remains the full direct pass.
   - At unoccluded glass pixels, the occlusion-pass depth belongs to whatever lies **behind** the glass, so the hit is retained. A surface at or behind the glass is never treated as an occluder.
   - This uses only reconstructed geometry: no learned mask and no new parameters.
   - `--ablate_occlusion` (selected in `extract_glass.py`, saved in `settings`) restores Eq. 9 verbatim.
3. Return (H, W) maps: `patch` (int32, −1 means no glass), `visible`, `occluded`, `distance`, `cosine = |nᵀd|` (Eq. 10), and (H, W, 3) maps: `hits` x_t(u), `normals`, `real_directions` d, `reflected` r (Eq. 9a).

This rule is applied **identically** during patch extraction (quality measurement), Phase II initialization and training, and rendering.

## 7. Phase II (Eq. 5–8, 11)

### 7.1. Reflected rays and dynamic virtual camera V_dyn

For a glass pixel: `r = d − 2(nᵀd) n` (Eq. 9a). The physically correct ray is `x_t(u) + s·r_t(u)`, `s ≥ 0` (Sec. 3, "Render"). It leaves the glass back toward **the same side as the real camera**. G_R lives on that side, in ordinary world coordinates, at the true positions of the reflected objects. It is never mirrored across the glass.

`dynamic_virtual_camera` implements Eq. 9b for each visible patch:
- `u_ref` is the region's pixel nearest its mean pixel position.
- `c_v = x(u_ref) + ε r(u_ref)`: the virtual camera sits on the glass, offset by ε along the reflected direction, so it is on the same side as the real camera and G_R.
- `Q_v = LookAt(r(u_ref), h_t)`: a proper rotation (det = +1) with x pointing right, y down, and z along r, i.e., the virtual camera looks away from the glass, toward G_R. The "up" direction comes from the real camera (the real camera's −y axis in world coordinates).
- This is a single camera: for each region/patch, the function resets the pose of the same camera, rather than storing a separate camera for each patch.

A geometric property used for binning (Section 7.4), **not** a camera placement: within a planar patch, all reflected rays, when extended **backward** through the glass, meet at one point, `c'_k = mirror_k(c)`, the real camera center mirrored across the patch plane (`mirrored_center`, test `test_dynamic_virtual_camera_pose_and_mirrored_center`). `c'_k` lies behind the glass and no ray is ever cast from it. It only means that a patch's exact ray bundle has the directions of a central projection from `c'_k`, while the rays themselves start at the glass surface and travel toward G_R. The union of rays across multiple patches no longer shares such a point, as stated in the PDF ("non-central rays").

### 7.2. Three ray modes (`ray_bundle`)

| Mode | Ray origin | Direction | Purpose |
|---|---|---|---|
| `exact` (default) | `x_t(u) + ε r` | `r_t(u)` | Framework renderer |
| `pinhole` (`--pinhole_virtual_camera`) | The region's `c_v` (Eq. 9b) | `r_t(u)` | The approximation mentioned in the PDF: "using that pose alone in an ordinary pinhole rasterizer is generally only an approximation". Large parallax errors for objects near the glass |
| `unreflected` (`--ablate_virtual_camera`) | `x_t(u) + ε d` | `d_t(u)` | No virtual camera: rays are not reflected and continue straight through the glass, so G_R is seen behind the glass instead of on the camera side |

In `exact` and `pinhole` modes the rays travel from the glass toward the camera side, where G_R is. The test `test_a_gaussian_in_front_of_the_glass_appears_at_its_mirror_image` places a small Gaussian of G_R in front of the glass (on the camera side) and leaves it there. With `exact` rays, its reflection appears at the brightest pixel, which matches, with error ≤ 1 px, the pixel where the real camera would see the Gaussian's mirror image. This is where a real reflection appears. With `pinhole`, it is displaced by more than 2 px.

### 7.3. Ray-based renderer (CUDA, from `quan_dang`)

`diff_surfel_anych.render_reflected_rays` (`rasterize_points.cu`):
- For each pixel and each candidate Gaussian in the tile, find the point on the ray closest to the Gaussian center under the Gaussian's metric (`A = R / s`). Discard the Gaussian if that point lies behind the ray origin (`s ≤ 0`, following the PDF's `s ≥ 0` condition) or outside `ray_cutoff` (3.5σ).
- `α = min(0.99, o · exp(−q/2))`, composite front to back, stop when T < 1e-4.
- The backward pass computes gradients with respect to centers, inverse-axis matrices (then propagated by PyTorch to rotation, scale, and deformation), opacity, and color. Tests compare the forward pass and gradients against an independently written PyTorch reference.
- Approximations (as in 3DGS): compositing order follows Gaussian-center depth within the tile (the real-camera depth of `mirror_k(μ)`, Section 7.4, which approximates the order along the reflected rays), and Gaussians are evaluated at the closest point rather than through volume integration. Refraction, multiple reflections, and multiple glass intersections are not modeled.

### 7.4. Tile candidate lists (`tile_candidates`)

To build tile lists, the code must know which pixel's ray passes through a Gaussian center P. P is a point of G_R, on the camera side of the glass. Mirroring is used **only to find that pixel**: neither G_R nor the virtual camera is moved. The pixel is the real-camera projection of:
- `mirror_k(P)` in `exact` mode. Mirroring the reflected ray `x_t(u) + s·r` across the patch plane gives `x_t(u) + s·d`, the real camera ray of the same pixel continued through the glass. So the reflected ray of pixel u passes through P exactly when the real ray of u passes through `mirror_k(P)`, which lies behind the glass. This is the same fact as the rays meeting at `c'_k = mirror_k(c)` when extended backward (Section 7.1);
- `mirror_k(P + c'_k − c_v)` in `pinhole` mode (a pinhole ray is an exact ray translated by `c_v − c'_k`);
- `P` in `unreflected` mode (no reflection).

The sort depth is the real-camera depth of the same mapped point. In `exact` mode, this approximates the distance along the reflected rays from the glass toward G_R.

Each Gaussian is bounded by a sphere of radius `ray_cutoff × max scale`. Its conservative projection gives a tile range, retaining only tiles where that patch actually appears. Entries `k·N + i` are sorted by (tile, depth) using a 64-bit key. Exceeding `--max_tile_entries` total (Gaussian, tile) pairs raises an explicit error rather than silently dropping Gaussians. Tests verify that, in all three modes, every Gaussian intersected by a ray appears in the tile's candidate list.

Color is each Gaussian's SH evaluated along the ray direction toward it, `normalize(μ − patch ray center)`. A Gaussian seen through two patches therefore has two view-dependent colors. Linear radiance is nonnegative and may exceed 1 before blending.

### 7.5. Reflection weight and blending (`gaussian_renderer/composition.py`)

```
F0 = ((η_a − η_g)/(η_a + η_g))²,   F(θ) = F0 + (1 − F0)(1 − cos θ)⁵        (η_a = 1.0, η_g = 1.5, so F0 = 0.04)
α = clip_[0,1](F(θ) + h_ψ) on selected glass pixels, 0 elsewhere
Î = (1 − α) B + α R
```

**h_ψ** (`OpticalCorrection`, enabled by default according to Eq. 7–8; `--pure_fresnel` disables it):
- Inputs: `γ(x̂_t(u))` (the glass point intersected by the pixel's ray, normalized by center/extent, 4 frequencies), `γ(t)` (2 frequencies), and `cos θ`. MLP with 2 hidden layers × 64, output layer initialized to 0, so it starts as pure Fresnel. Evaluated only on glass pixels; h = 0 outside glass.
- The PDF writes `h_ψ(u, t)`. Given the camera, `x_t(u)` is a function of u, so this is still `h_ψ(u, t)`. Unlike raw pixel coordinates (`quan_dang` used normalized (u, v)), `x_t(u)` identifies **the same location on the glass** across views, allowing corrections learned on training views to transfer to test views.
- `--optical_bound` defaults to 0: unbounded, with only Eq. 11's `clip[0,1]` and `L_opt`. A value > 0 applies `bound · tanh(h / bound)`. `quan_dang` imposed a fixed ±0.1 bound.

`--ablate_fresnel`: replace F(θ) with a learned constant `σ(c)` shared by all glass pixels, initialized to F0. If h_ψ is enabled, it is added to that constant.

### 7.6. G_R initialization

Across about 12 training views, randomly sample valid glass pixels and place a point at `ray origin + s · ray direction`, with `s ∈ extent · [0.1, 3.1]` (using the active ray mode), and color `(I − (1 − F) B) / F`, clamped to [0, 20]. This is the radiance that could explain the residual on its own. Opacity is 0.1, and scale is based on the nearest neighbor.

### 7.7. Loss and training loop (`train_reflection.py`)

```
L_R = pixel_l1(Î, I, Ω) + λ_m · motion_loss(Î_k, Î_{k−1}, I_k, I_{k−1}, Ω_k ∩ Ω_{k−1}) + λ_o · optical_loss(h, Ω)
```

- **L_motion (Eq. 7a–7b):** frame k−1 is the immediately preceding frame in time from **the same sequence (`source_clip`) and the same camera (`sensor_id`)** (`previous_frames`). VSGD 0758 has two clips, so the first frame of each clip has no predecessor. Both predictions are rendered, and both receive gradients. This constrains image changes rather than penalizing deformation magnitude. Default `λ_m` = 0.1.
- **L_opt (Eq. 7c):** mean h² over Ω (h = 0 outside glass), `λ_o` = 0.01, only when h_ψ is present.
- G_I, D_I, and the patches are completely fixed. `DirectPass` renders B_t and the occlusion pass (depth, alpha) once per view and caches them in RAM (LRU, `--cache_views` = 32), since Phase II never changes them. Clean images are not loaded in Phase II.
- Optimizers: Adam for G_R (3DGS learning rates), for D_R (`deform_lr`, exponential decay), and for h_ψ / the constant (`optical_lr` = 1e-3).
- Every `test_interval` iterations, evaluate composite and direct PSNR on the test set, over the whole image and the glass region (linear, Ω). The difference between the two PSNR values indicates the contribution explained by reflections.
- Output: `<model>/reflection/{config.json, latest.pth, training.jsonl, point_cloud.ply}`.

## 8. Inference (`render.py`, Eq. 12–13)

- Load Phase I, Phase II (configuration from `reflection/config.json`), and `glass_geometry.json`. Each pixel selects the first visible glass patch using the same rule as training. **Test masks are never used for blending.**
- Output: `<model>/<split>/ours_<iter>/{renders, gt, glass_masks, valid_masks}`.
- `--components` additionally saves: `direct/`, `reflection/`, `alpha/`, `patches/` (colored by selected patch), `clean/` (the view's clean image, for layer-separation metrics), `virtual_cameras.json` (V_dyn poses for each visible patch in each frame).
- `--direct_only` (automatically enabled for the no-separation ablation) renders `direct_<iter>`.
- `comparison.mp4`: ours on the left, GT on the right, ordered by sequence, camera, and time.
- `render_stats.json`: per-frame rendering time measured using CUDA events, including the direct pass, FirstHit, ray renderer, and blending; excluding data loading and file writing.

## 9. Metrics (`metrics.py`)

Evaluated on sRGB PNGs, one view at a time:

| Metric | Definition |
|---|---|
| PSNR, SSIM, LPIPS | Whole image against the original image (LPIPS VGG, input [−1, 1]) |
| Valid PSNR | Only on Ω |
| Glass PSNR / Opaque PSNR | Annotated glass / non-glass pixels within Ω, divided by the corresponding pixel count |
| Masked PSNR/SSIM/LPIPS (RT) | Multiply both images by the mask, then average over the whole image, following RT-Splatting's protocol (background values of 0 inflate scores) |
| LPIPS (RT input) | LPIPS with input [0, 1], RT-Splatting's convention |
| **Transmission PSNR**, Glass Transmission PSNR | B against the clean image (requires `--components`): how well the non-reflective layer is recovered |
| FPS, Training Time | From `render_stats.json` and `<phase>/training.jsonl` (summed across resumed sessions) |

Transmission PSNR measures layer separation itself. PSNR against the original images favors methods that directly fit those images (e.g., RT-Splatting or the `no_separation` ablation), so it is insufficient on its own to evaluate VIGOR-GS.

## 10. Configuration, checkpoints, and resume

- `arguments/__init__.py`: `ModelParams` (scene, resolution, architecture, Phase I ablations), `PipelineParams`, `InterfaceOptimizationParams`, `GlassParams`, `ReflectionParams` (Phase II model and ablations), `ReflectionOptimizationParams`.
- Phase I writes `ModelParams` to `<model>/config.json`. Subsequent stages parse `ModelParams` with `sentinel=True`, and `load_model_args` overrides only explicitly supplied arguments. Changing the Phase I architecture (SH, deformation, Phase I ablations) in a later stage is an error.
- Checkpoints (`utils/checkpoint_utils.py`, version 2) include: the field, its optimizer and moments, deformation and its optimizer, auxiliary modules (h_ψ, the Fresnel constant) and their optimizers, and RNG state. Files are written to a temporary path, then committed with `os.replace`. Resume continues bit for bit (`test_checkpoint_resume_continues_exactly`).
- Without `--resume`, an existing checkpoint is never overwritten.

## 11. Commands

Build the extension (once, and after each CUDA or header change):

```bash
cd submodules/diff-surfel-anych
rm -rf build && CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

You must run `rm -rf build` when changing only a header (e.g., `auxiliary.h`): ninja does not automatically rebuild after header changes.

Run the full pipeline:

```bash
python train.py -s data/datasets/datasets/3D-FRONT-T/scene_1 -m output/3dfront_full -r 2
python train.py -s <scene> -m <model> --glass_geometry patches.json        # known glass geometry
python train.py -s <scene> -m <model> --reflection_args "--iterations 20000 --lambda_motion 0.05"
```

Individual stages:

```bash
python train_interface.py -s <scene> -m <model> -r 2
python extract_glass.py -m <model>                  # or --glass_geometry patches.json
python train_reflection.py -m <model>
python render.py -m <model> --skip_train --components
python metrics.py -m <model>
```

Ablation suite: `python run_ablations.py -s <scene> --root output/<scene>_ablations` (resolution chosen per dataset: `-r 1` for VSGD and 3D-FRONT-T, `-r 2` for WayveScene101; see `docs/ablations.md`).

Tests: `python -m pytest`, including 50 tests using the synthetic scene in `tests/synthetic.py`. GPU tests are skipped when CUDA is unavailable. Tests with short training runs are marked `slow`.

## 12. Interpretations and additions relative to the PDF

| Area | PDF | Code | Reason |
|---|---|---|---|
| Occlusion in FirstHit | Eq. 9 considers only patches | Reject hits when frozen G_I depth (excluding the disk layer on the glass) is in front of the glass | You selected option A; `no_occlusion` ablation |
| Glass-mask prior in Phase I | Eq. 3 contains only L1 + λ_s L_surf | Add λ_glass · BCE(glass_map, M_t) | Glass geometry is needed to extract `S_k` from Phase I (Sec. 1, Scope). Can be disabled with known geometry |
| Source of `S_k` | "known geometry or Phase I interface geometry" | Both: `--glass_geometry` or RANSAC + convex hulls from the surface pass, with quality thresholds | |
| Renderer `R_3D^refl` | Non-central rays `x + s r` | Ray-based CUDA kernel (closest approach), ordered by center depth | Approximation as in 3DGS, Section 7.3 |
| Role of V_dyn (Eq. 9b) | Pose describing the viewing direction; using this pose in a pinhole renderer is only an approximation | Computed per region/frame, exported to `virtual_cameras.json`; used as the ray origin in the `pinhole_virtual_camera` ablation | The main renderer uses exact rays |
| Inputs to h_ψ(u, t) | Function of pixel and time | `γ(x_t(u))`, `γ(t)`, `cos θ` | Applies corrections to the correct glass location in new views |
| Bounding h_ψ | None, only clip[0,1] | Unbounded by default; `--optical_bound` enables a bound | |
| D-SSIM | Omitted | `--lambda_dssim` defaults to 0 | |
| Ordering in L_motion | "neighboring timestamps", "ordered training sequence" | Immediately preceding frame with the same `source_clip` and `sensor_id` | Different clips are not continuous |
| 2DGS normals | 2DGS [1] | `DUAL_VISIABLE 1` | RT-Splatting disables this flag, causing the L1 term in Eq. 3a to penalize backward-facing disks |
| Densification, reset, pruning | Not specified | Following 3DGS/2DGS (Section 3) | Stabilizes point count and memory use |
| G_R initialization | Not specified | Points along rays, colors from residuals (Section 7.6) | |
| Valid region Ω | "evaluated image pixels" | Pixels with a corresponding source pixel after undistortion | |

## 13. Limitations and open issues

- In scenes with only pseudo-GT clean images (DAI), B is constrained by those images, and G_I is frozen in Phase II as required by the framework. Earlier analysis on VSGD 0758 identified this as the main reason PSNR against the original images is lower than RT-Splatting. Transmission PSNR should also be reported.
- Glass geometry extracted from Phase I depends on the mask prior and Phase I quality. The 0.5/0.5 quality thresholds only reject broken geometry; they do not measure accuracy.
- **Performance.** Tile candidate lists are built using PyTorch (`tile_candidates`), while the per-pixel CUDA kernel traverses tile entries directly from global memory (without shared memory as in 3DGS). These two parts account for most of Phase II and rendering time. Memory scales with the number of (Gaussian, tile) pairs, bounded by `--max_tile_entries`. Optimization directions: move binning to CUDA (duplicate keys + radix sort, as in 3DGS) and load each tile's Gaussians into shared memory.
- Refraction, nested reflections, and multiple glass intersections are not modeled (PDF Scope).
- The RT-Splatting baseline is not included in this branch (the `quan_dang` branch replaced RT's `train.py`). Run it from `viet_huynh` or `main`.

## 14. Trial run on real data

3D-FRONT-T scene_1, `-r 4` (240×135), DAI clean images, RTX 3050 Laptop 4 GB (GPU thermally throttled to 240–712 MHz during the run):

```bash
python train.py -s data/datasets/datasets/3D-FRONT-T/scene_1 -m output/smoke_3dfront_r4 -r 4 --iterations 1500 --skip_lpips     --reflection_args "--iterations 300 --test_interval 300 --checkpoint_interval 300 --densify_until_iter 250"
```

| Stage | Result |
|---|---|
| Phase I, 1500 iterations | 2 minutes, peak VRAM 577 MB; test PSNR of B against clean images 26.7 dB (linear); glass-map IoU 0.75 |
| Glass extraction | 4 patches for the glass box with multiple faces, coverage 0.72, precision 0.62 (accepted) |
| Occlusion | Excluding the disk layer on the glass from the occlusion pass reduced the false-positive rate within the mask from 3.9% to 2.2% (Section 6.3) |
| Phase II, 300 iterations | 6.5 minutes; test glass-region PSNR (linear) 25.95 dB for direct, 27.63 dB for composite |
| Rendering | About 0.86 s/frame while the GPU was throttled (Section 13) |

This is only an integration check (very few iterations, low resolution), not a result suitable for reporting.

