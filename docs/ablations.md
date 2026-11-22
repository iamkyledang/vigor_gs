# VIGOR-GS: ablation suite

Each ablation removes or replaces **one** component of the framework (`docs/framework.pdf`) and answers one question: is that component necessary, and what does it contribute? The **main suite** has five ablations (compared against `full`, which is trained separately), one per claim of the framework. Further variants are kept for the appendix and run only when named.

All variants run with `run_ablations.py`. Any ablation that changes only Phase II or the glass-selection rule **copies Phase I from `full`**, so its difference from `full` comes solely from that change. An ablation that changes Phase I trains its own Phase I.

```bash
python run_ablations.py -s data/datasets/datasets/3D-FRONT-T/scene_1 --root output/3dfront_ablations      # the main suite, -r 1
python run_ablations.py -s data/datasets/datasets/WayveScene101/scene_001 --root output/wayve_ablations   # the main suite, -r 2
python run_ablations.py -s <scene> --root <runs> --variants no_occlusion,single_patch                    # named variants only
```

**Default suite.** Without `--variants`, `run_ablations.py` runs `MAIN_SUITE`: `no_separation`, `no_factorization`, `no_virtual_camera`, `pure_fresnel`, `static_reflection`. `full` is **not** rerun. The ablations that copy Phase I from `full` read it from `<root>/full/interface`, so place (or symlink) your existing `full` run at `<root>/full`. If that folder has no Phase I, only its Phase I is trained there. When `<root>/full/results.json` exists, it is the reference row of the summary even though `full` was not run. Use `--variants full,...` to run it as well.

**Resolution.** Unless you pass `-r` / `--resolution`, it is chosen from the scene path: **`-r 1` for VSGD and 3D-FRONT-T, `-r 2` for WayveScene101** (`SCENE_RESOLUTIONS` in `run_ablations.py`). Every variant of a scene gets the same value, and it must equal the resolution of `<root>/full`, because Phase I is copied from it. Passing `-r` explicitly overrides the default for all variants.

**Tile budget.** Unless you pass `--max_tile_entries`, every variant gets `--max_tile_entries 2000000000` (`MAX_TILE_ENTRIES`). The default cap of 64M (Gaussian, tile) pairs in the reflected-ray renderer is exceeded in Phase II on all three scenes, and raising it does not change any result (it only removes a memory-safety error). `render.py` reads the value back from `reflection/config.json`. Pass your own value to override it, for example on a smaller GPU.

Results are saved to `<root>/summary.csv`: PSNR, SSIM, LPIPS, Glass PSNR, Opaque PSNR, Transmission PSNR, Glass Transmission PSNR, FPS, training time, and PSNR / Glass PSNR differences relative to `full`.

## Which metrics to read

- **Glass PSNR** (against the original image, on glass pixels only) measures reflection reconstruction. Phase II ablations affect only this metric, since α = 0 outside the glass.
- **Transmission PSNR** (B against the clean image) measures the quality of the non-reflective layer, i.e., the layer separation itself. This metric distinguishes VIGOR-GS from methods that only fit the original images.
- Whole-image **PSNR** against the original image is needed, but it consistently favors variants that fit the original images directly (`no_separation`, RT-Splatting). It should not be used alone to draw conclusions about layer separation.
- Read Glass PSNR on **test views**. Several ablations can still fit the training views; their failure shows up as wrong parallax or wrong time dependence on held-out views.

## Main suite

| Variant | Change | Phase I | Question | Expected outcome | Scenes |
|---|---|---|---|---|---|
| *`full`* | *Reference, not rerun by default* | *Trained separately* | | | *All* |
| `no_separation` | `--ignore_clean_images`: one field G_I fits the original images; no glass extraction, no Phase II; renders `B` | Trained separately | Is separating the interface field and the reflection field necessary? | PSNR against the original images close to RT-Splatting; low Transmission PSNR because reflections are "baked" into B | All |
| `no_factorization` | `--ablate_factorization`: p ≡ 1, only o remains | Trained separately | Does factorizing occupancy p and opacity o help recover glass geometry? | Worse patch extraction (lower coverage/precision in `glass_geometry.json`), possibly failing the quality threshold so that Phase II refuses to run; lower Glass PSNR otherwise | All |
| `no_virtual_camera` | `--ablate_virtual_camera`: G_R is queried along the real camera rays `x_t(u) + s·d`, which continue straight through the glass, instead of the reflected rays `x_t(u) + s·r` | Copied from `full` | Do reflections need reflected rays, i.e., a virtual camera on the camera side of the glass? | Lower Glass PSNR on test views. G_R must place a "painted" layer behind the glass instead of the reflected objects in front of it, so reflections move with the wrong parallax across views | Strong reflections, wide baselines (3D-FRONT-T scene_1) |
| `pure_fresnel` | `--pure_fresnel`: h_ψ = 0 | Copied from `full` | How much does the learned correction h_ψ contribute beyond Schlick? | Lower Glass PSNR when real reflections are stronger than Schlick with F0 = 0.04 (coated or dirty glass) | All |
| `static_reflection` | `--static_reflection`: no D_R | Copied from `full` | Does G_R need to change over time in dynamic scenes? | Worse when reflections contain moving objects; nearly equal to `full` on static scenes | Dynamic only (Wayve) |

**If only static scenes are available**, `static_reflection` cannot show an effect: replace it with `no_occlusion` (below).

### How `no_virtual_camera` differs from `full`

In `full`, for a glass pixel u, the camera ray d hits the glass at `x_t(u)` and is reflected to `r = d − 2(nᵀd) n` (Eq. 9a). G_R is queried along `x_t(u) + s·r`, s ≥ 0. These rays leave the glass toward the **same side as the real camera**, which is where G_R and the reflected objects are. The virtual camera V_dyn (Eq. 9b) is placed at `c_v = x(u_ref) + ε r(u_ref)`, on the glass and on that same side, looking along r. G_R is never mirrored across the glass.

`no_virtual_camera` (`unreflected` ray mode) keeps the same glass pixels (`first_hit`), the same weight α = F + h_ψ, the same blending (Eq. 6), losses and CUDA renderer, but queries G_R along `x_t(u) + s·d`, which passes through the glass to the far side. G_R initialization (§7.6 of `implementation_details.md`) follows the active ray mode, so G_R starts behind the glass, and only Gaussians behind the glass are ever hit by a ray. This placement can fit training views, but it cannot reproduce the parallax of real reflections in new views.

## Additional variants (appendix, run only when named)

| Variant | Change | Phase I | Question | Expected outcome |
|---|---|---|---|---|
| `no_occlusion` | `--ablate_occlusion`: Eq. 9 verbatim, without occlusion checks | Copied from `full` | Are occlusion checks necessary? | Lower Glass PSNR in scenes with objects in front of the glass; the "occluded" fraction in `glass_preview/` indicates the extent |
| `single_patch` | `--max_patches 1` | Copied from `full` | Is support for multiple patches necessary? | Substantially worse on 3D-FRONT-T scene_1 (a glass box with multiple faces) and facades with multiple panes |
| `static_interface` | `--static_interface`: no D_I | Trained separately | Does G_I need deformation for dynamic scenes? | Worse on Wayve (moving cars and people); nearly equal to `full` on static scenes |
| `pinhole_virtual_camera` | `--pinhole_virtual_camera`: all rays in a region start at V_dyn's `c_v` (Eq. 9b), rather than at their own glass points | Copied from `full` | Are exact rays better than the pinhole approximation mentioned in the PDF? | Worse for reflected objects near the glass (parallax); nearly equal when they are far away |
| `no_fresnel` | `--ablate_fresnel`: a learned constant replaces F(θ) (h_ψ is still added) | Copied from `full` | Does Schlick's angular dependence matter? | Worse at oblique views; earlier results on VSGD 0758 favored the constant over Schlick, so this needs to be measured again |
| `no_motion_loss` | `--lambda_motion 0` | Copied from `full` | Does L_motion help? | Nearly equal Glass PSNR; worse temporal stability (flicker between frames), so inspect `comparison.mp4` |

## Further experiments if resources allow

- **Known versus extracted glass geometry** (`--glass_geometry`). For 3D-FRONT-T, glass geometry can be obtained from the 3D-FRONT mesh. Comparing `full` (extracted from Phase I) with known geometry reveals how much error comes from patch extraction. With known geometry, an additional `--lambda_glass 0` ablation (no mask prior) implements Eq. 3 verbatim.
- **Clean images.** DAI versus DiffRS (`CLEAN_TAG`). B is constrained by the clean images, so their quality directly limits Transmission PSNR and Opaque PSNR.
- **The same iteration budget as RT-Splatting** (61k): `--iterations 30500` for each phase, for a fair comparison.
- **Sensitivity to loss weights:** `--lambda_motion` ∈ {0.03, 0.1, 0.3}, `--lambda_optical` ∈ {0.001, 0.01, 0.1}. The PDF requires selection using validation; the current values are provisional.

## How to add an ablation

Add a flag to the appropriate parameter group in `arguments/__init__.py`. The flag must be saved in that stage's config so that `render.py` reuses it correctly. Then add an entry to `VARIANTS` in `run_ablations.py`: `(Phase I source, model flags, glass flags, Phase II flags)`. A variant that changes Phase I must use itself as its Phase I source. To run it by default, also add its name to `MAIN_SUITE`.
