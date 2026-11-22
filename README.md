# VIGOR-GS

Virtual-View Interface-Guided Optical Reflection Gaussian Splatting: a two-phase Gaussian splatting method for scenes seen through glass. Phase I fits an interface field of 2D disks (occupancy × opacity) to reflection-removed images; Phase II freezes it and fits a separate 3D reflection field, queried along the rays reflected by fixed glass patches, blended with a Fresnel weight.

- Framework (equations): [docs/framework.pdf](docs/framework.pdf), source [docs/framework.tex](docs/framework.tex)
- Implementation details, equation-to-code map and the choices made beyond the PDF: [docs/implementation_details.md](docs/implementation_details.md)
- Ablations: [docs/ablations.md](docs/ablations.md)
- Changes from the `quan_dang` branch: [docs/changes_from_quan_dang.md](docs/changes_from_quan_dang.md)

## Setup

Python 3.10, PyTorch with CUDA, and the two extensions of `submodules/`. `simple-knn` is installed as usual; `diff-surfel-anych` (2DGS rasterizer plus the reflected-ray renderer) is built in place, so an older `diff_surfel_anych` in the environment is left untouched:

```bash
pip install -r requirements.txt
pip install submodules/simple-knn
cd submodules/diff-surfel-anych && rm -rf build && TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace && cd ../..
```

Rebuild after editing any CUDA source or header (`rm -rf build` first: header changes are not tracked).

## Data

Each scene uses the aligned layout read by `scene/dataset_readers.py`: `colmap/images`, `colmap/sparse/0`, `colmap/frames.json` (time, split, sequence), `colmap/masks` and `colmap/clean_image`. Masks and clean images made on distorted source images are warped onto the COLMAP grid once with `undistort_sources.py`.

## Run

```bash
# Everything: Phase I, glass patches, Phase II, test renders, metrics
python train.py -s data/datasets/datasets/3D-FRONT-T/scene_1 -m output/3dfront -r 2

# Step by step
python train_interface.py -s <scene> -m <model> -r 2
python extract_glass.py -m <model>                 # or --glass_geometry patches.json for known glass
python train_reflection.py -m <model>
python render.py -m <model> --skip_train --components
python metrics.py -m <model>

# The ablation suite (shares Phase I where possible, writes <root>/summary.csv)
python run_ablations.py -s <scene> --root output/<scene>_ablations   # 5 ablations; -r 1 for VSGD / 3D-FRONT-T, -r 2 for Wayve; full is read from <root>/full
```

Interrupted runs continue with the same command (`train.py` skips or resumes each step). Outputs are described in [docs/implementation_details.md](docs/implementation_details.md#8-suy-luận-renderpy-eq-1213).

## Tests

```bash
python -m pytest            # 50 tests on a synthetic glass scene; GPU tests are skipped without CUDA
python -m pytest -m "not slow"
```

The RT-Splatting baseline is not part of this branch; run it from `viet_huynh` or `main`.
