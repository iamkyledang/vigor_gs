# VIGOR-GS

**Virtual-View Interface-Guided Optical Reflection Gaussian Splatting**

VIGOR-GS reconstructs scenes seen through glass using two Gaussian fields trained in two phases:

1. **Interface field (2DGS):** trained on reflection-removed images to recover the glass surface, which is then frozen and used to estimate the glass plane.
2. **Reflection field (3DGS):** trained on the original images and rendered from a virtual camera mirrored across the glass plane.

At render time, the direct and reflected views are rasterized separately and blended per pixel with a Fresnel-guided weight. Refraction and multiple reflections are not modeled.

## Installation

```shell
conda create -n rtsplat python=3.10 -y
conda activate rtsplat
pip install -r requirements.txt

pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-surfel-anych
pip install --no-build-isolation submodules/diff-gaussian-rasterization

pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
```

## Datasets

Put your datasets under the `data` folder:

```
data/
├── rt-splatting/
│   ├── van/
│   │   ├── images/
│   │   ├── sparse/
│   │   └── transparent_masks/
│   └── swab/
├── nerf-casting/
├── ...
```

VIGOR-GS scenes use this layout (see `readVigorSceneInfo` in `scene/dataset_readers.py`):

```
<scene>/
├── colmap/
│   ├── images/          original images
│   ├── sparse/0/        COLMAP model (PINHOLE cameras)
│   ├── frames.json      per-frame metadata, including the normalized time
│   ├── split.json       train / test image names
│   └── masks/           glass masks
└── clean/               reflection-removed images (same file stems as colmap/images)
```

## Training & Evaluation

VIGOR-GS, one scene at a time:

```shell
python train_interface.py -s <scene> -m output/<name> --eval    # Phase I: interface field on the clean images
python extract_plane.py -m output/<name>                        # glass plane from the interface geometry
python train_reflection.py -m output/<name>                     # Phase II: reflection field on the original images
python render_vigor.py -m output/<name>
python metrics.py -m output/<name>
```

Without clean images, add `--clean_fallback_original` to `train_interface.py` to train Phase I on the original images instead.

Ablations (each variant needs its own `-m` folder):

| Flag | Script | Variant |
|---|---|---|
| `--ablate_factorization` | `train_interface.py` | no occupancy/opacity factorization |
| `--ignore_clean_images` | `train_interface.py` | a single field fitted to the original images; evaluate with `render_vigor.py --direct_only` |
| `--ablate_virtual_camera` | `train_reflection.py` | reflection field rendered from the real camera |
| `--ablate_fresnel` | `train_reflection.py` | one learned blend constant instead of the Fresnel weight |
| `--shared_reflection_field` | `train_reflection.py` | no separate reflection field: the interface field seen from the virtual camera |

RT-Splatting baseline, all benchmark scenes:

```shell
sh eval.sh
```

## Tests

```shell
python -m pytest
```

## Author

[@iamkyledang](https://github.com/iamkyledang)
