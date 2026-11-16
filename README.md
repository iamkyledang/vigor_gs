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

## Training & Evaluation

```shell
sh eval.sh
```

## Author

[@iamkyledang](https://github.com/iamkyledang)
