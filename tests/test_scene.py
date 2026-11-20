"""Scene reader and cameras: layout, calibration, resolution, color space and masks."""

import json
import os

import numpy as np
import pytest
import torch

from scene.dataset_readers import find_clean_image, readVigorSceneInfo
from tests.conftest import model_args

pytestmark = pytest.mark.gpu  # Camera keeps its transforms on the GPU


def test_reader_reads_splits_times_and_aligned_files(synthetic_scene):
    info = readVigorSceneInfo(synthetic_scene.path)
    assert len(info.train_cameras) == 9 and len(info.test_cameras) == 3
    assert all(c.split == 'train' for c in info.train_cameras)
    times = sorted(c.time for c in info.train_cameras + info.test_cameras)
    assert times[0] == 0.0 and times[-1] == 1.0
    camera = info.train_cameras[0]
    assert camera.mask_path.endswith(os.path.join('colmap', 'masks', camera.image_name + '.png'))
    assert os.path.join('colmap', 'clean_image', 'train', 'images') in camera.clean_path
    assert camera.valid is None  # no source_warp.json: every pixel is valid
    assert info.extent > 0 and len(info.point_cloud.points) == 3000


def test_missing_clean_training_image_is_an_error_unless_ignored(synthetic_scene, tmp_path):
    import shutil

    copy = tmp_path / 'scene'
    shutil.copytree(synthetic_scene.path, copy)
    os.remove(copy / 'colmap' / 'clean_image' / 'train' / 'images' / '00000000.png')
    with pytest.raises(FileNotFoundError, match='no clean image'):
        readVigorSceneInfo(str(copy))
    info = readVigorSceneInfo(str(copy), ignore_clean_images=True)
    assert all(c.clean_path is None for c in info.train_cameras)


def test_clean_lookup_order(tmp_path):
    (tmp_path / 'train' / 'images').mkdir(parents=True)
    (tmp_path / 'train' / 'images' / 'a.png').touch()
    (tmp_path / 'a.jpg').touch()
    assert find_clean_image(str(tmp_path), {}, 'train', 'a').endswith(os.path.join('train', 'images', 'a.png'))
    assert find_clean_image(str(tmp_path), {}, 'test', 'a').endswith('a.jpg')
    assert find_clean_image(str(tmp_path), {}, 'test', 'b') is None


def test_camera_scales_intrinsics_and_decodes_linear_rgb(synthetic_scene):
    from scene.cameras import Camera

    info = readVigorSceneInfo(synthetic_scene.path)
    full = Camera(info.train_cameras[0], resolution=1)
    half = Camera(info.train_cameras[0], resolution=2)
    assert (half.image_width, half.image_height) == (32, 24)
    assert half.fx == pytest.approx(full.fx / 2) and half.cx == pytest.approx(full.cx / 2)
    srgb, linear = full.original_srgb, full.original_linear
    assert srgb.shape == (3, 48, 64) and srgb.is_cuda
    assert torch.all(linear <= srgb + 1e-6)  # sRGB decoding darkens mid tones
    assert full.glass_mask.dtype == torch.bool and full.glass_mask.any()
    assert full.valid_mask.all()
    # The rasterizer projection and the ray convention of utils/point_utils.py agree on pixel centers.
    from utils.point_utils import camera_rays

    origin, directions = camera_rays(full)
    point = origin + 2.0 * directions[10, 20]
    clip = torch.cat([point, point.new_ones(1)]) @ full.full_proj_transform
    ndc = clip[:2] / clip[3]
    pixel = ((ndc + 1) * torch.tensor([64.0, 48.0], device='cuda') - 1) / 2
    assert torch.allclose(pixel, torch.tensor([20.0, 10.0], device='cuda'), atol=1e-3)


def test_valid_region_from_source_warp(synthetic_scene, tmp_path):
    import shutil

    copy = tmp_path / 'scene'
    shutil.copytree(synthetic_scene.path, copy)
    # A source image half as wide as the COLMAP grid, with no distortion: only the middle half is covered.
    warp = {'cameras': {'1': {'undistorted': {'fx': 57.6, 'fy': 57.6, 'cx': 32.0, 'cy': 24.0, 'width': 64, 'height': 48},
                              'source_size': [32, 48], 'model': 'radial',
                              'params': {'fx': 57.6, 'fy': 57.6, 'cx': 16.0, 'cy': 24.0, 'k1': 0.0, 'k2': 0.0}}}}
    with open(copy / 'colmap' / 'source_warp.json', 'w') as f:
        json.dump(warp, f)
    camera = readVigorSceneInfo(str(copy)).train_cameras[0]
    assert camera.valid.shape == (48, 64)
    columns = camera.valid.all(0)
    assert columns[16:48].all() and not columns[:15].any() and not columns[49:].any()


def test_scene_builds_cameras_at_the_resolution(synthetic_scene):
    from scene import Scene

    scene = Scene(model_args(synthetic_scene.path, resolution=2))
    assert len(scene.getTrainCameras()) == 9
    assert scene.getTrainCameras()[0].image_width == 32
    assert np.isfinite(scene.center).all()
