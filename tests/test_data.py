"""Module 01: the VIGOR-GS data contract (original image, clean image, glass mask, camera, time)."""

import json
import os
from argparse import Namespace

import pytest
import torch
from PIL import Image

from scene.dataset_readers import readVigorSceneInfo
from utils.camera_utils import loadCam
from utils.color_utils import srgb2linear

pytestmark = [pytest.mark.gpu, pytest.mark.data]

CLEAN_COLOR = (64, 128, 192)


@pytest.fixture(scope='module')
def scene_info(scene_dir):
    return readVigorSceneInfo(scene_dir, eval=True, clean_fallback_original=True)


@pytest.fixture(scope='module')
def clean_dir(scene_dir, tmp_path_factory):
    """A stand-in clean folder: one flat-colored image per frame."""
    folder = tmp_path_factory.mktemp('clean')
    with open(os.path.join(scene_dir, 'colmap', 'split.json')) as f:
        split = json.load(f)
    flat = Image.new('RGB', (96, 54), CLEAN_COLOR)
    for name in split['train'] + split['test']:
        flat.save(folder / (os.path.splitext(name)[0] + '.png'))
    return str(folder)


def load_args(**overrides):
    args = Namespace(resolution=4, data_device='cuda', linear_rgb=False)
    vars(args).update(overrides)
    return args


def test_split_follows_split_json(scene_dir, scene_info):
    with open(os.path.join(scene_dir, 'colmap', 'split.json')) as f:
        split = json.load(f)
    train = {os.path.basename(c.image_path) for c in scene_info.train_cameras}
    test = {os.path.basename(c.image_path) for c in scene_info.test_cameras}
    assert train == set(split['train'])
    assert test == set(split['test'])
    assert not train & test


def test_without_eval_everything_trains(scene_dir, scene_info):
    info = readVigorSceneInfo(scene_dir, eval=False, clean_fallback_original=True)
    assert len(info.train_cameras) == len(scene_info.train_cameras) + len(scene_info.test_cameras)
    assert info.test_cameras == []


def test_time_is_normalized_and_ordered(scene_info):
    cams = sorted(scene_info.train_cameras + scene_info.test_cameras, key=lambda c: c.image_name)
    times = [c.time for c in cams]
    assert times[0] == 0.0 and times[-1] == 1.0
    assert all(a < b for a, b in zip(times, times[1:]))


def test_masks_match_images_and_points_exist(scene_info):
    for cam in scene_info.train_cameras[:10]:
        assert cam.transparent_mask.size == cam.image.size == (cam.width, cam.height)
        assert cam.transparent_mask.mode == 'L'
    assert scene_info.point_cloud.points.shape[0] > 1000
    assert scene_info.nerf_normalization['radius'] > 0


def test_missing_clean_images_is_an_error(scene_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match='clean'):
        readVigorSceneInfo(scene_dir, eval=True, clean_images=str(tmp_path))


def test_fallback_uses_original_as_clean(scene_info):
    cam = loadCam(load_args(), 0, scene_info.train_cameras[0], 1.0)
    assert not cam.has_clean_image
    assert torch.equal(cam.clean_image, cam.original_image)


def test_camera_carries_clean_image_mask_and_time(scene_dir, clean_dir):
    info = readVigorSceneInfo(scene_dir, eval=True, clean_images=clean_dir)
    cam_info = info.train_cameras[5]
    cam = loadCam(load_args(), 0, cam_info, 1.0)

    assert cam.time == cam_info.time
    assert cam.has_clean_image
    assert cam.original_image.shape == cam.clean_image.shape == (3, cam.image_height, cam.image_width)
    assert cam.gt_transparent_mask.shape == (1, cam.image_height, cam.image_width)
    assert cam.gt_transparent_mask.dtype == torch.bool

    expected = torch.tensor(CLEAN_COLOR, device=cam.clean_image.device).float() / 255
    assert torch.allclose(cam.clean_image.mean(dim=(1, 2)), expected, atol=1e-3)
    assert not torch.allclose(cam.original_image, cam.clean_image)


def test_clean_images_in_the_split_layout_are_found_and_resized(scene_dir, tmp_path):
    """The shipped layout: <clean>/<split>/images/<stem>.png, mirroring <scene>/<split>/images/, at another size."""
    with open(os.path.join(scene_dir, 'colmap', 'split.json')) as f:
        split = json.load(f)
    flat = Image.new('RGB', (100, 50), CLEAN_COLOR)
    for split_name, names in split.items():
        os.makedirs(tmp_path / split_name / 'images')
        for name in names:
            flat.save(tmp_path / split_name / 'images' / (os.path.splitext(name)[0] + '.png'))

    info = readVigorSceneInfo(scene_dir, eval=True, clean_images=str(tmp_path))
    assert all(c.clean_image is not None for c in info.train_cameras + info.test_cameras)
    cam = loadCam(load_args(), 0, info.test_cameras[0], 1.0)
    assert cam.clean_image.shape == cam.original_image.shape
    expected = torch.tensor(CLEAN_COLOR, device=cam.clean_image.device).float() / 255
    assert torch.allclose(cam.clean_image.mean(dim=(1, 2)), expected, atol=1e-3)


def test_linear_rgb_converts_both_targets(scene_dir, clean_dir):
    info = readVigorSceneInfo(scene_dir, eval=True, clean_images=clean_dir)
    srgb = loadCam(load_args(), 0, info.train_cameras[0], 1.0)
    linear = loadCam(load_args(linear_rgb=True), 0, info.train_cameras[0], 1.0)

    assert torch.allclose(linear.original_image, srgb2linear(srgb.original_image), atol=1e-6)
    assert torch.allclose(linear.clean_image, srgb2linear(srgb.clean_image), atol=1e-6)
    assert linear.original_image.mean() < srgb.original_image.mean()


def test_srgb_linear_conversions_are_inverse():
    from utils.color_utils import linear2srgb

    srgb = torch.linspace(0.0, 1.0, 256)
    linear = srgb2linear(srgb)
    assert torch.allclose(linear[[0, -1]], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(srgb2linear(torch.tensor([0.5])), torch.tensor([0.21404]), atol=1e-4)
    assert torch.allclose(linear2srgb(linear), srgb, atol=1e-5)


def test_scene_detects_vigor_layout(scene_dir, tmp_path):
    from scene import Scene

    class StubGaussians:
        def create_from_pcd(self, pcd, spatial_lr_scale):
            self.num_points = pcd.points.shape[0]
            self.spatial_lr_scale = spatial_lr_scale

    args = Namespace(source_path=scene_dir, model_path=str(tmp_path), images='images', eval=True, resolution=8, data_device='cpu', white_background=False, clean_fallback_original=True)
    gaussians = StubGaussians()
    scene = Scene(args, gaussians, shuffle=False)

    assert len(scene.getTrainCameras()) == 137
    assert len(scene.getTestCameras()) == 34
    assert gaussians.num_points > 1000
    assert scene.getTrainCameras()[1].time > scene.getTrainCameras()[0].time
    assert os.path.exists(tmp_path / 'cameras.json')
