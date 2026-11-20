"""Where per-view data lives (utils/memory_utils.py): uint8 images, GPU / pinned RAM placement, lazy files, bounded saving."""

import threading
import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scene.cameras import Camera
from utils.camera_utils import loadCam
from utils.color_utils import srgb2linear
from utils.general_utils import PILtoTorch
from utils.memory_utils import BoundedExecutor, DataPlacer, LazyImage, image_to_tensor

pytestmark = pytest.mark.gpu

H, W = 30, 40


def make_camera(image, mask=None, clean=None, **kw):
    mask = torch.zeros(1, image.shape[1], image.shape[2], dtype=torch.bool) if mask is None else mask
    return Camera(0, np.eye(3), np.zeros(3), 1.0, 0.8, image, None, mask, 'view', 0, clean_image=clean, **kw)


def random_u8(channels=3):
    return torch.randint(0, 256, (channels, H, W), dtype=torch.uint8)


def test_placer_modes_and_the_vram_budget():
    small = torch.zeros(1000, dtype=torch.uint8)  # 1000 bytes
    assert DataPlacer('cuda').place(small).is_cuda
    assert DataPlacer('cuda:0').place(small).is_cuda
    on_cpu = DataPlacer('cpu').place(small.cuda())
    assert not on_cpu.is_cuda and on_cpu.is_pinned()

    auto = DataPlacer('auto', budget_bytes=2500)
    placed = [auto.place(small) for _ in range(4)]
    assert [t.is_cuda for t in placed] == [True, True, False, False]  # GPU until the budget is used, then RAM
    assert (auto.gpu_bytes, auto.cpu_bytes) == (2000, 2000)
    assert 'auto' in auto.summary()
    with pytest.raises(ValueError):
        DataPlacer('tpu')


@pytest.mark.parametrize('linear_rgb', [False, True])
def test_uint8_images_read_back_exactly_like_the_float_path(linear_rgb):
    image, clean = random_u8(), random_u8()
    mask = torch.rand(1, H, W) > 0.5
    cam = make_camera(image, mask, clean, linear_rgb=linear_rgb)

    assert cam._original.dtype == torch.uint8 and cam._clean.dtype == torch.uint8  # stored compactly
    expected = image.float() / 255.0  # what PILtoTorch gave before
    expected_clean = clean.float() / 255.0
    if linear_rgb:
        expected, expected_clean = srgb2linear(expected), srgb2linear(expected_clean)
    assert cam.original_image.is_cuda and cam.original_image.dtype == torch.float32
    assert torch.allclose(cam.original_image, expected.cuda(), atol=1e-6)
    assert torch.allclose(cam.clean_image, expected_clean.cuda(), atol=1e-6)
    assert torch.equal(cam.gt_transparent_mask, mask.cuda())
    assert (cam.image_height, cam.image_width) == (H, W)


def test_views_beyond_the_budget_live_in_ram_and_read_back_on_the_gpu():
    image = random_u8()
    cam = make_camera(image, data_device='cpu')
    assert not cam._original.is_cuda and cam._original.is_pinned() and cam.data_device.type == 'cpu'
    assert cam.original_image.is_cuda and cam.gt_transparent_mask.is_cuda
    assert torch.allclose(cam.original_image, (image.float() / 255.0).cuda(), atol=1e-6)  # /255 on the GPU: within an ulp


def test_without_a_clean_image_the_original_stands_in_and_attributes_can_be_set():
    cam = make_camera(random_u8())
    assert not cam.has_clean_image and torch.equal(cam.clean_image, cam.original_image)
    target = torch.rand(3, H, W, device='cuda')  # tests and tools replace targets with float tensors
    cam.original_image = target
    assert torch.equal(cam.original_image, target)
    cam.gt_transparent_mask = torch.ones(1, H, W, dtype=torch.bool, device='cuda')
    assert cam.gt_transparent_mask.all()


def test_float_images_keep_working():
    image = torch.rand(3, H, W) * 1.2  # e.g. 16-bit sources come in as float; values are clamped as before
    cam = make_camera(image)
    assert torch.allclose(cam.original_image, image.clamp(0, 1).cuda())


def test_lazy_image_matches_pil_and_holds_nothing(tmp_path):
    rgb = (np.random.default_rng(0).uniform(0, 255, (H, W, 3))).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / 'a.png')
    Image.fromarray(rgb).convert('RGB').save(tmp_path / 'm.png')  # a mask saved as RGB is read as L

    lazy = LazyImage(str(tmp_path / 'a.png'))
    assert lazy.size == (W, H) and lazy.mode == 'RGB' and lazy.filename.endswith('a.png')
    assert not hasattr(lazy, 'im') and not hasattr(lazy, 'fp')  # no pixels, no open file
    for resolution in ((W, H), (20, 15)):
        reference = PILtoTorch(Image.open(tmp_path / 'a.png'), resolution)
        tensor = image_to_tensor(lazy, resolution)
        assert tensor.dtype == torch.uint8 and torch.equal(tensor.float() / 255.0, reference)

    mask = LazyImage(str(tmp_path / 'm.png'), convert='L')
    assert mask.mode == 'L'
    assert image_to_tensor(mask, (20, 15)).shape == (1, 15, 20)


def test_load_cam_from_lazy_images_matches_the_eager_pil_path(tmp_path):
    rng = np.random.default_rng(1)
    for name in ('image', 'clean'):
        Image.fromarray(rng.uniform(0, 255, (H, W, 3)).astype(np.uint8)).save(tmp_path / f'{name}.png')
    Image.fromarray(((rng.uniform(size=(H, W)) > 0.5) * 255).astype(np.uint8)).save(tmp_path / 'mask.png')

    def info(lazy):
        def open_(name, convert=None):
            if lazy:
                return LazyImage(str(tmp_path / f'{name}.png'), convert=convert)
            image = Image.open(tmp_path / f'{name}.png')
            return image.convert(convert) if convert else image

        return SimpleNamespace(uid=1, R=np.eye(3), T=np.zeros(3), FovX=1.0, FovY=0.8, image_name='v', time=0.5,
                               image=open_('image'), transparent_mask=open_('mask', 'L'), clean_image=open_('clean'))

    for linear in (False, True):
        args = Namespace(resolution=2, data_device='cuda', linear_rgb=linear)
        lazy, eager = loadCam(args, 0, info(True), 1.0), loadCam(args, 0, info(False), 1.0)
        reference_image = PILtoTorch(Image.open(tmp_path / 'image.png'), (W // 2, H // 2))
        if linear:
            reference_image = srgb2linear(reference_image)
        assert torch.allclose(lazy.original_image, reference_image.cuda(), atol=1e-6)
        assert torch.equal(lazy.original_image, eager.original_image)
        assert torch.equal(lazy.clean_image, eager.clean_image)
        assert torch.equal(lazy.gt_transparent_mask, eager.gt_transparent_mask)
        assert lazy.time == 0.5


def test_bounded_executor_never_holds_more_than_its_limit():
    pending, peak, lock = [0], [0], threading.Lock()

    def job():
        time.sleep(0.01)
        with lock:
            pending[0] -= 1

    with BoundedExecutor(max_workers=2, max_pending=4) as executor:
        for _ in range(20):
            with lock:
                pending[0] += 1
                peak[0] = max(peak[0], pending[0])
            executor.submit(job)
    assert pending[0] == 0 and peak[0] <= 5  # the one being submitted plus at most 4 waiting or running
