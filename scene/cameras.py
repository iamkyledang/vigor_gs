"""
A calibrated training or test view.

Pixels are decoded once, at the training resolution, and kept as uint8 (images) or bool (masks) on `data_device`
(host RAM by default), so a long sequence costs a quarter of the float32 memory and the GPU only holds the frame in
use. The properties return CUDA tensors: images in **linear RGB**, where the framework fits and blends (Eq. 3, 6),
except `original_srgb`, which is what renders and metrics are compared against after encoding.
"""

import math

import cv2
import numpy as np
import torch
from PIL import Image

from utils.color_utils import srgb2linear
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def load_rgb(path, size):
    """(H, W, 3) uint8 at `size` = (width, height); area filter when shrinking."""
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'))
    if (array.shape[1], array.shape[0]) != size:
        array = cv2.resize(array, size, interpolation=cv2.INTER_AREA)
    return array


def load_mask(path, size):
    """(H, W) bool at `size`; a pixel is glass when more than half of its area is."""
    with Image.open(path) as image:
        array = np.asarray(image.convert('L')).astype(np.float32) / 255.0
    if (array.shape[1], array.shape[0]) != size:
        array = cv2.resize(array, size, interpolation=cv2.INTER_AREA)
    return array > 0.5


def resize_region(region, size):
    """A full-resolution valid region at `size`; a pixel is valid only if all of its area is."""
    if (region.shape[1], region.shape[0]) == size:
        return region
    return cv2.resize(region.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.999


class Camera:
    def __init__(self, info, resolution=1, data_device='cpu', znear=0.01, zfar=1000.0):
        if resolution < 1:
            raise ValueError(f'resolution is a downsampling factor >= 1, got {resolution}')
        self.uid = info.uid
        self.sensor_id = info.sensor_id
        self.image_name = info.image_name
        self.split = info.split
        self.time = info.time
        self.sequence = info.sequence
        self.R, self.T = info.R, info.T

        self.image_width = max(1, round(info.width / resolution))
        self.image_height = max(1, round(info.height / resolution))
        sx, sy = self.image_width / info.width, self.image_height / info.height
        self.fx, self.fy = info.fx * sx, info.fy * sy
        self.cx, self.cy = info.cx * sx, info.cy * sy
        self.FoVx = 2 * math.atan(self.image_width / (2 * self.fx))
        self.FoVy = 2 * math.atan(self.image_height / (2 * self.fy))
        self.znear, self.zfar = znear, zfar

        size = (self.image_width, self.image_height)
        self.data_device = torch.device(data_device)
        self._original = self._store(load_rgb(info.image_path, size))
        self._clean = self._store(load_rgb(info.clean_path, size)) if info.clean_path else None
        mask = load_mask(info.mask_path, size) if info.mask_path else np.zeros(size[::-1], bool)
        valid = resize_region(info.valid, size) if info.valid is not None else np.ones(size[::-1], bool)
        self._mask = torch.from_numpy(mask)[None].to(self.data_device)
        self._valid = torch.from_numpy(valid)[None].to(self.data_device)
        self.has_glass_mask = info.mask_path is not None

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()
        projection = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy)
        # Off-center principal point: x_ndc = (fx x / z + cx - W / 2) / (W / 2).
        projection[0, 2] = 2 * self.cx / self.image_width - 1
        projection[1, 2] = 2 * self.cy / self.image_height - 1
        self.projection_matrix = projection.transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def _store(self, array):
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).to(self.data_device)

    @staticmethod
    def _decode(stored):
        return stored.cuda(non_blocking=True).float().div_(255.0)

    @property
    def original_srgb(self):
        return self._decode(self._original)

    @property
    def original_linear(self):
        return srgb2linear(self.original_srgb)

    @property
    def has_clean(self):
        return self._clean is not None

    @property
    def clean_linear(self):
        if self._clean is None:
            raise ValueError(f'{self.image_name} has no clean image')
        return srgb2linear(self._decode(self._clean))

    def release_clean(self):
        """Phase II never reads the clean targets; free them."""
        self._clean = None

    @property
    def glass_mask(self):
        """(1, H, W) bool annotation M_t. Phase I geometry prior, patch extraction and glass metrics only."""
        return self._mask.cuda(non_blocking=True)

    @property
    def valid_mask(self):
        """(1, H, W) bool: pixels covered by the source image (False on undistortion borders)."""
        return self._valid.cuda(non_blocking=True)
