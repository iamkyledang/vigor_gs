#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from torch import nn

from utils.color_utils import srgb2linear
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.memory_utils import DEFAULT_VRAM_FRACTION, get_placer


class Camera(nn.Module):
    """
    A training / test view. Its images and glass mask are stored where `data_device` says (utils/memory_utils.py):
    on the GPU, in pinned host RAM, or either one depending on the VRAM budget ('auto'). Images given as uint8 are
    kept as uint8 and converted when read. Reading original_image, clean_image, gt_transparent_mask or gt_alpha_mask
    always returns a CUDA tensor: float images in [0, 1] (linear RGB when linear_rgb is set) and a bool mask.
    """
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        gt_transparent_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device='cuda',
        time=0.0,
        clean_image=None,
        linear_rgb=False,
        data_vram_fraction=DEFAULT_VRAM_FRACTION,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.time = float(time)
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.placer = get_placer(data_device, data_vram_fraction)
        self.linear_rgb = bool(linear_rgb)

        self.original_image = image if image.dtype == torch.uint8 else image.clamp(0.0, 1.0)
        self.data_device = self._original.device  # where this view's data ended up
        self.image_height, self.image_width = image.shape[1], image.shape[2]

        self.gt_transparent_mask = gt_transparent_mask
        self.gt_alpha_mask = gt_alpha_mask

        # Reflection-removed target for the interface phase. Without one, the original image stands in.
        self.has_clean_image = clean_image is not None
        self.clean_image = None if clean_image is None else (clean_image if clean_image.dtype == torch.uint8 else clean_image.clamp(0.0, 1.0))

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # ---- per-view data: stored by the placer, read back as CUDA tensors

    def _store(self, tensor):
        return None if tensor is None else self.placer.place(tensor)

    def _image(self, stored):
        image = stored.to('cuda', non_blocking=True)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
            if self.linear_rgb:  # the colour channels only; an alpha channel stays as it is
                image = torch.cat([srgb2linear(image[:3]), image[3:]], dim=0) if image.shape[0] > 3 else srgb2linear(image)
        return image

    @property
    def original_image(self):
        return self._image(self._original)

    @original_image.setter
    def original_image(self, value):
        self._original = self._store(value)

    @property
    def clean_image(self):
        return self.original_image[:3] if self._clean is None else self._image(self._clean)

    @clean_image.setter
    def clean_image(self, value):
        self._clean = self._store(value)

    @property
    def gt_transparent_mask(self):
        return self._mask.to('cuda', non_blocking=True)

    @gt_transparent_mask.setter
    def gt_transparent_mask(self, value):
        self._mask = self._store(value)

    @property
    def gt_alpha_mask(self):
        return None if self._alpha is None else self._alpha.to('cuda', non_blocking=True)

    @gt_alpha_mask.setter
    def gt_alpha_mask(self, value):
        self._alpha = self._store(value)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
