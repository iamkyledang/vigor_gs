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

from scene.cameras import Camera
from utils.color_utils import srgb2linear
from utils.graphics_utils import fov2focal
from utils.memory_utils import DEFAULT_VRAM_FRACTION, image_to_tensor

WARNED = False


def get_rays(W, H, FoVx, FoVy, world_view_transform):
    fx = fov2focal(FoVx, W)
    fy = fov2focal(FoVy, H)

    K = torch.tensor([[fx, 0, 0.5 * W], [0, fy, 0.5 * H], [0, 0, 1]], device='cuda')

    c2w = world_view_transform.T.inverse()
    c2w[:3, 1:3] *= -1

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device='cuda'), torch.linspace(0, H - 1, H, device='cuda'))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 5, 6, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # uint8 as decoded (the Camera converts on use, see utils/memory_utils.py); float in [0, 1] for other bit depths.
    resized_image_rgb = image_to_tensor(cam_info.image, resolution)
    gt_transparent_mask = image_to_tensor(cam_info.transparent_mask, resolution) > 0

    clean_image = None
    if getattr(cam_info, 'clean_image', None) is not None:
        clean_image = image_to_tensor(cam_info.clean_image, resolution)[:3]

    # The reflection blend is physically linear, so the VIGOR-GS pipeline trains on linear RGB. uint8 images are
    # converted by the Camera when read; images that already are float are converted here.
    linear_rgb = getattr(args, 'linear_rgb', False)
    if linear_rgb and resized_image_rgb.dtype != torch.uint8:
        resized_image_rgb = torch.cat([srgb2linear(resized_image_rgb[:3]), resized_image_rgb[3:]], dim=0)
    if linear_rgb and clean_image is not None and clean_image.dtype != torch.uint8:
        clean_image = srgb2linear(clean_image)

    gt_image = resized_image_rgb
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        alpha = resized_image_rgb[3:4, ...]
        loaded_mask = alpha.float() / 255.0 if alpha.dtype == torch.uint8 else alpha

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        gt_transparent_mask=gt_transparent_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        time=getattr(cam_info, 'time', 0.0),
        clean_image=clean_image,
        linear_rgb=linear_rgb,
        data_vram_fraction=getattr(args, 'data_vram_fraction', DEFAULT_VRAM_FRACTION),
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    if camera_list:
        print('[ INFO ]', camera_list[0].placer.summary())
    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
