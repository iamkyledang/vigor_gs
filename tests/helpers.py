"""Small builders shared by the tests."""

import math

import numpy as np
import torch

from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def look_at_camera(eye, target, up=(0.0, -1.0, 0.0), fov_deg=60.0, width=96, height=64, znear=0.01, zfar=100.0):
    """A MiniCam at `eye` looking at `target` (OpenCV convention: +z forward, +y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(-np.asarray(up, dtype=np.float64), forward)  # x = y (down) cross z (forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    c2w_rot = np.stack([right, down, forward], axis=1)  # columns are the camera axes in world space
    R = c2w_rot  # the codebase stores R as the transposed W2C rotation, i.e. the C2W rotation
    T = -c2w_rot.T @ eye

    fovx = math.radians(fov_deg)
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)

    world_view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)


def parse_interface_args(argv):
    """Parse a train_interface.py command line into (dataset, opt, pipe)."""
    from argparse import ArgumentParser

    from arguments import InterfaceOptimizationParams, ModelParams, PipelineParams

    parser = ArgumentParser()
    lp, op, pp = ModelParams(parser), InterfaceOptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args(argv)
    return lp.extract(args), op.extract(args), pp.extract(args)


def project_points(cam, points):
    """Project world points (N, 3) to pixel coordinates (N, 2) using the camera's full projection."""
    points = torch.as_tensor(points, dtype=torch.float32, device='cuda')
    hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1) @ cam.full_proj_transform
    ndc = hom[:, :2] / hom[:, 3:4]
    px = (ndc[:, 0] + 1) * cam.image_width / 2 - 0.5
    py = (ndc[:, 1] + 1) * cam.image_height / 2 - 0.5
    return torch.stack([px, py], dim=-1)
