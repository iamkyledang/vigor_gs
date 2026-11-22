"""
Virtual camera C^v = V(C, Pi): the real camera mirrored across the glass plane Pi: n^T x + d = 0.

The reflection field G_R lives in real world coordinates (the reflected objects are on the camera side of the
glass). Seen through the glass, an object at X appears at its mirror image M X, so the real camera images it at

    P * W2C * (M X) = P * (W2C * M) * X.

Rendering G_R with the view matrix W2C * M therefore gives the reflection exactly as the real camera sees it,
pixel for pixel: no image flip, and the Gaussians are never mirrored themselves. The mirrored view matrix has
determinant -1 (it flips handedness), which is what makes the two images line up.
"""

import torch


def reflection_matrix(normal, offset):
    """4x4 matrix M of the reflection across the plane: M [x; 1] = [x - 2 (n^T x + d) n; 1]."""
    matrix = torch.eye(4, dtype=normal.dtype, device=normal.device)
    matrix[:3, :3] -= 2.0 * torch.outer(normal, normal)
    matrix[:3, 3] = -2.0 * offset * normal
    return matrix


def reflect_points(points, normal, offset):
    """Mirror (..., 3) points across the plane: x - 2 (n^T x + d) n."""
    return points - 2.0 * (points @ normal + offset).unsqueeze(-1) * normal


class VirtualCamera:
    """The attributes the rasterizers read from a camera, for a mirrored view."""

    def __init__(self, image_width, image_height, FoVx, FoVy, znear, zfar, world_view_transform, projection_matrix, camera_center, time=0.0):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.projection_matrix = projection_matrix
        self.full_proj_transform = world_view_transform @ projection_matrix
        self.camera_center = camera_center
        self.time = time


def reflect_camera(camera, normal, offset):
    """
    Mirror a camera across the glass plane.

    The codebase stores view matrices transposed (row vectors: x_cam = x_world @ world_view_transform), so the
    mirrored view W2C * M is M^T @ world_view_transform. Intrinsics, image size and timestamp are unchanged, and the
    camera center becomes c - 2 (n^T c + d) n.
    """
    normal = normal.to(camera.world_view_transform)
    mirror = reflection_matrix(normal, offset)
    return VirtualCamera(
        image_width=camera.image_width,
        image_height=camera.image_height,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        znear=camera.znear,
        zfar=camera.zfar,
        world_view_transform=mirror.T @ camera.world_view_transform,
        projection_matrix=camera.projection_matrix,
        camera_center=reflect_points(camera.camera_center, normal, offset),
        time=getattr(camera, 'time', 0.0),
    )
