"""Module 06: the virtual camera, i.e. the real camera mirrored across the glass plane."""

import pytest
import torch

from tests.helpers import look_at_camera, project_points
from utils.virtual_camera import reflect_camera, reflect_points, reflection_matrix

pytestmark = pytest.mark.gpu  # the camera helper builds its matrices on the GPU


def plane(normal, offset):
    return torch.nn.functional.normalize(torch.tensor(normal, dtype=torch.float32, device='cuda'), dim=0), offset


def view_depth(cam, points):
    """Depth along the optical axis of world points (N, 3)."""
    hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1) @ cam.world_view_transform
    return hom[:, 2]


def test_reflection_formula_on_the_document_example():
    # Plane z = 0: the camera at (1, 0, 2) is mirrored to (1, 0, -2).
    normal, offset = plane([0.0, 0.0, 1.0], 0.0)
    cam = look_at_camera(eye=(1.0, 0.0, 2.0), target=(0.0, 0.0, 0.0))
    virtual = reflect_camera(cam, normal, offset)
    assert torch.allclose(virtual.camera_center, torch.tensor([1.0, 0.0, -2.0], device='cuda'), atol=1e-6)
    # c^v = c - 2 (n^T c + d) n for a general plane as well
    normal, offset = plane([0.2, -0.4, 0.9], 1.3)
    expected = cam.camera_center - 2 * (cam.camera_center @ normal + offset) * normal
    assert torch.allclose(reflect_camera(cam, normal, offset).camera_center, expected, atol=1e-6)


def test_reflection_matrix_matches_point_reflection_and_is_an_involution():
    normal, offset = plane([0.3, 0.5, -0.8], -0.7)
    mirror = reflection_matrix(normal, offset)
    points = torch.randn(20, 3, device='cuda')
    hom = torch.cat([points, torch.ones(20, 1, device='cuda')], dim=-1)
    assert torch.allclose((hom @ mirror.T)[:, :3], reflect_points(points, normal, offset), atol=1e-6)
    assert torch.allclose(mirror @ mirror, torch.eye(4, device='cuda'), atol=1e-6)
    assert torch.det(mirror[:3, :3]).item() == pytest.approx(-1.0, abs=1e-5)

    reflected = reflect_points(points, normal, offset)
    assert torch.allclose(points @ normal + offset, -(reflected @ normal + offset), atol=1e-5)


def test_virtual_camera_sees_objects_where_the_real_camera_sees_their_mirror_images():
    # Glass at z = 1 with the camera in front of it; the reflected objects are on the camera's side.
    normal, offset = plane([0.0, 0.0, -1.0], 1.0)
    cam = look_at_camera(eye=(0.3, -0.2, -2.0), target=(0.0, 0.0, 1.0))
    virtual = reflect_camera(cam, normal, offset)

    objects = torch.tensor([[0.5, 0.2, -1.0], [-0.8, 0.4, -3.0], [0.1, -0.6, 0.2]], device='cuda')
    mirror_images = reflect_points(objects, normal, offset)
    assert (mirror_images[:, 2] > 1.0).all()  # the mirror images lie behind the glass

    assert torch.allclose(project_points(virtual, objects), project_points(cam, mirror_images), atol=1e-3)
    assert torch.allclose(view_depth(virtual, objects), view_depth(cam, mirror_images), atol=1e-5)
    assert (view_depth(virtual, objects) > 0).all()


def test_points_on_the_glass_project_to_the_same_pixel_in_both_cameras():
    normal, offset = plane([0.1, 0.2, -1.0], 1.0)
    cam = look_at_camera(eye=(0.0, 0.0, -2.0), target=(0.0, 0.0, 1.0))
    virtual = reflect_camera(cam, normal, offset)

    in_plane = torch.randn(10, 3, device='cuda') * 0.5
    on_glass = in_plane - (in_plane @ normal + offset).unsqueeze(-1) * normal
    assert torch.allclose(on_glass @ normal + offset, torch.zeros(10, device='cuda'), atol=1e-6)
    assert torch.allclose(project_points(virtual, on_glass), project_points(cam, on_glass), atol=1e-3)


def test_moving_the_real_camera_moves_the_virtual_one_symmetrically():
    normal, offset = plane([0.0, 0.0, -1.0], 1.0)
    near = reflect_camera(look_at_camera(eye=(0.0, 0.0, -1.0), target=(0.0, 0.0, 1.0)), normal, offset)
    far = reflect_camera(look_at_camera(eye=(0.5, 0.0, -2.0), target=(0.0, 0.0, 1.0)), normal, offset)
    # Sideways motion is shared, motion along the normal is opposite.
    assert torch.allclose(far.camera_center - near.camera_center, torch.tensor([0.5, 0.0, 1.0], device='cuda'), atol=1e-6)


def test_mirrored_view_flips_handedness_and_keeps_intrinsics():
    normal, offset = plane([0.4, -0.1, 0.9], 0.6)
    cam = look_at_camera(eye=(1.0, 0.5, 3.0), target=(0.0, 0.0, 0.0))
    cam.time = 0.25
    virtual = reflect_camera(cam, normal, offset)

    assert torch.det(virtual.world_view_transform[:3, :3]).item() == pytest.approx(-1.0, abs=1e-5)
    assert torch.allclose(torch.inverse(virtual.world_view_transform)[3, :3], virtual.camera_center, atol=1e-5)
    assert torch.allclose(virtual.full_proj_transform, virtual.world_view_transform @ cam.projection_matrix)
    assert (virtual.image_width, virtual.image_height, virtual.FoVx, virtual.FoVy, virtual.time) == (cam.image_width, cam.image_height, cam.FoVx, cam.FoVy, 0.25)


def test_reflecting_twice_returns_the_real_camera():
    normal, offset = plane([0.4, -0.1, 0.9], 0.6)
    cam = look_at_camera(eye=(1.0, 0.5, 3.0), target=(0.0, 0.0, 0.0))
    back = reflect_camera(reflect_camera(cam, normal, offset), normal, offset)
    assert torch.allclose(back.world_view_transform, cam.world_view_transform, atol=1e-5)
    assert torch.allclose(back.full_proj_transform, cam.full_proj_transform, atol=1e-5)
    assert torch.allclose(back.camera_center, cam.camera_center, atol=1e-5)
