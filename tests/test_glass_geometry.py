"""Glass patches, first visible hit (Eq. 9), reflection (Eq. 9a), the dynamic virtual camera (Eq. 9b), extraction."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.synthetic import look_at as colmap_look_at
from utils.glass_geometry import dynamic_virtual_camera, first_hit, look_at, mirror_points, reflect, validate_patches
from utils.patch_extraction import fit_patches


def square(z, half=1.0, cx=0.0):
    return {'normal': [0, 0, 1], 'offset': -z, 'vertices': [[cx - half, -half, z], [cx + half, -half, z], [cx + half, half, z], [cx - half, half, z]]}


def view(center=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), width=33, height=25, focal=30.0):
    """A CPU stand-in for scene.cameras.Camera with what the geometry needs."""
    R, t = colmap_look_at(center, target)
    w2c = torch.eye(4)
    w2c[:3, :3] = torch.tensor(R, dtype=torch.float32)
    w2c[:3, 3] = torch.tensor(t, dtype=torch.float32)
    return SimpleNamespace(image_width=width, image_height=height, fx=focal, fy=focal, cx=width / 2, cy=height / 2,
                           world_view_transform=w2c.T.contiguous(), camera_center=torch.tensor(center, dtype=torch.float32),
                           FoVx=2 * math.atan(width / (2 * focal)), FoVy=2 * math.atan(height / (2 * focal)))


def test_validation_rejects_bad_patches():
    validate_patches([square(0.0)], device='cpu')
    bent = square(0.0)
    bent['vertices'][2][2] = 0.5
    with pytest.raises(ValueError, match='plane'):
        validate_patches([bent], device='cpu')
    with pytest.raises(ValueError, match='zero-area'):
        validate_patches([{'normal': [0, 0, 1], 'offset': 0, 'vertices': [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}], device='cpu')
    concave = {'normal': [0, 0, 1], 'offset': 0, 'vertices': [[0, 0, 0], [2, 0, 0], [2, 2, 0], [1, 0.5, 0], [0, 2, 0]]}
    with pytest.raises(ValueError, match='convex'):
        validate_patches([concave], device='cpu')
    scaled = validate_patches([{'normal': [0, 0, 2], 'offset': -4.0, 'vertices': square(2.0)['vertices']}], device='cpu')[0]
    assert torch.allclose(scaled['normal'], torch.tensor([0.0, 0.0, 1.0])) and float(scaled['offset']) == pytest.approx(-2.0)


def test_first_hit_picks_the_nearest_patch_within_its_support():
    cam = view()
    patches = validate_patches([square(0.0, half=1.5), square(1.0, half=0.3), square(5.0, half=10.0)], device='cpu')  # the last is behind the camera
    geometry = first_hit(cam, patches)
    center = geometry['patch'][12, 16]
    assert center == 1  # the small near patch in front of the large one
    assert geometry['distance'][12, 16] == pytest.approx(2.0, abs=1e-3)
    assert geometry['patch'][12, 2] == 0  # outside the small patch, inside the large one
    assert geometry['patch'][0, 0] == -1  # outside every finite support
    assert not (geometry['patch'] == 2).any()
    off = geometry['patch'] < 0
    assert torch.all(geometry['reflected'][off] == 0) and torch.all(geometry['normals'][off] == 0)


def test_reflection_obeys_the_mirror_law():
    d = torch.nn.functional.normalize(torch.tensor([[0.3, -0.2, -1.0]]), dim=-1)
    n = torch.tensor([[0.0, 0.0, 1.0]])
    r = reflect(d, n)
    assert torch.allclose(r.norm(dim=-1), torch.ones(1))
    assert torch.allclose((r * n).sum(-1), -(d * n).sum(-1))  # angle of reflection = angle of incidence
    assert torch.allclose(r[:, :2], d[:, :2])  # tangential part unchanged


def test_occlusion_drops_hits_behind_opaque_surfaces_only():
    cam = view()
    patches = validate_patches([square(0.0, half=2.0)], device='cpu')
    depth = torch.full((1, 25, 33), 10.0)  # what lies behind the glass, 7 units further
    alpha = torch.ones(1, 25, 33)
    depth[0, 10:15, 14:19] = 2.0  # an opaque object 1 unit in front of the glass (glass at depth 3)
    alpha[0, 0:3, :] = 0.1  # a transparent region: never an occluder
    depth[0, 0:3, :] = 1.0
    occlusion = {'depth': depth, 'alpha': alpha, 'tolerance': 0.02, 'alpha_min': 0.5}
    literal = first_hit(cam, patches)
    tested = first_hit(cam, patches, occlusion=occlusion)
    assert literal['visible'][12, 16] and not tested['visible'][12, 16] and tested['occluded'][12, 16]
    assert tested['patch'][12, 16] == -1 and tested['cosine'][12, 16] == 0
    assert tested['visible'][1, 16]  # low alpha: kept
    assert tested['visible'][20, 5]  # depth behind the glass: kept
    assert int(tested['occluded'].sum()) == 25


def test_look_at_is_a_proper_rotation():
    rotation = look_at(torch.tensor([1.0, 2.0, -0.5]), torch.tensor([0.0, -1.0, 0.0]))
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-6)
    assert torch.det(rotation) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(rotation[:, 2], torch.nn.functional.normalize(torch.tensor([1.0, 2.0, -0.5]), dim=0), atol=1e-6)
    assert look_at(torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])).det() == pytest.approx(1.0, abs=1e-6)


def test_dynamic_virtual_camera_pose_and_mirrored_center():
    cam = view(center=(0.5, 0.2, 3.0))
    patches = validate_patches([square(0.0, half=1.0, cx=-1.2), square(0.0, half=0.6, cx=1.0)], device='cpu')
    geometry = first_hit(cam, patches)
    for k in range(2):
        pose = dynamic_virtual_camera(cam, geometry, patches, k, epsilon=1e-3)
        x, y = pose['reference_pixel']
        assert geometry['patch'][y, x] == k
        assert torch.allclose(pose['center'], geometry['hits'][y, x] + 1e-3 * geometry['reflected'][y, x])
        assert torch.allclose(pose['rotation'][:, 2], geometry['reflected'][y, x], atol=1e-6)
        real_down = cam.world_view_transform[:3, :3][:, 1]
        assert pose['rotation'][:, 1] @ real_down > 0  # the image y axis keeps pointing down, as in the real camera
        mirrored = pose['mirrored_center']
        assert torch.allclose(mirrored, mirror_points(cam.camera_center, patches[k]['normal'], patches[k]['offset']))
        # Every reflected ray of the region passes through the mirrored center: rays are central per patch.
        region = geometry['patch'] == k
        to_center = torch.nn.functional.normalize(geometry['hits'][region] - mirrored, dim=-1)
        assert torch.allclose(to_center, geometry['reflected'][region], atol=1e-5)
    assert dynamic_virtual_camera(cam, geometry, patches + validate_patches([square(9.0, 0.1)], device='cpu'), 2) is None


def test_fit_patches_splits_coplanar_panes_and_finds_a_second_plane():
    rng = np.random.default_rng(0)
    left = np.c_[rng.uniform(-3, -1, (800, 2)), rng.normal(0, 0.002, 800)]
    right = np.c_[rng.uniform(1, 3, (800, 2)), rng.normal(0, 0.002, 800)]
    side = np.c_[rng.normal(5, 0.002, 600), rng.uniform(-1, 1, (600, 2))]
    records = fit_patches(np.concatenate([left, right, side]), threshold=0.02, min_inliers=100, max_patches=4)
    assert len(records) == 3
    normals = np.abs(np.array([r['normal'] for r in records]))
    assert sorted(np.argmax(normals, 1).tolist()) == [0, 2, 2]
    patches = validate_patches(records, device='cpu')
    areas = sorted(float(torch.cross(p['vertices'] - p['vertices'][0], p['vertices'].roll(-1, 0) - p['vertices'][0], dim=-1).sum(0).norm() / 2) for p in patches)
    assert areas[0] > 3.0 and areas[-1] < 4.1  # each pane is about 2 x 2


@pytest.mark.gpu
@pytest.mark.slow
def test_extract_glass_with_known_and_extracted_geometry(trained_interface, synthetic_scene, tmp_path):
    import extract_glass

    known = tmp_path / 'known.json'
    known.write_text(json.dumps({'patches': synthetic_scene.patches}))
    extract_glass.main(['-m', trained_interface, '--glass_geometry', str(known)])
    record = json.load(open(os.path.join(trained_interface, 'glass_geometry.json')))
    assert record['accepted'] and record['settings']['ablate_occlusion'] is False
    assert record['quality']['coverage'] > 0.95 and record['quality']['precision'] > 0.95
    assert len(os.listdir(os.path.join(trained_interface, 'glass_preview'))) > 0

    records = []
    for _ in range(2):  # the extraction is deterministic
        try:
            extract_glass.main(['-m', trained_interface, '--plane_min_inliers', '50'])
        except SystemExit:
            pass  # a 300-iteration Phase I may not pass the quality gate; the record is written either way
        records.append(json.load(open(os.path.join(trained_interface, 'glass_geometry.json'))))
    record = records[-1]
    assert record['source'].startswith('phase_i') and len(record['patches']) >= 1
    assert 0 <= record['quality']['coverage'] <= 1
    assert records[0]['patches'] == records[1]['patches']
    assert record['settings']['glass_margin'] > 0


def test_on_patches_selects_points_on_the_glass_layer_only():
    from utils.glass_geometry import on_patches

    patches = validate_patches([square(0.0, half=1.0), square(2.0, half=0.5)], device='cpu')
    points = torch.tensor([[0.2, 0.3, 0.005],   # on the first pane
                           [0.2, 0.3, 0.2],     # in front of it: an occluder candidate
                           [1.5, 0.0, 0.0],     # on its plane but outside the polygon
                           [0.1, -0.1, 1.999],  # on the second pane
                           [0.9, 0.0, 2.0]])    # outside the second pane
    assert on_patches(points, patches, margin=0.01).tolist() == [True, False, False, True, False]
