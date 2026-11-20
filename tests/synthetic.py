"""
A small, physically consistent glass scene written in the aligned VIGOR-GS layout (scene/dataset_readers.py).

    z = 0, |x|, |y| <= 1        glass patch (the only glass)
    z = 0, elsewhere in |x|,|y| <= 4   opaque wall around the window
    z = -1.5                    textured interior, seen through the glass
    z = +6                      textured surroundings, seen in the glass by reflection
    optional pillar             opaque box in front of the glass (occlusion tests)

Cameras sit on an arc in front of the window and look at its center. Pixel colors are evaluated analytically in
linear RGB: clean = what the glass transmits (no reflection), original = (1 - F) clean + F reflection on glass pixels,
with Schlick's F for `eta_glass`; images are written in sRGB. The glass mask is the set of pixels whose first hit is
the glass. SfM points are sampled on the wall, the interior and the glass.
"""

import json
import os

import numpy as np
from PIL import Image

GLASS_HALF = 1.0
WALL_HALF = 4.0
INTERIOR_Z = -1.5
SURROUND_Z = 6.0


def srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


def interior(x, y):
    return np.stack([0.5 + 0.4 * np.sin(3 * x), 0.5 + 0.4 * np.cos(2 * y), 0.4 + 0.3 * np.sin(x + y)], -1)


def surroundings(x, y):
    return np.stack([0.6 + 0.35 * np.sin(1.3 * y), 0.3 + 0.25 * np.cos(1.7 * x), 0.8 + 0.15 * np.sin(x - y)], -1)


def wall(x, y):
    return np.stack([0.25 + 0.1 * np.sin(5 * x), 0.2 + 0.1 * np.cos(5 * y), 0.15 + 0.0 * x], -1)


def look_at(center, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """COLMAP world-to-camera (R, t): x right, y down, z forward."""
    center = np.asarray(center, float)
    z = np.asarray(target, float) - center
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.stack([x, y, z], 1)
    R = c2w.T
    return R, -R @ center


def rotmat2qvec(R):
    from scene.colmap_loader import rotmat2qvec as convert

    return convert(R)


def plane_hit(origins, directions, z):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (z - origins[..., 2]) / directions[..., 2]
    s = np.where(np.isfinite(s) & (s > 1e-6), s, np.inf)
    return s, origins + s[..., None] * directions


def render(R, t, K, width, height, eta_glass=1.5, pillar=None):
    """(original, clean, mask) in linear RGB / bool for one camera. `pillar`: (xmin, xmax, ymin, ymax, z) or None."""
    c2w = R.T
    center = -R.T @ t
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    local = np.stack([(cols - K[0, 2]) / K[0, 0], (rows - K[1, 2]) / K[1, 1], np.ones_like(cols)], -1)
    d = local @ c2w.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(center, d.shape)

    s0, p0 = plane_hit(o, d, 0.0)
    on_glass = (np.abs(p0[..., 0]) <= GLASS_HALF) & (np.abs(p0[..., 1]) <= GLASS_HALF) & np.isfinite(s0)
    on_wall = ~on_glass & (np.abs(p0[..., 0]) <= WALL_HALF) & (np.abs(p0[..., 1]) <= WALL_HALF) & np.isfinite(s0)
    _, pin = plane_hit(o, d, INTERIOR_Z)
    clean = np.full(d.shape, 0.05)
    clean[on_wall] = wall(p0[..., 0], p0[..., 1])[on_wall]
    clean[on_glass] = interior(pin[..., 0], pin[..., 1])[on_glass]

    r = d - 2 * d[..., 2:3] * np.array([0.0, 0.0, 1.0])
    _, pout = plane_hit(p0, r, SURROUND_Z)
    reflection = surroundings(pout[..., 0], pout[..., 1])
    f0 = ((1.0 - eta_glass) / (1.0 + eta_glass)) ** 2
    cos = np.abs(d[..., 2])
    fresnel = f0 + (1 - f0) * (1 - cos) ** 5
    original = clean.copy()
    original[on_glass] = ((1 - fresnel[..., None]) * clean + fresnel[..., None] * reflection)[on_glass]
    mask = on_glass

    if pillar is not None:
        xmin, xmax, ymin, ymax, z = pillar
        sp, pp = plane_hit(o, d, z)
        hit = (pp[..., 0] >= xmin) & (pp[..., 0] <= xmax) & (pp[..., 1] >= ymin) & (pp[..., 1] <= ymax) & np.isfinite(sp) & (sp < s0)
        color = np.array([0.9, 0.85, 0.1])
        original[hit] = color
        clean[hit] = color
        mask = mask & ~hit
    return original, clean, mask


def write_scene(root, num_views=12, width=64, height=48, eta_glass=1.5, pillar=None, num_points=3000, seed=0, sequences=1):
    """Write the scene under `root`; returns the ground-truth glass patch and the camera list."""
    rng = np.random.default_rng(seed)
    colmap = os.path.join(root, 'colmap')
    for sub in ('images', 'masks', 'clean_image/train/images', 'clean_image/test/images', 'sparse/0'):
        os.makedirs(os.path.join(colmap, sub), exist_ok=True)
    focal = 0.9 * width
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])

    frames, split, images = [], {'train': [], 'test': []}, []
    for i in range(num_views):
        phi = np.deg2rad(-35 + 70 * i / max(num_views - 1, 1))
        center = (3.2 * np.sin(phi), 0.4 * np.cos(2 * phi), 3.2 * np.cos(phi))
        R, t = look_at(center)
        original, clean, mask = render(R, t, K, width, height, eta_glass, pillar)
        name = f'{i:08d}.png'
        part = 'test' if i % 4 == 2 else 'train'
        Image.fromarray((srgb(original) * 255).round().astype(np.uint8)).save(os.path.join(colmap, 'images', name))
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(colmap, 'masks', name))
        Image.fromarray((srgb(clean) * 255).round().astype(np.uint8)).save(os.path.join(colmap, 'clean_image', part, 'images', name))
        sequence = f'clip{i * sequences // num_views}'
        frames.append({'name': name, 'time': i / max(num_views - 1, 1), 'split': part, 'source_clip': sequence})
        split[part].append(name)
        images.append((i + 1, rotmat2qvec(R), t, name))

    with open(os.path.join(colmap, 'frames.json'), 'w') as f:
        json.dump({'frames': frames}, f)
    with open(os.path.join(colmap, 'split.json'), 'w') as f:
        json.dump(split, f)
    with open(os.path.join(colmap, 'sparse/0/cameras.txt'), 'w') as f:
        f.write(f'1 PINHOLE {width} {height} {focal} {focal} {width / 2} {height / 2}\n')
    with open(os.path.join(colmap, 'sparse/0/images.txt'), 'w') as f:
        for image_id, q, t, name in images:
            f.write(f'{image_id} {" ".join(map(str, q))} {" ".join(map(str, t))} 1 {name}\n\n')

    n_wall, n_in = num_points // 2, num_points // 3
    n_glass = num_points - n_wall - n_in
    xy = rng.uniform(-WALL_HALF, WALL_HALF, (4 * n_wall, 2))
    xy = xy[(np.abs(xy) > GLASS_HALF).any(-1)][:n_wall]
    wall_points = np.c_[xy, np.zeros(len(xy))]
    xy_in = rng.uniform(-2.0, 2.0, (n_in, 2))
    interior_points = np.c_[xy_in, np.full(n_in, INTERIOR_Z)]
    glass_points = np.c_[rng.uniform(-GLASS_HALF, GLASS_HALF, (n_glass, 2)), np.zeros(n_glass)]
    colors = np.concatenate([wall(*wall_points[:, :2].T), interior(*interior_points[:, :2].T), np.full((n_glass, 3), 0.5)])
    points = np.concatenate([wall_points, interior_points, glass_points])
    with open(os.path.join(colmap, 'sparse/0/points3D.txt'), 'w') as f:
        for j, (p, c) in enumerate(zip(points, (srgb(colors) * 255).round().astype(int))):
            f.write(f'{j + 1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0.5\n')

    patch = {'normal': [0.0, 0.0, 1.0], 'offset': 0.0,
             'vertices': [[-GLASS_HALF, -GLASS_HALF, 0.0], [GLASS_HALF, -GLASS_HALF, 0.0], [GLASS_HALF, GLASS_HALF, 0.0], [-GLASS_HALF, GLASS_HALF, 0.0]]}
    return {'patches': [patch], 'K': K, 'width': width, 'height': height}
