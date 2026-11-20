"""
Reader for the aligned VIGOR-GS scene layout.

    <scene>/colmap/images/<name>         original images I_t, undistorted onto the COLMAP PINHOLE cameras
    <scene>/colmap/sparse/0/             COLMAP model: intrinsics, poses and the SfM points that initialize G_I
    <scene>/colmap/frames.json           per frame: normalized time t in [0, 1], split, source_clip (the sequence)
    <scene>/colmap/split.json            optional {'train': [names], 'test': [names]}, used when a frame has no split
    <scene>/colmap/masks/<stem>.png      glass masks M_t, aligned with colmap/images
    <scene>/colmap/<clean_images>/       clean targets I_t^clean, aligned: <split>/images/<stem>.* or <stem>.*
    <scene>/colmap/source_warp.json      optional distortion model of the source images (undistort_sources.py)

Masks and clean images are made in the coordinates of the distorted source images. `undistort_sources.py` warps
them onto the COLMAP grid once per scene, so this reader never warps anything. Undistorting a wide-angle source
leaves pixels that no source pixel maps to (black borders of the COLMAP images). When `source_warp.json` exists,
those pixels are marked invalid and excluded from every loss and metric.

The reader only resolves paths and calibration; `scene/cameras.py` decodes the pixels at the training resolution.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, read_points3D_binary, read_points3D_text
from utils.graphics_utils import BasicPointCloud

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')


@dataclass
class CameraInfo:
    uid: int  # COLMAP image id
    sensor_id: int  # COLMAP camera id
    image_name: str  # file stem
    split: str
    time: float  # normalized timestamp in [0, 1]
    sequence: str  # source clip; motion loss pairs neighbors only within one sequence and sensor
    R: np.ndarray  # camera-to-world rotation (3DGS convention: transposed COLMAP rotation)
    T: np.ndarray  # world-to-camera translation
    width: int  # full-resolution size and intrinsics; cameras.py scales them to the training resolution
    height: int
    fx: float
    fy: float
    cx: float  # principal point, COLMAP convention (pixel centers at +0.5)
    cy: float
    image_path: str
    mask_path: Optional[str]
    clean_path: Optional[str]
    valid: Optional[np.ndarray]  # (H, W) bool at full resolution, None when every pixel is valid


@dataclass
class SceneInfo:
    train_cameras: list
    test_cameras: list
    point_cloud: BasicPointCloud
    center: np.ndarray  # mean training camera center
    extent: float  # 1.1 x the largest distance of a training camera from the center (3DGS getNerfppNorm)


def read_colmap_model(sparse_dir):
    if os.path.exists(os.path.join(sparse_dir, 'cameras.bin')):
        return read_intrinsics_binary(os.path.join(sparse_dir, 'cameras.bin')), read_extrinsics_binary(os.path.join(sparse_dir, 'images.bin'))
    return read_intrinsics_text(os.path.join(sparse_dir, 'cameras.txt')), read_extrinsics_text(os.path.join(sparse_dir, 'images.txt'))


def read_colmap_points(sparse_dir):
    if os.path.exists(os.path.join(sparse_dir, 'points3D.bin')):
        xyz, rgb, _ = read_points3D_binary(os.path.join(sparse_dir, 'points3D.bin'))
    else:
        xyz, rgb, _ = read_points3D_text(os.path.join(sparse_dir, 'points3D.txt'))
    finite = np.isfinite(xyz).all(-1)
    return xyz[finite], rgb[finite]


def pinhole_intrinsics(intr):
    """(fx, fy, cx, cy) of an undistorted COLMAP camera; distorted models are refused."""
    if intr.model == 'SIMPLE_PINHOLE':
        f, cx, cy = intr.params[:3]
        return float(f), float(f), float(cx), float(cy)
    if intr.model == 'PINHOLE':
        fx, fy, cx, cy = intr.params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f'COLMAP camera {intr.id} is {intr.model}; VIGOR-GS needs undistorted PINHOLE cameras (see undistort_sources.py)')


def _find_by_stem(folder, stem):
    for extension in IMAGE_EXTENSIONS:
        path = os.path.join(folder, stem + extension)
        if os.path.exists(path):
            return path
    return None


def find_clean_image(clean_dir, frame, split, stem):
    """
    Path of a frame's clean image under `clean_dir`, or None. Tried in order:

        <clean_dir>/<dir of frame['source_image']>/<stem>.*   mirror of the source layout, e.g. train/images/<stem>.png
        <clean_dir>/<split>/images/<stem>.*
        <clean_dir>/<stem>.*                                  a flat folder
    """
    folders = []
    if frame.get('source_image'):
        folders.append(os.path.join(clean_dir, os.path.dirname(frame['source_image'])))
    if split:
        folders.append(os.path.join(clean_dir, split, 'images'))
    folders.append(clean_dir)
    for folder in folders:
        found = _find_by_stem(folder, stem)
        if found is not None:
            return found
    return None


def clean_image_dir(scene, clean_images):
    """The clean folder: absolute, or relative to colmap/ (aligned copies) and else to the scene."""
    if os.path.isabs(clean_images):
        return clean_images
    aligned = os.path.join(scene, 'colmap', clean_images)
    return aligned if os.path.isdir(aligned) else os.path.join(scene, clean_images)


def valid_regions(colmap_dir):
    """
    {camera id: (H, W) bool} of the COLMAP pixels that some source pixel maps to, from source_warp.json; {} when the
    scene has no warp record (its COLMAP images are the source images, so every pixel is valid).
    """
    path = os.path.join(colmap_dir, 'source_warp.json')
    if not os.path.exists(path):
        return {}
    from utils.undistort_utils import MODELS, build_remap

    with open(path) as f:
        record = json.load(f)
    regions = {}
    for camera_id, camera in record['cameras'].items():
        # Older records keep the model at the top level and one focal length 'f' for both axes.
        model = camera.get('model', record.get('model')).lower()
        named = dict(camera['params'])
        if 'f' in named:
            named.setdefault('fx', named['f'])
            named.setdefault('fy', named['f'])
        params = [named[name] for name in MODELS[model]]
        map_x, map_y = build_remap(camera['undistorted'], params, model)
        width, height = camera['source_size']
        # cv2.remap samples pixel indices: index i covers [i - 0.5, i + 0.5].
        regions[int(camera_id)] = (map_x >= -0.5) & (map_y >= -0.5) & (map_x <= width - 0.5) & (map_y <= height - 0.5)
    return regions


def readVigorSceneInfo(source_path, clean_images='clean_image', ignore_clean_images=False, init_points=0, seed=0):
    """
    Cameras of the train and test splits and the SfM point cloud.

    Every training frame needs a glass mask and a clean image. `ignore_clean_images` (the no-separation ablation)
    skips the clean folder: Phase I then fits the original images. Test frames without a clean image are allowed;
    they are only scored against the original images. `init_points` > 0 subsamples the SfM points.
    """
    colmap_dir = os.path.join(source_path, 'colmap')
    sparse_dir = os.path.join(colmap_dir, 'sparse', '0')
    intrinsics, extrinsics = read_colmap_model(sparse_dir)
    with open(os.path.join(colmap_dir, 'frames.json')) as f:
        frames = {frame['name']: frame for frame in json.load(f)['frames']}
    split_of = {}
    split_path = os.path.join(colmap_dir, 'split.json')
    if os.path.exists(split_path):
        with open(split_path) as f:
            split_of = {name: split for split, names in json.load(f).items() for name in names}
    clean_dir = clean_image_dir(source_path, clean_images)
    regions = valid_regions(colmap_dir)

    cameras, skipped, missing = [], [], []
    for extr in sorted(extrinsics.values(), key=lambda e: e.name):
        name = os.path.basename(extr.name)
        frame = frames.get(name)
        split = (frame or {}).get('split') or split_of.get(name)
        if frame is None or split not in ('train', 'test'):
            skipped.append(name)  # e.g. the other cameras of a rig, kept in the COLMAP model only for SfM
            continue
        stem = os.path.splitext(name)[0]
        intr = intrinsics[extr.camera_id]
        fx, fy, cx, cy = pinhole_intrinsics(intr)

        mask_path = os.path.join(colmap_dir, 'masks', stem + '.png')
        if not os.path.exists(mask_path):
            mask_path = None
            if split == 'train':
                raise FileNotFoundError(f'No glass mask for training frame {name} in {os.path.join(colmap_dir, "masks")} (run undistort_sources.py)')
        clean_path = None if ignore_clean_images else find_clean_image(clean_dir, frame, split, stem)
        if clean_path is None and split == 'train' and not ignore_clean_images:
            missing.append(name)

        valid = regions.get(extr.camera_id)
        if valid is not None and valid.shape != (intr.height, intr.width):
            raise ValueError(f'source_warp.json camera {extr.camera_id} is {valid.shape[::-1]}, the COLMAP camera is {intr.width}x{intr.height}')
        cameras.append(CameraInfo(
            uid=extr.id, sensor_id=extr.camera_id, image_name=stem, split=split, time=float(frame['time']),
            sequence=str(frame.get('source_clip') or 'default'), R=np.transpose(qvec2rotmat(extr.qvec)), T=np.array(extr.tvec),
            width=intr.width, height=intr.height, fx=fx, fy=fy, cx=cx, cy=cy,
            image_path=os.path.join(colmap_dir, 'images', name), mask_path=mask_path, clean_path=clean_path, valid=valid))
    if missing:
        raise FileNotFoundError(f'{len(missing)} training frames have no clean image in {clean_dir}, e.g. {missing[0]}. '
                                'Provide them (--clean_images) or train the no-separation ablation (--ignore_clean_images).')
    if skipped:
        print(f'[ INFO ] {len(skipped)} COLMAP images are not frames of this scene and are skipped, e.g. {skipped[0]}')

    train = [c for c in cameras if c.split == 'train']
    test = [c for c in cameras if c.split == 'test']
    if not train:
        raise ValueError(f'No training cameras in {source_path}')
    centers = np.stack([-c.R @ c.T for c in train])
    center = centers.mean(0)
    extent = max(float(np.linalg.norm(centers - center, axis=-1).max()) * 1.1, 1e-3)

    xyz, rgb = read_colmap_points(sparse_dir)
    if len(xyz) < 4:
        raise ValueError('Need at least four finite SfM points to initialize the interface field.')
    if 0 < init_points < len(xyz):
        keep = np.random.default_rng(seed).choice(len(xyz), init_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    point_cloud = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))
    print(f'[ INFO ] {source_path}: {len(train)} train / {len(test)} test frames, {len(xyz)} SfM points, '
          f'{"clean images ignored" if ignore_clean_images else "clean images from " + clean_dir}')
    return SceneInfo(train_cameras=train, test_cameras=test, point_cloud=point_cloud, center=center, extent=extent)
