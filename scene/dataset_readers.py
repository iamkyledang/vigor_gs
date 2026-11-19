"""Paired VSGD/COLMAP reader with explicit time/split and in-memory rectification."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text, read_points3D_binary, read_points3D_text, qvec2rotmat)
from utils.graphics_utils import BasicPointCloud


def read_model(folder):
    folder = Path(folder)
    if (folder / 'cameras.bin').exists():
        return read_intrinsics_binary(str(folder / 'cameras.bin')), read_extrinsics_binary(str(folder / 'images.bin'))
    return read_intrinsics_text(str(folder / 'cameras.txt')), read_extrinsics_text(str(folder / 'images.txt'))


def camera_calibration(cam):
    p = np.asarray(cam.params, dtype=np.float64)
    if cam.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
        fx, fy, cx, cy = p[0], p[0], p[1], p[2]
        dist = np.zeros(5)
        if cam.model == 'SIMPLE_RADIAL': dist[0] = p[3]
        if cam.model == 'RADIAL': dist[:2] = p[3:5]
    elif cam.model in ('PINHOLE', 'OPENCV', 'FULL_OPENCV'):
        fx, fy, cx, cy = p[:4]
        dist = np.zeros(8 if cam.model == 'FULL_OPENCV' else 5)
        if cam.model != 'PINHOLE': dist[:len(p)-4] = p[4:]
    else:
        raise ValueError(f'Unsupported camera {cam.model}; rectify it to PINHOLE first.')
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64), dist


def readVigorSceneInfo(args):
    root = Path(args.source_path)
    colmap = root / 'colmap' if (root / 'colmap/sparse/0').exists() else root
    intrinsics, extrinsics = read_model(colmap / 'sparse/0')
    metadata_path = colmap / 'frames.json'
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text())
        records = data['frames'] if isinstance(data, dict) else data
    elif (root / 'frames.csv').exists():
        with (root / 'frames.csv').open() as stream:
            records = list(csv.DictReader(stream))
        total = max(int(r['global_index']) for r in records)
        records = [dict(r, name=Path(r['image']).stem+'.jpg', time=int(r['global_index'])/max(total,1),
                        source_image=r.get('original_image') or r['image'], source_mask=r['mask']) for r in records]
    else:
        raise FileNotFoundError('Expected frames.json or frames.csv with explicit timestamps and train/test splits.')
    records = {Path(r['name']).stem: r for r in records}
    source_folder = root / 'source_metadata/colmap/sparse/0'
    raw_intr, raw_ext = read_model(source_folder) if source_folder.exists() else (None, None)
    raw_by_name = {Path(e.name).stem: e for e in raw_ext.values()} if raw_ext else {}
    cameras = []
    alignment_path=Path(args.model_path)/'data_alignment.json'
    alignment=json.loads(alignment_path.read_text()) if alignment_path.exists() else {}
    map_cache={}
    for ext in sorted(extrinsics.values(), key=lambda e: e.name):
        stem = Path(ext.name).stem
        if stem not in records:
            raise ValueError(f'No timestamp/split metadata for {ext.name}')
        record = records[stem]
        split = record['split']
        if split not in ('train', 'test'): continue
        intr = intrinsics[ext.camera_id]
        K, target_dist = camera_calibration(intr)
        if np.any(target_dist):
            raise ValueError('Target COLMAP cameras must be undistorted.')
        if args.resolution <= 0: raise ValueError('Resolution must be a positive downsampling factor.')
        width, height = max(1, round(intr.width/args.resolution)), max(1, round(intr.height/args.resolution))
        target_K = K.copy()
        target_K[0] *= width/intr.width
        target_K[1] *= height/intr.height
        source_clean = root / split / 'images' / (stem+'.png')
        if args.clean_images:
            folder = Path(args.clean_images)
            folder = folder if folder.is_absolute() else root / folder
            choices = [folder/split/'images'/(stem+'.png'), folder/(stem+'.png')]
            source_clean = next((p for p in choices if p.exists()), choices[0])
        clean = None
        source_original = root / record.get('source_image', f'{split}/images/{stem}.jpg')
        target_original = colmap / 'images' / ext.name
        if not target_original.exists():
            raise FileNotFoundError(f'Missing undistorted ORIGINAL image: {target_original}')
        image = np.asarray(Image.open(target_original).convert('RGB'))
        image = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA)
        mask_path = root / record.get('source_mask', f'{split}/masks/{stem}.png')
        with Image.open(source_clean if source_clean.exists() else mask_path) as source:
            source_size=source.size
        cache_key=(ext.camera_id,source_size,width,height)
        if cache_key not in map_cache:
            raw_cam=raw_intr[raw_by_name[stem].camera_id] if raw_intr else intr
            raw_K,distortion=camera_calibration(raw_cam)
            needs_fit=(raw_cam.width,raw_cam.height)!=source_size
            # Some exports retain only the UNDISTORTED camera under source_metadata.
            if needs_fit:
                key=str(ext.camera_id)
                calibration=alignment.get(key)
                if calibration is None:
                    calibration=fit_source_calibration(root,colmap,intr,extrinsics,records,source_size)
                    alignment[key]=calibration
                    alignment_path.parent.mkdir(parents=True,exist_ok=True)
                    alignment_path.write_text(json.dumps(alignment,indent=2))
                if tuple(calibration['source_size'])!=source_size:
                    raise ValueError('Saved source calibration dimensions do not match the dataset.')
                cols,rows=np.meshgrid(np.arange(width),np.arange(height))
                target={'fx':target_K[0,0],'fy':target_K[1,1],'cx':target_K[0,2],'cy':target_K[1,2]}
                mapped=distort_points(np.stack([cols.ravel(),rows.ravel()],1),target,calibration['params'],calibration['model'])
                maps=(mapped[:,0].reshape(height,width).astype(np.float32),mapped[:,1].reshape(height,width).astype(np.float32))
            else:
                source_K=raw_K.copy();source_K[:2,2]-=0.5
                cv_target=target_K.copy();cv_target[:2,2]-=0.5
                maps=cv2.initUndistortRectifyMap(source_K,distortion,None,cv_target,(width,height),cv2.CV_32FC1)
            map_cache[cache_key]=maps
        map_x,map_y=map_cache[cache_key]
        def rectify(path, mask=False):
            data = np.asarray(Image.open(path).convert('L' if mask else 'RGB'))
            if data.shape[:2] != (source_size[1],source_size[0]):
                raise ValueError(f'{path}: source size {data.shape[:2]} does not match source calibration {source_size}')
            return cv2.remap(data, map_x, map_y, cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if source_clean.exists(): clean = rectify(source_clean)
        elif split == 'train': raise FileNotFoundError(f'Missing CLEAN training target: {source_clean}')
        mask = rectify(mask_path, True) > 127 if mask_path.exists() else np.zeros((height,width), bool)
        if split == 'train' and not mask_path.exists() and not args.glass_geometry:
            raise FileNotFoundError(f'Glass extraction requires the training mask: {mask_path}')
        # Exclude undistortion padding from all photometric and temporal objectives.
        valid = (map_x >= 0) & (map_y >= 0) & (map_x <= source_size[0]-1) & (map_y <= source_size[1]-1)
        R = qvec2rotmat(ext.qvec).T
        cameras.append(SimpleNamespace(uid=ext.id, sensor_id=ext.camera_id, R=R, T=np.array(ext.tvec), K=target_K,
            image=image, clean_image=clean, transparent_mask=mask, valid_mask=valid,
            image_name=stem, width=width, height=height, time=float(record['time']),
            sequence=record.get('source_clip','default'), split=split))
    train = [c for c in cameras if c.split == 'train']
    test = [c for c in cameras if c.split == 'test']
    if not train: raise ValueError('No registered training cameras.')
    centers = np.stack([-c.R @ c.T for c in train])
    center = centers.mean(0)
    extent = max(float(np.linalg.norm(centers-center,axis=-1).max())*1.1, 1e-3)
    sparse = colmap / 'sparse/0'
    if (sparse/'points3D.bin').exists(): xyz, rgb, _ = read_points3D_binary(str(sparse/'points3D.bin'))
    else: xyz, rgb, _ = read_points3D_text(str(sparse/'points3D.txt'))
    finite = np.isfinite(xyz).all(-1)
    xyz, rgb = xyz[finite], rgb[finite]
    if len(xyz) < 4: raise ValueError('Need at least four finite SfM initialization points.')
    generator = np.random.default_rng(args.seed)
    if len(xyz) > args.init_points:
        indices = generator.choice(len(xyz), args.init_points, replace=False)
        xyz, rgb = xyz[indices], rgb[indices]
    print(f'Paired dataset: {len(train)} train, {len(test)} test; {width}x{height}; {len(xyz)} SfM points')
    return SimpleNamespace(train_cameras=train, test_cameras=test,
        point_cloud=BasicPointCloud(xyz, rgb/255., np.zeros_like(xyz)),
        nerf_normalization={'radius':extent,'center':center})


# Source-camera recovery adapted from Viet Huynh's undistort_utils.py.
"""
Warp from the distorted source images of a scene to its undistorted COLMAP images.

The VSGD and Wayve scenes keep only the undistorted PINHOLE model in colmap/sparse/0/; the distorted camera
that produced the COLMAP images is gone. Masks and clean images are made on the source images, so they do not
line up with the COLMAP images. This module recovers the distorted camera from point matches between a source
image and its COLMAP image, then builds the warp that maps one onto the other.

Conventions follow COLMAP (WarpImageBetweenCameras): pixel (i, j) of an image has its center at (i + 0.5, j + 0.5)
in camera coordinates. An undistorted pixel (u', v') becomes the normalized point x = (u' - cx') / fx',
y = (v' - cy') / fy', r = sqrt(x^2 + y^2), and lands on the source pixel u = fx s x + cx, v = fy s y + cy, where

    radial   (COLMAP RADIAL / SIMPLE_RADIAL with fx, fy):  s = 1 + k1 r^2 + k2 r^4
    fisheye  (COLMAP OPENCV_FISHEYE):                      s = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) / r,
                                                           theta = atan(r)

VSGD phone videos are mildly distorted (radial fits to ~0.15 px); the WayveScenes101 cameras are wide-angle
(~120 degrees) and need the fisheye model.
"""

from scipy.optimize import least_squares

MODELS = {
    'radial': ('fx', 'fy', 'cx', 'cy', 'k1', 'k2'),
    'fisheye': ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'),
}


def distort_points(points, undistorted, params, model='radial'):
    """
    Map pixel indices of the undistorted image to pixel indices of the source image.

    points: (N, 2) array of (column, row) indices in the undistorted image.
    undistorted: dict with fx, fy, cx, cy of the undistorted PINHOLE camera (COLMAP convention).
    params: the distorted camera, in the order of MODELS[model].
    """
    fx, fy, cx, cy, *k = params
    x = (points[:, 0] + 0.5 - undistorted['cx']) / undistorted['fx']
    y = (points[:, 1] + 0.5 - undistorted['cy']) / undistorted['fy']
    r2 = x * x + y * y
    if model == 'radial':
        s = 1.0 + k[0] * r2 + k[1] * r2 * r2
    elif model == 'fisheye':
        r = np.sqrt(r2)
        theta = np.arctan(r)
        t2 = theta * theta
        theta_d = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
        s = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
    else:
        raise ValueError(f'Unknown distortion model {model!r}; expected one of {list(MODELS)}')
    return np.stack([fx * s * x + cx - 0.5, fy * s * y + cy - 0.5], axis=1)


def initial_params(undistorted, source_size, model='radial', focal_scale=1.0):
    """No distortion, the focal length of the undistorted camera (times focal_scale), principal point at the source center."""
    width, height = source_size
    base = [undistorted['fx'] * focal_scale, undistorted['fy'] * focal_scale, width / 2.0, height / 2.0]
    return np.array(base + [0.0] * (len(MODELS[model]) - 4))


def fit_distortion(undist_points, source_points, undistorted, source_size, model='radial', focal_scales=(1.0, 0.7, 1.4), stages=((20.0, 30.0), (5.0, 8.0), (1.0, 3.0), (1.0, 3.0))):
    """
    Fit the distorted camera so that distort_points(undist_points) lands on source_points.

    The matches may contain outliers and the warp may move points by hundreds of pixels (a fisheye), so the fit runs
    coarse to fine: each stage is a robust (soft L1) fit with the given scale, after which only the matches within the
    given distance are kept for the next stage. Several initial focal lengths are tried and the one keeping the most
    matches within 3 px wins. Returns the parameters and the residual distance (pixels) of every match.
    """
    undist_points = np.asarray(undist_points, dtype=np.float64)
    source_points = np.asarray(source_points, dtype=np.float64)
    if len(undist_points) < 20:
        raise ValueError(f'Only {len(undist_points)} matches; too few to fit a distortion model.')

    def residuals(v, a, b):
        return (distort_points(a, undistorted, v, model) - b).ravel()

    best = None
    for focal_scale in focal_scales:
        v = initial_params(undistorted, source_size, model, focal_scale)
        keep = np.ones(len(undist_points), dtype=bool)
        try:
            for f_scale, trim in stages:
                v = least_squares(residuals, v, args=(undist_points[keep], source_points[keep]), loss='soft_l1', f_scale=f_scale, x_scale='jac').x
                dist = np.linalg.norm(distort_points(undist_points, undistorted, v, model) - source_points, axis=1)
                keep = dist < trim
                if keep.sum() < 20:
                    break
        except (ValueError, np.linalg.LinAlgError):
            continue
        inliers = int((dist < 3.0).sum())
        if best is None or inliers > best[0]:
            best = (inliers, v, dist)
    if best is None or best[0] < 20:
        raise ValueError(f'Only {0 if best is None else best[0]} matches agree with the {model} fit; the source and COLMAP images may not correspond.')
    return best[1], best[2]


def build_remap(undistorted, params, model='radial'):
    """cv2.remap maps (float32, H' x W') that sample the source image on the undistorted COLMAP pixel grid."""
    width, height = undistorted['width'], undistorted['height']
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    src = distort_points(np.stack([cols.ravel(), rows.ravel()], axis=1), undistorted, params, model)
    return src[:, 0].reshape(height, width).astype(np.float32), src[:, 1].reshape(height, width).astype(np.float32)


def warp(image, maps, nearest=False):
    """Warp a source image (H x W [x C] array) to the COLMAP grid. Nearest keeps a mask's label values."""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(image, maps[0], maps[1], interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


def match_points(source_gray, colmap_gray, ratio=0.75, max_features=8000):
    """SIFT matches (ratio test) between a source image and its COLMAP image, as (colmap_points, source_points)."""
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp_s, des_s = sift.detectAndCompute(source_gray, None)
    kp_c, des_c = sift.detectAndCompute(colmap_gray, None)
    if des_s is None or des_c is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_c, des_s, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    pc = np.array([kp_c[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    ps = np.array([kp_s[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    return pc, ps



def fit_source_calibration(root,colmap,intr,extrinsics,records,source_size):
    candidates=[e for e in extrinsics.values() if e.camera_id==intr.id and records[Path(e.name).stem]['split']=='train']
    candidates.sort(key=lambda e:e.name)
    indices=np.unique(np.linspace(0,len(candidates)-1,min(8,len(candidates))).round().astype(int))
    if len(indices)<2: raise ValueError('Need at least two training images to recover missing source calibration.')
    K,_=camera_calibration(intr)
    target={'fx':K[0,0],'fy':K[1,1],'cx':K[0,2],'cy':K[1,2]}
    pairs=[]
    for index in indices:
        ext=candidates[index];record=records[Path(ext.name).stem]
        source=cv2.imread(str(root/record['source_image']),cv2.IMREAD_GRAYSCALE)
        image=cv2.imread(str(colmap/'images'/ext.name),cv2.IMREAD_GRAYSCALE)
        if source is None: raise FileNotFoundError(f'Missing original source JPG for calibration: {record["source_image"]}')
        pairs.append(match_points(source,image,max_features=4000))
    # Hold out the last selected TRAINING view to check the recovered warp.
    pc=np.concatenate([a for a,b in pairs[:-1]]);ps=np.concatenate([b for a,b in pairs[:-1]])
    fits=[]
    for model in ('radial','fisheye'):
        try:
            params,residual=fit_distortion(pc,ps,target,source_size,model)
            check=np.linalg.norm(distort_points(pairs[-1][0],target,params,model)-pairs[-1][1],axis=1)
            inliers=check<3
            median=float(np.median(check[inliers])) if inliers.any() else float('inf')
            fits.append((median,model,params,residual,check))
        except ValueError: continue
    if not fits: raise ValueError('Cannot recover source-to-COLMAP calibration from original JPG matches.')
    fits.sort(key=lambda item:item[0])
    median,model,params,residual,check=fits[0]
    if median>1 or (check<3).sum()<20:
        raise ValueError(f'Source calibration failed held-out training-view validation: median={median}, inliers={(check<3).sum()}')
    record={'model':model,'params':params.tolist(),'source_size':list(source_size),
        'heldout_median_px':median,'heldout_inliers':int((check<3).sum()),'training_frames':len(indices),
        'target_size':[intr.width,intr.height]}
    print('Recovered source calibration:',record,flush=True)
    return record
