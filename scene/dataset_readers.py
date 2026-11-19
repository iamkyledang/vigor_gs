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
from utils.calibration_utils import camera_calibration, distort_points, fit_source_calibration


def read_model(folder):
    folder = Path(folder)
    if (folder / 'cameras.bin').exists():
        return read_intrinsics_binary(str(folder / 'cameras.bin')), read_extrinsics_binary(str(folder / 'images.bin'))
    return read_intrinsics_text(str(folder / 'cameras.txt')), read_extrinsics_text(str(folder / 'images.txt'))




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
