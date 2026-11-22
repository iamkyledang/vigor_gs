"""Where readVigorSceneInfo looks for a frame's clean (reflection-removed) image. Only the `data` test needs the dataset."""

import json
import os
import shutil

import pytest
from PIL import Image

from scene.dataset_readers import _find_clean_image, readVigorSceneInfo


def save(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 4)).save(path)
    return str(path)


def test_mirrors_the_source_image_folder_with_any_extension(tmp_path):
    expected = save(tmp_path / 'test' / 'images' / '00000002.png')
    frame = {'source_image': 'test/images/00000002.jpg', 'split': 'test'}
    assert _find_clean_image(str(tmp_path), frame, 'test', '00000002') == expected


def test_falls_back_to_the_split_folder_then_a_flat_folder(tmp_path):
    in_split = save(tmp_path / 'train' / 'images' / 'a.png')
    flat = save(tmp_path / 'b.jpg')
    assert _find_clean_image(str(tmp_path), {}, 'train', 'a') == in_split
    assert _find_clean_image(str(tmp_path), {'source_image': 'train/images/b.jpg'}, 'train', 'b') == flat


@pytest.mark.data
def test_reader_prefers_the_undistorted_copy_under_colmap(scene_dir, tmp_path):
    """readVigorSceneInfo reads <scene>/colmap/<clean_images>/ before <scene>/<clean_images>/."""
    scene = tmp_path / 'scene'
    shutil.copytree(os.path.join(scene_dir, 'colmap'), scene / 'colmap', ignore=shutil.ignore_patterns('images', '*.ply'))
    os.symlink(os.path.join(scene_dir, 'colmap', 'images'), scene / 'colmap' / 'images')
    for name in ('train', 'test'):
        os.symlink(os.path.join(scene_dir, name), scene / name)
    with open(scene / 'colmap' / 'split.json') as f:
        split = json.load(f)
    for root, color in ((scene / 'clean_image', (255, 0, 0)), (scene / 'colmap' / 'clean_image', (0, 255, 0))):
        for split_name, names in split.items():
            for name in names:
                path = root / split_name / 'images' / (os.path.splitext(name)[0] + '.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 8), color).save(path)

    info = readVigorSceneInfo(str(scene), eval=True)
    assert all(os.sep + os.path.join('colmap', 'clean_image') in c.clean_image.filename for c in info.train_cameras + info.test_cameras)


def test_does_not_take_a_frame_from_the_other_split(tmp_path):
    save(tmp_path / 'train' / 'images' / 'c.png')
    assert _find_clean_image(str(tmp_path), {'source_image': 'test/images/c.jpg'}, 'test', 'c') is None
