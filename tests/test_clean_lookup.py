"""Where readVigorSceneInfo looks for a frame's clean (reflection-removed) image. No GPU or dataset needed."""

from PIL import Image

from scene.dataset_readers import _find_clean_image


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


def test_does_not_take_a_frame_from_the_other_split(tmp_path):
    save(tmp_path / 'train' / 'images' / 'c.png')
    assert _find_clean_image(str(tmp_path), {'source_image': 'test/images/c.jpg'}, 'test', 'c') is None
