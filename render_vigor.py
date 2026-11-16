"""
VIGOR-GS inference: render the trained model for the training and test cameras.

For a camera C and time t, with every learned parameter fixed:

    B     = R_2D(D_phi_I(G_I, t), C)
    R     = R_3D(D_phi_R(G_R, t), V(C, Pi*))
    I_hat = (1 - alpha) B + alpha R

Images are written under <model_path>/{train,test}/ours_<iteration>/ in the layout metrics.py reads:

    renders/               the composite I_hat
    gt/                    the original image I
    transparent_masks/     the glass mask M
    vis/                   direct B, reflection R, its contribution alpha * R, alpha and the rendered glass map

    python render_vigor.py -m output/<scene>
    python metrics.py -m output/<scene>
"""

import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer.vigor import render_composite, render_direct
from utils.color_utils import linear2srgb
from utils.fresnel import schlick_f0
from utils.general_utils import safe_state
from utils.plane_utils import load_plane
from utils.render_utils import save_img_u8
from utils.vigor_utils import PLANE_FILE, REFLECTION_DIR, load_interface, load_reflection


@torch.no_grad()
def render_set(out_dir, cameras, model, pipe, background, linear_rgb, use_gt_mask=False):
    """Render every camera of one split and save the images. `model` is the dict built by `load_model`."""
    folders = {name: os.path.join(out_dir, name) for name in ('renders', 'gt', 'transparent_masks', 'vis')}
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    def to_display(image):
        """(C, H, W) tensor -> (H, W, C) or (H, W) array in display (sRGB) space."""
        image = image.clamp(0.0, 1.0)
        if linear_rgb and image.shape[0] == 3:
            image = linear2srgb(image)
        return image.permute(1, 2, 0).squeeze(-1).cpu().numpy()

    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, cam in enumerate(tqdm(cameras, desc=f'Rendering {os.path.basename(os.path.dirname(out_dir))}')):
            direct_out = render_direct(cam, model['interface'], model['interface_deform'], pipe, background)
            glass_map = cam.gt_transparent_mask.cuda().float() if use_gt_mask else direct_out['glass_map']
            out = render_composite(
                cam,
                direct_out['direct'],
                glass_map,
                model['reflection'],
                pipe,
                model['normal'],
                model['offset'],
                reflection_offsets=model['reflection_deform'].step(model['reflection'].get_xyz, cam.time),
                f0=model['f0'],
                correction=model['correction'],
            )

            name = '{0:05d}.png'.format(idx)
            executor.submit(save_img_u8, to_display(out['composite']), os.path.join(folders['renders'], name))
            executor.submit(save_img_u8, to_display(cam.original_image[:3]), os.path.join(folders['gt'], name))
            executor.submit(save_img_u8, cam.gt_transparent_mask.squeeze(0).float().cpu().numpy(), os.path.join(folders['transparent_masks'], name))
            for label, image in (
                ('direct', out['direct']),
                ('reflection', out['reflection']),
                ('reflection_contribution', out['alpha'] * out['reflection']),
                ('alpha', out['alpha']),
                ('glass_map', out['glass_map']),
            ):
                executor.submit(save_img_u8, to_display(image), os.path.join(folders['vis'], f'{label}_{name}'))


def load_model(dataset, iteration=-1, interface_iteration=-1):
    """Everything inference needs. Returns (scene, model dict, reflection iteration)."""
    scene, interface, interface_deform = load_interface(dataset, interface_iteration)
    reflection, reflection_deform, correction, loaded_iter = load_reflection(dataset, iteration)
    normal, offset, _ = load_plane(os.path.join(dataset.model_path, PLANE_FILE))
    model = {
        'interface': interface,
        'interface_deform': interface_deform,
        'reflection': reflection,
        'reflection_deform': reflection_deform,
        'correction': correction,
        'normal': normal,
        'offset': offset,
        'f0': schlick_f0(dataset.eta_air, dataset.eta_glass),
    }
    return scene, model, loaded_iter


def render_sets(dataset, pipe, iteration=-1, interface_iteration=-1, skip_train=False, skip_test=False, use_gt_mask=False):
    scene, model, loaded_iter = load_model(dataset, iteration, interface_iteration)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    for split, cameras, skip in (('train', scene.getTrainCameras(), skip_train), ('test', scene.getTestCameras(), skip_test)):
        if skip or not cameras:
            continue
        render_set(os.path.join(dataset.model_path, split, 'ours_{}'.format(loaded_iter)), cameras, model, pipe, background, dataset.linear_rgb, use_gt_mask)
    return loaded_iter


if __name__ == '__main__':
    parser = ArgumentParser(description='VIGOR-GS: render the trained model')
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int, help='Phase II iteration to load (default: the latest)')
    parser.add_argument('--interface_iteration', default=-1, type=int, help='Phase I iteration to load (default: the latest)')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--gt_glass_mask', action='store_true', help='Gate the reflection with the ground-truth glass mask instead of the rendered glass map')
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser, cfg_subdir=REFLECTION_DIR)
    print('Rendering ' + args.model_path)

    safe_state(args.quiet)
    render_sets(model_params.extract(args), pipeline.extract(args), args.iteration, args.interface_iteration, args.skip_train, args.skip_test, args.gt_glass_mask)
