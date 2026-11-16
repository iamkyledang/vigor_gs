"""
VIGOR-GS Phase II: dynamic reflection reconstruction.

Everything from Phase I is frozen: the interface field G_I, its deformation MLP D_phi_I and the glass plane Pi*.
The reflection field G_R (3D Gaussians), its deformation MLP D_phi_R and, optionally, the reflectance correction
h_psi are fitted to the original images:

    R_t   = R_3D(D_phi_R(G_R, t), V(C_t, Pi*))                       reflection through the virtual camera
    I_hat = (1 - alpha_t) B_t + alpha_t R_t                          alpha_t: Fresnel weight on the glass
    L_R   = L_rgb(I_hat_t, I_t) + lambda_m * L_motion + lambda_o * L_opt

Reads <model_path>/interface and <model_path>/glass_plane.json, writes <model_path>/reflection.

    python train_reflection.py -m output/<scene>

Ablations (model parameters, see arguments/__init__.py): --ablate_virtual_camera, --ablate_fresnel and
--shared_reflection_field, which has no G_R at all and renders the frozen interface through the virtual camera.
"""

import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, ReflectionOptimizationParams, get_combined_args
from gaussian_renderer.vigor import render_composite, render_direct
from scene.deform_model import DeformModel
from scene.reflection_model import ReflectionGaussianModel
from train_interface import time_noise_std
from utils.fresnel import BlendModel, optical_loss
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.plane_utils import load_plane
from utils.vigor_utils import INTERFACE_DIR, PLANE_FILE, REFLECTION_DIR, load_interface, save_reflection

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def reflection_losses(out, viewpoint_cam, opt, offsets=None):
    """The terms of L_R for one view, as a dict of scalar tensors."""
    gt_image = viewpoint_cam.original_image[:3].cuda()
    prediction = out['composite']
    losses = {'rgb': (1.0 - opt.lambda_dssim) * l1_loss(prediction, gt_image) + opt.lambda_dssim * (1.0 - ssim(prediction, gt_image))}
    if offsets is not None and opt.lambda_motion > 0:
        losses['motion'] = opt.lambda_motion * offsets[0].abs().mean()
    if out['h'] is not None and opt.lambda_opt > 0:
        losses['opt'] = opt.lambda_opt * optical_loss(out['h'])
    return losses


class DirectCache:
    """
    The frozen interface's direct image B_t and glass map for each training view, rendered once.

    Phase I never changes during Phase II, so there is no reason to rasterize it again every iteration.
    """

    def __init__(self, cameras, interface, interface_deform, pipe, background, use_gt_mask=False):
        self.entries = {}
        for cam in tqdm(cameras, desc='Rendering the frozen interface'):
            out = render_direct(cam, interface, interface_deform, pipe, background)
            glass_map = cam.gt_transparent_mask.float() if use_gt_mask else out['glass_map']
            self.entries[cam.uid] = (out['direct'].to(cam.data_device), glass_map.to(cam.data_device))

    def __call__(self, cam):
        direct, glass_map = self.entries[cam.uid]
        return direct.cuda(), glass_map.cuda()


@torch.no_grad()
def evaluate_composite(cameras, interface, interface_deform, reflection, reflection_deform, pipe, background, normal, offset, blend, use_gt_mask=False, use_virtual_camera=True):
    """
    Mean PSNR against the original images of the composite and of the direct image alone, over the whole image
    and over the glass region. The gap between the two is what the reflection explains.
    """
    stats = {'psnr': [], 'psnr_direct': [], 'glass_psnr': [], 'glass_psnr_direct': []}
    for cam in cameras:
        direct_out = render_direct(cam, interface, interface_deform, pipe, background)
        glass_map = cam.gt_transparent_mask.cuda().float() if use_gt_mask else direct_out['glass_map']
        offsets = reflection_deform.step(reflection.get_xyz, cam.time)
        out = render_composite(cam, direct_out['direct'], glass_map, reflection, pipe, normal, offset, reflection_offsets=offsets, blend=blend, use_virtual_camera=use_virtual_camera)
        gt_image = cam.original_image[:3].cuda()
        composite, direct = out['composite'].clamp(0.0, 1.0), out['direct'].clamp(0.0, 1.0)
        stats['psnr'].append(psnr(composite, gt_image).mean().item())
        stats['psnr_direct'].append(psnr(direct, gt_image).mean().item())
        mask = cam.gt_transparent_mask.cuda()
        if mask.any():
            stats['glass_psnr'].append(psnr(composite * mask, gt_image * mask).mean().item())
            stats['glass_psnr_direct'].append(psnr(direct * mask, gt_image * mask).mean().item())
    return {key: float(np.mean(values)) if values else float('nan') for key, values in stats.items()}


def prepare_output_and_logger(args):
    reflection_path = os.path.join(args.model_path, REFLECTION_DIR)
    print('Output folder: {}'.format(reflection_path))
    os.makedirs(reflection_path, exist_ok=True)
    with open(os.path.join(reflection_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(reflection_path) if TENSORBOARD_FOUND else None
    if tb_writer is None:
        print('Tensorboard not available: not logging progress')
    return reflection_path, tb_writer


def training(dataset, opt, pipe, testing_iterations, saving_iterations, interface_iteration=-1):
    _, tb_writer = prepare_output_and_logger(dataset)

    # ---- Phase I results, frozen for the whole of Phase II
    scene, interface, interface_deform = load_interface(dataset, interface_iteration, shuffle=True)
    interface.freeze()
    interface_deform.freeze()
    normal, offset, plane_record = load_plane(os.path.join(dataset.model_path, PLANE_FILE))
    print(f'Glass plane ({plane_record.get("source", "?")}): n = {[round(v, 4) for v in normal.tolist()]}, d = {offset:.4f}')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    use_virtual_camera = not dataset.ablate_virtual_camera

    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    num_frames = len(viewpoint_stack)
    camera_centers = torch.stack([cam.camera_center for cam in viewpoint_stack])
    direct_cache = DirectCache(viewpoint_stack, interface, interface_deform, pipe, background, use_gt_mask=opt.gt_glass_mask)

    # ---- What Phase II trains
    blend = BlendModel.from_args(dataset).cuda()
    blend_optimizer = torch.optim.Adam(blend.parameters(), lr=opt.optical_lr) if blend.is_trainable else None

    own_field = not dataset.shared_reflection_field
    if own_field:
        reflection = ReflectionGaussianModel(dataset.sh_degree)
        reflection.create_random(normal, offset, camera_centers, scene.cameras_extent, opt.refl_init_points, opt.refl_init_radius_scale, opt.refl_init_color)
        reflection.training_setup(opt)
        reflection_deform = DeformModel.from_args(ReflectionGaussianModel.SCALE_DIM, dataset)
        reflection_deform.train_setting(opt, spatial_lr_scale=scene.cameras_extent)
    else:
        # Shared-field ablation: the reflection is the frozen interface itself, seen from the virtual camera.
        reflection, reflection_deform = interface, interface_deform
        if blend_optimizer is None:
            print('Shared reflection field with pure Fresnel mixing: nothing to train, evaluating only.')

    num_iterations = opt.iterations if (own_field or blend_optimizer is not None) else 0
    testing_iterations = sorted(set(i for i in testing_iterations if i <= num_iterations) | {num_iterations})
    saving_iterations = sorted(set(i for i in saving_iterations if 0 < i <= num_iterations) | {num_iterations})

    def evaluate(iteration):
        torch.cuda.empty_cache()
        results = []
        train_subset = [viewpoint_stack[idx % num_frames] for idx in range(5, 30, 5)]
        for name, cameras in (('test', scene.getTestCameras()), ('train', train_subset)):
            if not cameras:
                continue
            metrics = evaluate_composite(cameras, interface, interface_deform, reflection, reflection_deform, pipe, background, normal, offset, blend, opt.gt_glass_mask, use_virtual_camera)
            results.append({'iteration': iteration, 'split': name, **metrics})
            print(
                f'\n[ITER {iteration}] Evaluating {name}: PSNR {metrics["psnr"]:.3f} (direct only {metrics["psnr_direct"]:.3f}), '
                f'glass PSNR {metrics["glass_psnr"]:.3f} (direct only {metrics["glass_psnr_direct"]:.3f})'
            )
            if tb_writer:
                for key, value in metrics.items():
                    tb_writer.add_scalar(f'{name}/{key}', value, iteration)
        torch.cuda.empty_cache()
        return results

    def save(iteration):
        print('\n[ITER {}] Saving reflection'.format(iteration))
        save_reflection(dataset.model_path, iteration, blend, reflection if own_field else None, reflection_deform if own_field else None)

    history = []
    ema_loss_dict = {}
    progress_bar = tqdm(range(0, num_iterations), desc='Reflection training')

    for iteration in range(1, num_iterations + 1):
        if own_field:
            reflection.update_learning_rate(iteration)
            reflection_deform.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                reflection.oneupSHdegree()

        viewpoint_cam = viewpoint_stack[np.random.randint(num_frames)]
        direct, glass_map = direct_cache(viewpoint_cam)

        # The reflection field trains statically first; a shared (frozen) field always uses its own deformation.
        deforming = iteration > opt.warm_up if own_field else True
        noise = time_noise_std(opt, iteration, num_frames) if own_field else 0.0
        offsets = reflection_deform.step(reflection.get_xyz, viewpoint_cam.time, noise) if deforming else None

        out = render_composite(viewpoint_cam, direct, glass_map, reflection, pipe, normal, offset, reflection_offsets=offsets, blend=blend, use_virtual_camera=use_virtual_camera)
        loss_dict = reflection_losses(out, viewpoint_cam, opt, offsets if own_field else None)
        loss = sum(loss_dict.values())
        loss.backward()

        with torch.no_grad():
            for key, value in loss_dict.items():
                ema_loss_dict[key] = 0.4 * value.item() + 0.6 * ema_loss_dict.get(key, value.item())
            if iteration % 10 == 0:
                progress_bar.set_postfix({**{k: f'{v:.4f}' for k, v in ema_loss_dict.items()}, 'Points': reflection.num_points})
                progress_bar.update(10)

            if tb_writer:
                tb_writer.add_scalar('train_loss/total', loss.item(), iteration)
                for key, value in loss_dict.items():
                    tb_writer.add_scalar(f'train_loss/{key}', value.item(), iteration)
                tb_writer.add_scalar('total_points', reflection.num_points, iteration)

            if iteration in testing_iterations:
                history.extend(evaluate(iteration))
            if iteration in saving_iterations:
                save(iteration)

            # Densification
            if own_field and iteration < opt.densify_until_iter:
                visibility_filter, radii = out['visibility_filter'], out['radii']
                reflection.max_radii2D[visibility_filter] = torch.max(reflection.max_radii2D[visibility_filter], radii[visibility_filter])
                reflection.add_densification_stats(out['viewspace_points'], visibility_filter, iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    reflection.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0:
                    reflection.reset_opacity()

            if iteration < num_iterations:
                if own_field:
                    reflection.optimizer.step()
                    reflection.optimizer.zero_grad(set_to_none=True)
                    if deforming:
                        reflection_deform.optimizer.step()
                    reflection_deform.optimizer.zero_grad(set_to_none=True)
                if blend_optimizer is not None:
                    blend_optimizer.step()
                    blend_optimizer.zero_grad(set_to_none=True)
    progress_bar.close()

    if num_iterations == 0:
        history.extend(evaluate(0))
        save(0)

    if tb_writer:
        tb_writer.close()
    return history


if __name__ == '__main__':
    parser = ArgumentParser(description='VIGOR-GS Phase II: reflection reconstruction')
    lp = ModelParams(parser, sentinel=True)  # the scene settings come from Phase I's cfg_args
    op = ReflectionOptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--interface_iteration', default=-1, type=int, help='Phase I iteration to load (default: the latest)')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser, cfg_subdir=INTERFACE_DIR)

    print('Optimizing ' + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, getattr(args, 'interface_iteration', -1))

    print('\nReflection training complete.')
