"""
VIGOR-GS Phase I: fit the interface field G_I = {G_I^0, phi_I} to the clean targets (Eq. 2-4).

    B_t = R_2D(G_I(t), C_t)                                             Eq. (2)
    L_I = (1/|Omega|) sum_u ||B_t(u) - I_t^clean(u)||_1 + lambda_s L_surf    Eq. (3), L_surf of Eq. (3a)

plus, when lambda_glass > 0, the glass-mask prior that lets the extracted patches come from this geometry (see
utils/losses.py). The deformation network D_I is used after `warm_up` iterations. With --ignore_clean_images the
targets are the original images (the no-separation ablation: one field, no Phase II).

    python train_interface.py -s <scene> -m <model> [-r 2]

Writes <model>/config.json (the model settings every later step reuses), <model>/interface/{config.json,
latest.pth, training.jsonl, point_cloud.ply}. --resume continues <model>/interface/latest.pth.
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from arguments import MODEL_CONFIG, InterfaceOptimizationParams, ModelParams, PipelineParams, save_config
from gaussian_renderer.interface import render_interface
from scene import Scene
from scene.deformation import Deformation
from scene.gaussian_model import InterfaceField
from utils.checkpoint_utils import load_checkpoint, restore_phase, restore_rng_state, save_checkpoint
from utils.color_utils import srgb2linear
from utils.image_utils import masked_psnr
from utils.losses import dssim, glass_prior_loss, pixel_l1, surface_loss
from utils.run_utils import TrainingLog, ViewSampler, phase_dir, seed_everything

PHASE = 'interface'


def interface_target(cam, model_args):
    """I_t^clean, or the original image in the no-separation ablation."""
    return cam.original_linear if model_args.ignore_clean_images else cam.clean_linear


def interface_losses(out, cam, target, opt, iteration):
    """The terms of L_I for one view, as a dict of scalar tensors (already weighted)."""
    valid = cam.valid_mask
    losses = {'rgb': pixel_l1(out['direct'], target, valid)}
    if opt.lambda_dssim > 0:
        losses['dssim'] = opt.lambda_dssim * dssim(out['direct'], target)
    if opt.lambda_surface > 0 and iteration >= opt.surface_from_iter:
        losses['surface'] = opt.lambda_surface * surface_loss(out['normal'], out['depth_normal'], out['surface_valid'] & valid)
    if opt.lambda_glass > 0:
        losses['glass'] = opt.lambda_glass * glass_prior_loss(out['glass_map'], cam.glass_mask, valid)
    return losses


@torch.no_grad()
def evaluate_interface(cameras, field, deformation, pipe, model_args):
    """Mean PSNR of B_t against its target (linear RGB, valid pixels) and IoU of the glass map with the mask."""
    psnrs, ious = [], []
    for cam in cameras:
        if not (cam.has_clean or model_args.ignore_clean_images):
            continue
        out = render_interface(cam, field, deformation, pipe, surface=True)
        valid = cam.valid_mask
        psnrs.append(masked_psnr(out['direct'].clamp(0, 1), interface_target(cam, model_args), valid))
        predicted, annotated = (out['glass_map'] > 0.5) & valid, cam.glass_mask & valid
        union = (predicted | annotated).sum().item()
        ious.append((predicted & annotated).sum().item() / union if union else 1.0)
    return {'psnr': float(np.mean(psnrs)) if psnrs else None, 'glass_iou': float(np.mean(ious)) if ious else None}


def build_interface(model_args, scene):
    field = InterfaceField(model_args.sh_degree, model_args.max_interface_points, factorized=not model_args.ablate_factorization)
    deformation = Deformation(2, model_args, scene.center, scene.extent, static=model_args.static_interface).cuda()
    return field, deformation


def training(model_args, opt, pipe, resume=False, quiet=False):
    seed_everything(model_args.seed)
    folder = phase_dir(model_args.model_path, PHASE, resume)
    save_config(os.path.join(model_args.model_path, MODEL_CONFIG), model_args)
    save_config(os.path.join(folder, 'config.json'), model_args, pipe, opt)

    scene = Scene(model_args)
    field, deformation = build_interface(model_args, scene)
    start = 0
    if resume:
        state = load_checkpoint(os.path.join(folder, 'latest.pth'))
        restore_phase(state, field, deformation, opt, opt.deform_lr)
        restore_rng_state(state)
        start = state['iteration']
    else:
        colors = srgb2linear(torch.as_tensor(scene.point_cloud.colors, dtype=torch.float32))
        field.create_from_points(scene.point_cloud.points, colors, scene.extent)
        field.training_setup(opt)
        deformation.training_setup(opt.deform_lr)

    log = TrainingLog(folder)
    sampler = ViewSampler(scene.getTrainCameras())
    surface = opt.lambda_glass > 0
    progress = tqdm(range(start + 1, opt.iterations + 1), desc='Phase I', disable=quiet)
    ema = None
    for iteration in progress:
        field.update_learning_rate(iteration)
        deformation.update_learning_rate(opt.deform_lr, iteration, opt.iterations)
        if iteration % 1000 == 0:
            field.oneupSHdegree()

        cam = sampler.next()
        out = render_interface(cam, field, deformation, pipe, surface=surface, deforming=iteration > opt.warm_up)
        losses = interface_losses(out, cam, interface_target(cam, model_args), opt, iteration)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f'Phase I loss is not finite at iteration {iteration}; the last checkpoint is intact')
        loss.backward()

        with torch.no_grad():
            visible = out['visibility_filter']
            if iteration < opt.densify_until_iter:
                field.update_max_radii(out['radii'], visible)
                field.add_densification_stats(out['viewspace_points'].grad[:, :2].norm(dim=-1, keepdim=True), visible)
            field.optimizer.step()
            field.optimizer.zero_grad(set_to_none=True)
            if deformation.optimizer is not None:
                deformation.optimizer.step()
                deformation.optimizer.zero_grad(set_to_none=True)
            if iteration < opt.densify_until_iter:
                reset = opt.presence_reset_interval
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    field.densify_and_prune(opt.densify_grad_threshold, opt.min_presence, scene.extent, opt.max_screen_size,
                                            prune_large=bool(reset) and iteration > reset)
                if reset and iteration % reset == 0:
                    field.reset_presence()

            ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
            if iteration % 10 == 0:
                progress.set_postfix(loss=f'{ema:.4f}', points=field.num_points)
            record = {}
            if iteration % opt.test_interval == 0 or iteration == opt.iterations:
                record.update({f'test_{k}': v for k, v in evaluate_interface(scene.getTestCameras(), field, deformation, pipe, model_args).items()})
                tqdm.write(f'[Phase I {iteration}] test {record}')
            if iteration % opt.report_interval == 0 or record:
                log.write(iteration, loss=loss, points=field.num_points, **losses, **record)
            if iteration % opt.checkpoint_interval == 0 or iteration == opt.iterations:
                save_checkpoint(os.path.join(folder, 'latest.pth'), field, deformation, iteration)
    field.save_ply(os.path.join(folder, 'point_cloud.ply'))
    return scene, field, deformation


def parse(argv):
    parser = ArgumentParser(description='VIGOR-GS Phase I: interface field on the clean targets')
    model, pipeline, optimization = ModelParams(parser), PipelineParams(parser), InterfaceOptimizationParams(parser)
    parser.add_argument('--resume', action='store_true', help='continue <model>/interface/latest.pth')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if not args.source_path or not args.model_path:
        parser.error('-s and -m are required')
    return model.extract(args), optimization.extract(args), pipeline.extract(args), args


def main(argv=None):
    model_args, opt, pipe, args = parse(sys.argv[1:] if argv is None else argv)
    model_args.model_path = os.path.abspath(model_args.model_path)
    training(model_args, opt, pipe, resume=args.resume, quiet=args.quiet)
    print('Phase I done:', os.path.join(model_args.model_path, PHASE))


if __name__ == '__main__':
    main()
