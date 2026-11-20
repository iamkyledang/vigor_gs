"""
VIGOR-GS Phase II: fit the reflection field G_R = {G_R^0, phi_R} (and h_psi) with G_I and the glass fixed (Eq. 5-8).

    R_t = R_3D^refl(G_R(t), C_t, P; V_dyn)                                  Eq. (5), gaussian_renderer/reflection.py
    I_hat_t = (1 - alpha_t) B_t + alpha_t R_t                               Eq. (6), alpha of Eq. (11)
    L_R = (1/|Omega|) sum_u ||I_hat_t(u) - I_t(u)||_1 + lambda_m L_motion + lambda_o L_opt    Eq. (7), (7b), (7c)

B_t comes from the frozen Phase I field (cached, utils/model_io.DirectPass); every parameter of G_I, D_I and the
patches stays fixed. L_motion compares the predicted and observed change between a frame and the previous frame of
the same sequence and camera (both predictions get gradients). G_R starts as points sampled along the training
rays, colored by the residual the reflection has to explain.

    python train_reflection.py -m <model> [--pure_fresnel] [--ablate_virtual_camera] ...

Needs <model>/interface/ (train_interface.py) and an accepted <model>/glass_geometry.json (extract_glass.py).
Writes <model>/reflection/{config.json, latest.pth, training.jsonl, point_cloud.ply}.
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, ReflectionOptimizationParams, ReflectionParams, load_model_args, save_config
from gaussian_renderer.composition import build_blend, ray_mode, render_composite, schlick
from gaussian_renderer.reflection import ray_bundle
from scene import Scene
from scene.deformation import Deformation
from scene.gaussian_model import ReflectionField
from utils.checkpoint_utils import load_checkpoint, restore_phase, restore_rng_state, save_checkpoint
from utils.glass_geometry import first_hit
from utils.image_utils import masked_psnr
from utils.losses import dssim, motion_loss, optical_loss, pixel_l1
from utils.model_io import DirectPass, load_glass, load_interface, occlusion_settings
from utils.run_utils import TrainingLog, ViewSampler, phase_dir, seed_everything

PHASE = 'reflection'


def previous_frames(cameras):
    """{uid: previous camera of the same sequence and sensor in time order, or None} for L_motion."""
    groups = {}
    for cam in cameras:
        groups.setdefault((cam.sequence, cam.sensor_id, cam.image_width, cam.image_height), []).append(cam)
    previous = {}
    for views in groups.values():
        views.sort(key=lambda c: c.time)
        for i, cam in enumerate(views):
            previous[cam.uid] = views[i - 1] if i > 0 and views[i - 1].time < cam.time else None
    return previous


def build_reflection(model_args, params, scene):
    field = ReflectionField(model_args.sh_degree, params.max_reflection_points)
    deformation = Deformation(3, model_args, scene.center, scene.extent, static=params.static_reflection).cuda()
    optical, constant = build_blend(params, scene.center, scene.extent)
    return field, deformation, (optical.cuda() if optical else None), (constant.cuda() if constant else None)


@torch.no_grad()
def initialize_reflection(field, scene, direct_pass, patches, settings, params, pipe):
    """
    Points along the rays of the reflection render of sampled training views, at 0.1-3.1 scene extents from the
    glass, colored (target - (1 - F) B) / F: the radiance that would explain the residual on its own.
    """
    views = scene.getTrainCameras()[::max(1, len(scene.getTrainCameras()) // 12)]
    per_view = int(np.ceil(params.reflection_init_points / len(views)))
    positions, colors = [], []
    for cam in views:
        direct = direct_pass(cam)
        geometry = first_hit(cam, patches, pipe.ray_epsilon, occlusion_settings(settings, direct))
        bundle = ray_bundle(cam, geometry, patches, ray_mode(params), pipe.ray_epsilon)
        candidates = (geometry['visible'] & cam.valid_mask[0]).flatten().nonzero().flatten()
        if not len(candidates):
            continue
        take = candidates[torch.randint(len(candidates), (per_view,), device='cuda')]
        distance = scene.extent * (0.1 + 3.0 * torch.rand(len(take), 1, device='cuda'))
        positions.append(bundle['origins'].reshape(-1, 3)[take] + distance * bundle['directions'].reshape(-1, 3)[take])
        target = cam.original_linear.permute(1, 2, 0).reshape(-1, 3)[take]
        base = direct['direct'].permute(1, 2, 0).reshape(-1, 3)[take]
        weight = schlick(geometry['cosine'].flatten()[take, None], params.eta_air, params.eta_glass).clamp_min(0.01)
        colors.append(((target - (1 - weight) * base) / weight).clamp(0, 20))
    if not positions:
        raise RuntimeError('No training ray meets the glass patches: check glass_geometry.json and glass_preview/')
    field.create_from_points(torch.cat(positions)[:params.reflection_init_points], torch.cat(colors)[:params.reflection_init_points], scene.extent)


def reflection_losses(out, cam, opt, previous=None, previous_cam=None, learned_correction=True):
    """The terms of L_R for one view (already weighted). `previous`: the render of the previous frame, if any."""
    valid = cam.valid_mask
    target = cam.original_linear
    losses = {'rgb': pixel_l1(out['composite'], target, valid)}
    if opt.lambda_dssim > 0:
        losses['dssim'] = opt.lambda_dssim * dssim(out['composite'], target)
    if previous is not None:
        losses['motion'] = opt.lambda_motion * motion_loss(out['composite'], previous['composite'], target, previous_cam.original_linear, valid & previous_cam.valid_mask)
    if opt.lambda_optical > 0 and learned_correction:
        losses['optical'] = opt.lambda_optical * optical_loss(out['h'], valid)
    return losses


@torch.no_grad()
def evaluate_composite(cameras, render):
    """
    Mean PSNR (linear RGB, valid pixels) of the composite and of the direct image alone against the original,
    over the whole image and over the annotated glass. Their difference is what the reflection explains.
    """
    scores = {'psnr': [], 'psnr_direct': [], 'glass_psnr': [], 'glass_psnr_direct': []}
    for cam in cameras:
        out = render(cam)
        target, valid, glass = cam.original_linear, cam.valid_mask, cam.glass_mask & cam.valid_mask
        for suffix, image in (('', out['composite']), ('_direct', out['direct'])):
            scores['psnr' + suffix].append(masked_psnr(image.clamp(0, 1), target, valid))
            value = masked_psnr(image.clamp(0, 1), target, glass)
            if value is not None:
                scores['glass_psnr' + suffix].append(value)
    return {k: float(np.mean(v)) if v else None for k, v in scores.items()}


def training(model_args, pipe, params, opt, resume=False, quiet=False):
    if model_args.ignore_clean_images:
        raise SystemExit('This model is the no-separation ablation (--ignore_clean_images): it has no Phase II.')
    seed_everything(model_args.seed)
    patches, glass_record = load_glass(model_args)
    if not glass_record.get('accepted', True):
        raise SystemExit('glass_geometry.json failed the extraction checks (accepted: false): improve Phase I or rerun extract_glass.py with --glass_geometry')
    settings = glass_record.get('settings', {'ablate_occlusion': True})
    folder = phase_dir(model_args.model_path, PHASE, resume)
    save_config(os.path.join(folder, 'config.json'), model_args, pipe, params, opt)

    scene = Scene(model_args, load_clean=False)
    interface, interface_deformation, _ = load_interface(model_args, scene)
    direct_pass = DirectPass(interface, interface_deformation, pipe, opt.cache_views)
    field, deformation, optical, constant = build_reflection(model_args, params, scene)
    blend_parameters = [p for module in (optical, constant) if module is not None for p in module.parameters()]
    blend_optimizer = torch.optim.Adam(blend_parameters, lr=opt.optical_lr) if blend_parameters else None
    start = 0
    if resume:
        state = load_checkpoint(os.path.join(folder, 'latest.pth'))
        restore_phase(state, field, deformation, opt, opt.deform_lr)
        for name, module in (('optical', optical), ('constant', constant)):
            if module is not None:
                module.load_state_dict(state['modules'][name])
        if blend_optimizer is not None:
            blend_optimizer.load_state_dict(state['optimizers']['blend'])
        restore_rng_state(state)
        start = state['iteration']
    else:
        initialize_reflection(field, scene, direct_pass, patches, settings, params, pipe)
        field.training_setup(opt)
        deformation.training_setup(opt.deform_lr)

    def render(cam, deforming=True):
        return render_composite(cam, direct_pass(cam), field, deformation, patches, settings, params, pipe, optical, constant, deforming)

    log = TrainingLog(folder)
    cameras = scene.getTrainCameras()
    sampler = ViewSampler(cameras)
    previous_of = previous_frames(cameras)
    progress = tqdm(range(start + 1, opt.iterations + 1), desc='Phase II', disable=quiet)
    ema = None
    for iteration in progress:
        field.update_learning_rate(iteration)
        deformation.update_learning_rate(opt.deform_lr, iteration, opt.iterations)
        if iteration % 1000 == 0:
            field.oneupSHdegree()
        deforming = iteration > opt.warm_up

        cam = sampler.next()
        out = render(cam, deforming)
        renders = [out]
        previous_cam = previous_of.get(cam.uid) if opt.lambda_motion > 0 else None
        previous = None
        if previous_cam is not None:
            previous = render(previous_cam, deforming)
            renders.append(previous)
        losses = reflection_losses(out, cam, opt, previous, previous_cam, learned_correction=optical is not None)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f'Phase II loss is not finite at iteration {iteration}; the last checkpoint is intact')
        loss.backward()

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                for entry in renders:
                    means = entry['deformed_means']
                    if means.grad is not None:
                        field.add_densification_stats(means.grad.norm(dim=-1, keepdim=True) * scene.extent, entry['visibility_filter'])
            field.optimizer.step()
            field.optimizer.zero_grad(set_to_none=True)
            for optimizer in (deformation.optimizer, blend_optimizer):
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if iteration < opt.densify_until_iter:
                reset = opt.presence_reset_interval
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    field.densify_and_prune(opt.densify_grad_threshold, opt.min_presence, scene.extent, prune_large=bool(reset) and iteration > reset)
                if reset and iteration % reset == 0:
                    field.reset_presence()

            ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
            if iteration % 10 == 0:
                progress.set_postfix(loss=f'{ema:.4f}', points=field.num_points)
            record = {}
            if iteration % opt.test_interval == 0 or iteration == opt.iterations:
                record.update({f'test_{k}': v for k, v in evaluate_composite(scene.getTestCameras(), render).items()})
                tqdm.write(f'[Phase II {iteration}] test {record}')
            if iteration % opt.report_interval == 0 or record:
                log.write(iteration, loss=loss, points=field.num_points, tile_entries=out['tile_entries'], **losses, **record)
            if iteration % opt.checkpoint_interval == 0 or iteration == opt.iterations:
                save_checkpoint(os.path.join(folder, 'latest.pth'), field, deformation, iteration,
                                modules={'optical': optical, 'constant': constant}, optimizers={'blend': blend_optimizer})
    field.save_ply(os.path.join(folder, 'point_cloud.ply'))
    return field, deformation, optical, constant


def parse(argv, known=False):
    """Parsed settings of this step; with `known`, arguments of other steps are left in args.unknown (train.py)."""
    parser = ArgumentParser(description='VIGOR-GS Phase II: reflection field with the interface and glass fixed')
    model, pipeline = ModelParams(parser, sentinel=True), PipelineParams(parser)
    params, optimization = ReflectionParams(parser), ReflectionOptimizationParams(parser)
    parser.add_argument('--resume', action='store_true', help='continue <model>/reflection/latest.pth')
    parser.add_argument('--quiet', action='store_true')
    args, unknown = parser.parse_known_args(argv) if known else (parser.parse_args(argv), [])
    if not args.model_path:
        parser.error('-m is required')
    model_args = model.extract(args)
    model_args.model_path = os.path.abspath(model_args.model_path)
    args.unknown = unknown
    return load_model_args(model_args), pipeline.extract(args), params.extract(args), optimization.extract(args), args


def main(argv=None, known=False):
    model_args, pipe, params, opt, args = parse(sys.argv[1:] if argv is None else argv, known)
    training(model_args, pipe, params, opt, resume=args.resume, quiet=args.quiet)
    print('Phase II done:', os.path.join(model_args.model_path, PHASE))


if __name__ == '__main__':
    main()
