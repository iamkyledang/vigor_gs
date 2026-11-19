"""Sequential VIGOR-GS training in the existing RT-Splatting entry point."""
import json
import os
from pathlib import Path
import random
import time
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel, Deformation, OpticalCorrection
from gaussian_renderer import render_interface, render_composite, ray_geometry
from utils.point_utils import extract_patches, validate_patches
from utils.color_utils import srgb2linear
from utils.loss_utils import ssim
from utils.system_utils import record_implementation_change




def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pixel_l1(prediction,target,valid):
    errors=(prediction-target).abs().sum(0)
    mask=valid[0].bool()
    if not mask.any(): raise ValueError('No valid image pixels.')
    return errors[mask].mean()


def checkpoint(path,pc,deformation,iteration,phase,optical=None,optical_optimizer=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'version':1,'phase':phase,'iteration':iteration,'field':pc.capture(),
        'deformation':deformation.state_dict(),'deform_optimizer':deformation.optimizer.state_dict(),
        'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
        'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()}
    if optical is not None:
        state['optical']=optical.state_dict()
        state['optical_optimizer']=optical_optimizer.state_dict() if optical_optimizer else None
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);os.replace(temporary,path)


def load_field(path,args,field,extent,center,training=False):
    state=torch.load(path,map_location='cpu',weights_only=False)
    field_state=state['field']
    field_state['params']={key:torch.nn.Parameter(value.detach().cuda(),requires_grad=training) for key,value in field_state['params'].items()}
    for key in ('max_radii2D','xyz_gradient_accum','denom','last_update'):
        field_state[key]=field_state[key].cuda()
    if not training: field_state['optimizer']=None
    pc=GaussianModel(args.sh_degree,args,field=field)
    pc.restore(state['field'],args)
    if training and pc.num_points>pc.max_points:
        raise ValueError('The resumed field already exceeds the requested point cap; keep a cap at least as large as its checkpoint.')
    deformation=Deformation(pc.SCALE_DIM,args,center,extent).cuda()
    deformation.load_state_dict(state['deformation'])
    deformation.optimizer=torch.optim.Adam(deformation.parameters(),lr=args.deform_lr) if training else None
    if training: deformation.optimizer.load_state_dict(state['deform_optimizer'])
    else:
        pc.freeze();deformation.requires_grad_(False);deformation.eval()
    return pc,deformation,state


class DirectCache:
    def __init__(self,interface,deformation,pipe,capacity=16):
        self.interface,self.deformation,self.pipe=interface,deformation,pipe
        self.capacity=capacity;self.entries=OrderedDict()
    @torch.no_grad()
    def __call__(self,cam):
        if cam.uid not in self.entries:
            direct=render_interface(cam,self.interface,self.deformation,self.pipe)['direct'].cpu()
            self.entries[cam.uid]=direct
            if len(self.entries)>self.capacity: self.entries.popitem(last=False)
        self.entries.move_to_end(cam.uid)
        return self.entries[cam.uid].cuda()


def previous_frames(cameras):
    previous={};groups={}
    for cam in cameras: groups.setdefault((cam.sequence,getattr(cam,'sensor_id',0),cam.image_height,cam.image_width),[]).append(cam)
    for views in groups.values():
        views.sort(key=lambda c:c.time)
        for i,cam in enumerate(views):
            previous[cam.uid]=views[i-1] if i>0 and views[i-1].time<cam.time else None
    return previous


def initialize_reflection(scene,interface,deformation,patches,args,pipe,cache):
    positions=[];colors=[]
    views=scene.getTrainCameras()[::max(1,len(scene.getTrainCameras())//12)]
    per_view=max(1,int(np.ceil(args.reflection_init_points/len(views))))
    with torch.no_grad():
        for cam in views:
            geo=ray_geometry(cam,patches,args,pipe)
            candidates=(geo['visible']&cam.valid_mask.cuda()[0]).flatten().nonzero().flatten()
            if not len(candidates): continue
            take=candidates[torch.randint(len(candidates),(per_view,),device='cuda')]
            origin=geo['origins'].reshape(-1,3)[take]
            direction=geo['directions'].reshape(-1,3)[take]
            distance=scene.cameras_extent*(0.1+3*torch.rand(len(take),1,device='cuda'))
            positions.append(origin+distance*direction)
            direct=cache(cam).permute(1,2,0).reshape(-1,3)[take]
            target=srgb2linear(cam.original_image.cuda()).permute(1,2,0).reshape(-1,3)[take]
            alpha=geo['fresnel'].flatten()[take,None].clamp_min(0.01)
            colors.append(((target-(1-alpha)*direct)/alpha).clamp(0,20))
    if not positions: raise RuntimeError('No training rays meet the frozen glass geometry.')
    pc=GaussianModel(args.sh_degree,args,field='reflection')
    pc.create_from_points(torch.cat(positions)[:args.reflection_init_points],torch.cat(colors)[:args.reflection_init_points],scene.cameras_extent)
    return pc


def train_phase(scene,args,pipe,phase,pc,deformation,patches=None,interface=None,interface_deform=None,resume=None):
    length=args.iterations if phase=='interface' else args.reflection_iterations
    if length<1: raise ValueError('Each trained phase needs at least one iteration.')
    path=Path(args.model_path)/phase/'latest.pth'
    start=0
    if resume:
        pc,deformation,state=load_field(resume,args,phase,scene.cameras_extent,scene.center,True)
        if state['phase']!=phase: raise ValueError('Resume phase does not match checkpoint.')
        start=state['iteration']
    else:
        pc.training_setup(args)
        deformation.optimizer=torch.optim.Adam(deformation.parameters(),lr=args.deform_lr)
    optical=OpticalCorrection(args.optical_bound).cuda() if phase=='reflection' else None
    optical_optimizer=None
    if optical is not None and not args.pure_fresnel:
        optical_optimizer=torch.optim.Adam(optical.parameters(),lr=args.optical_lr)
    if resume and optical is not None:
        optical.load_state_dict(state['optical'])
        if optical_optimizer and state.get('optical_optimizer'): optical_optimizer.load_state_dict(state['optical_optimizer'])
    if resume:
        # Restore after constructing all random-initialized modules.
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'].cpu());torch.cuda.set_rng_state(state['rng_cuda'].cpu())
    cameras=scene.getTrainCameras(); neighbors=previous_frames(cameras)
    cache=DirectCache(interface,interface_deform,pipe,args.cache_views) if phase=='reflection' else None
    history=Path(args.model_path)/phase/'training.jsonl';history.parent.mkdir(parents=True,exist_ok=True)
    timer=time.perf_counter();ema=None
    progress=tqdm(range(start+1,length+1),desc=phase)
    for iteration in progress:
        pc.optimizer.zero_grad(set_to_none=True);deformation.optimizer.zero_grad(set_to_none=True)
        if optical_optimizer: optical_optimizer.zero_grad(set_to_none=True)
        pc.update_learning_rate(iteration)
        if iteration%1000==0: pc.oneupSHdegree()
        fraction=min(iteration/max(length,1),1)
        for group in deformation.optimizer.param_groups:
            group['lr']=args.deform_lr*(0.01**fraction)
        cam=random.choice(cameras);valid=cam.valid_mask.cuda()
        deforming=iteration>min(args.warm_up,max(length-1,0))
        terms={}
        if phase=='interface':
            out=render_interface(cam,pc,deformation,pipe,surface=args.lambda_glass>0 or not args.glass_geometry,deforming=deforming)
            target=srgb2linear(cam.clean_image.cuda())
            terms['rgb']=pixel_l1(out['direct'],target,valid)
            if args.lambda_dssim: terms['ssim']=args.lambda_dssim*(1-ssim(out['direct'],target))
            surface=out['surface_valid']&valid
            if iteration>=args.surface_from_iter and surface.any():
                terms['surface']=args.lambda_surface*(out['normal']-out['depth_normal']).abs().sum(0)[surface[0]].mean()
            if args.lambda_glass>0:
                glass=out['glass_map'].clamp(1e-5,1-1e-5)
                terms['glass']=args.lambda_glass*F.binary_cross_entropy(glass[valid],cam.gt_transparent_mask.cuda().float()[valid])
        else:
            direct=cache(cam)
            out=render_composite(cam,interface,interface_deform,pc,deformation,patches,optical,args,pipe,direct=direct,deforming=deforming)
            target=srgb2linear(cam.original_image.cuda())
            terms['rgb']=pixel_l1(out['composite'],target,valid)
            if args.lambda_dssim: terms['ssim']=args.lambda_dssim*(1-ssim(out['composite'],target))
            terms['optical']=args.lambda_optical*out['h'].square()[valid[0]].mean()
            neighbor=neighbors[cam.uid]
            if args.lambda_motion>0 and neighbor is not None:
                previous=render_composite(neighbor,interface,interface_deform,pc,deformation,patches,optical,args,pipe,direct=cache(neighbor),deforming=deforming)
                previous_target=srgb2linear(neighbor.original_image.cuda())
                temporal_valid=valid&neighbor.valid_mask.cuda()
                terms['motion']=args.lambda_motion*pixel_l1(out['composite']-previous['composite'],target-previous_target,temporal_valid)
        loss=sum(terms.values())
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite {phase} loss at iteration {iteration}; last checkpoint remains intact.')
        loss.backward()
        # Collect gradients before optimizer step; mutate topology only after that step.
        with torch.no_grad():
            grow=iteration<min(args.densify_until_iter,length)
            if grow:
                if phase=='interface':
                    visible=out['visibility_filter']
                    pc.add_densification_stats(out['viewspace_points'],visible,iteration)
                else:
                    for entry in ([out,previous] if args.lambda_motion>0 and neighbor is not None else [out]):
                        means=entry['deformed_means']
                        if means.grad is not None:
                            visible=entry['visibility_filter']
                            pc.xyz_gradient_accum[visible]+=means.grad[visible].norm(dim=-1,keepdim=True)*scene.cameras_extent
                            pc.denom[visible]+=1
                            pc.last_update[visible]=iteration
            pc.optimizer.step()
            if deforming: deformation.optimizer.step()
            if optical_optimizer: optical_optimizer.step()
            if grow and iteration>args.densify_from_iter and iteration%args.densification_interval==0:
                pc.densify_and_prune(args.densify_grad_threshold if phase=='interface' else args.reflection_grad_threshold,scene.cameras_extent)
            value=float(loss);ema=value if ema is None else 0.95*ema+0.05*value
            if iteration%10==0: progress.set_postfix(loss=f'{ema:.5f}',points=pc.num_points)
            if iteration%args.report_interval==0 or iteration==length:
                torch.cuda.synchronize()
                record={'iteration':iteration,'loss':value,'points':pc.num_points,
                    'elapsed_seconds':time.perf_counter()-timer,'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20,
                    **{name:float(term) for name,term in terms.items()}}
                if phase=='reflection': record['tile_entries']=out['tile_entries']
                with history.open('a') as stream: stream.write(json.dumps(record)+'\n')
            if iteration%args.checkpoint_interval==0 or iteration==length:
                checkpoint(path,pc,deformation,iteration,phase,optical,optical_optimizer)
    if start>=length: print(f'{phase}: checkpoint already reached requested iteration {length}.')
    return pc,deformation,optical


def main():
    parser=ArgumentParser(description=__doc__)
    ModelParams(parser);PipelineParams(parser);OptimizationParams(parser)
    parser.add_argument('--phase',choices=['all','interface','reflection'],default='all')
    parser.add_argument('--resume',default='')
    parser.add_argument('--cache_views',type=int,default=16)
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args()
    import sys
    if args.model_path and (args.resume or args.phase=='reflection'):
        config_path=Path(args.model_path)/'config.json'
        if config_path.exists():
            saved=json.loads(config_path.read_text())
            explicit={token.split('=')[0].lstrip('-') for token in sys.argv[1:] if token.startswith('--')}
            explicit |= {'source_path' if token=='-s' else 'model_path' if token=='-m' else 'resolution' for token in sys.argv[1:] if token in ('-s','-m','-r')}
            for key,value in saved.items():
                if key not in explicit and key not in ('phase','resume'):
                    setattr(args,key,value)
    if not args.source_path or not args.model_path: parser.error('-s and -m are required')
    if args.checkpoint_interval<1 or args.report_interval<1 or args.cache_views<1: parser.error('Intervals/cache must be positive')
    if args.eta_air<=0 or args.eta_glass<=0: parser.error('Refractive indices must be positive')
    if args.sh_degree not in range(4): parser.error('SH degree must lie between zero and three')
    if args.init_points<4 or args.reflection_init_points<4: parser.error('At least four initialization points per field are required')
    if min(args.deform_depth,args.deform_width,args.max_patches,args.tile_size)<1: parser.error('Network, patch and tile dimensions must be positive')
    if min(args.lambda_surface,args.lambda_motion,args.lambda_optical,args.lambda_glass,args.lambda_dssim)<0: parser.error('Loss weights must be nonnegative')
    if args.plane_threshold<=0 or args.ray_cutoff<=0: parser.error('Geometry threshold and ray cutoff must be positive')
    if min(args.densify_grad_threshold,args.reflection_grad_threshold)<=0: parser.error('Densification gradient thresholds must be positive')
    if not torch.cuda.is_available(): raise RuntimeError('A CUDA GPU and the VIGOR CUDA extension are required.')
    args.source_path=str(Path(args.source_path).resolve());args.model_path=str(Path(args.model_path).resolve())
    output=Path(args.model_path);output.mkdir(parents=True,exist_ok=True)
    if (output/'interface/latest.pth').exists() and args.phase!='reflection' and not args.resume:
        raise FileExistsError('Model already has a checkpoint. Choose a fresh -m folder or --resume its checkpoint.')
    if (output/'reflection/latest.pth').exists() and args.phase in ('all','reflection') and not args.resume:
        raise FileExistsError('Reflection checkpoint exists. Use a fresh -m folder or --resume.')
    config=output/'config.json'
    if config.exists() and (args.resume or args.phase=='reflection'):
        saved=json.loads(config.read_text())
        for name in ('resolution','sh_degree','deform_depth','deform_width','deform_xyz_freqs','deform_time_freqs','eta_air','eta_glass','optical_bound','pure_fresnel'):
            if saved[name]!=getattr(args,name):
                raise ValueError(f'{name} differs from saved model configuration ({saved[name]}). Pass the same configuration when resuming.')
    config.write_text(json.dumps(vars(args),indent=2))
    record_implementation_change();seed_everything(args.seed)
    scene=Scene(args)
    if args.init_points>args.max_interface_points or args.reflection_init_points>args.max_reflection_points:
        raise ValueError('Initial point counts exceed their configured point caps.')
    resume_phase=torch.load(args.resume,map_location='cpu',weights_only=False)['phase'] if args.resume else None
    trained_interface=args.phase in ('all','interface') and resume_phase!='reflection'
    if trained_interface:
        pc=GaussianModel(args.sh_degree,args)
        pc.create_from_points(scene.point_cloud.points,srgb2linear(torch.as_tensor(scene.point_cloud.colors).float()).numpy(),scene.cameras_extent)
        deformation=Deformation(2,args,scene.center,scene.cameras_extent).cuda()
        interface,interface_deform,_=train_phase(scene,args,args,'interface',pc,deformation,resume=args.resume or None)
    else:
        interface,interface_deform,_=load_field(output/'interface/latest.pth',args,'interface',scene.cameras_extent,scene.center)
    interface.freeze();interface_deform.requires_grad_(False);interface_deform.eval();interface_deform.optimizer=None
    geometry_path=output/'glass_geometry.json'
    if args.glass_geometry:
        raw=json.loads(Path(args.glass_geometry).read_text())
        records=raw['patches'] if isinstance(raw,dict) else raw
        quality={'source':'provided'}
        validate_patches(records)
        geometry_path.write_text(json.dumps({'patches':records,'quality':quality},indent=2))
    elif geometry_path.exists() and not trained_interface:
        raw=json.loads(geometry_path.read_text());records,quality=raw['patches'],raw['quality']
    else:
        records,quality=extract_patches(scene.getTrainCameras(),interface,interface_deform,args,args,render_interface,scene.cameras_extent)
        geometry_path.write_text(json.dumps({'patches':records,'quality':quality},indent=2))
    if quality.get('coverage',1)<args.plane_min_coverage or quality.get('precision',1)<args.plane_min_precision:
        raise RuntimeError('Extracted glass geometry failed coverage/precision checks. See glass_geometry.json; improve Phase I or supply geometry. No Phase II training was performed.')
    if args.phase=='interface': return
    patches=validate_patches(records)
    # Release the frozen interface optimizer and stale gradient buffers before Phase II.
    interface.optimizer=None
    for name in interface.PARAMS: getattr(interface,'_'+name).grad=None
    for parameter in interface_deform.parameters(): parameter.grad=None
    torch.cuda.empty_cache()
    for camera in scene.getTrainCameras(): camera.clean_image=None
    cache=DirectCache(interface,interface_deform,args,args.cache_views)
    reflection=initialize_reflection(scene,interface,interface_deform,patches,args,args,cache) if resume_phase!='reflection' else GaussianModel(args.sh_degree,args,'reflection')
    reflection_deform=Deformation(3,args,scene.center,scene.cameras_extent).cuda()
    train_phase(scene,args,args,'reflection',reflection,reflection_deform,patches,interface,interface_deform,args.resume if resume_phase=='reflection' else None)
    print('Training complete:',args.model_path)


if __name__=='__main__': main()
