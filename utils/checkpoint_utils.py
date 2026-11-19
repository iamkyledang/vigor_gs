"""Atomic two-field checkpoints, restoration and reproducible resume state.

This module is shared by training and rendering; it does not import a trainer.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

from scene.gaussian_model import GaussianModel
from scene.deformation import Deformation


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


def restore_rng_state(state):
    """Call after constructing every random-initialized module during resume."""
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'].cpu())
    torch.cuda.set_rng_state(state['rng_cuda'].cpu())
