"""
Checkpoints of one phase: the canonical field, its deformation network, both optimizers, extra modules (the
optical correction of Phase II) and the RNG state, written atomically to `<model>/<phase>/latest.pth`.

Shared by training and rendering; it does not import a trainer.
"""

import os
import random

import numpy as np
import torch

CHECKPOINT_VERSION = 2


def save_checkpoint(path, field, deformation, iteration, modules=None, optimizers=None):
    """`modules` / `optimizers`: {name: nn.Module / torch.optim.Optimizer or None} saved next to the field."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'version': CHECKPOINT_VERSION,
        'iteration': iteration,
        'field': field.capture(),
        'deformation': deformation.state_dict(),
        'deformation_optimizer': deformation.optimizer.state_dict() if deformation.optimizer is not None else None,
        'modules': {name: module.state_dict() for name, module in (modules or {}).items() if module is not None},
        'optimizers': {name: optimizer.state_dict() for name, optimizer in (optimizers or {}).items() if optimizer is not None},
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()},
    }
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('version') != CHECKPOINT_VERSION:
        raise ValueError(f'{path}: checkpoint version {state.get("version")}, this code reads version {CHECKPOINT_VERSION}')
    return state


def restore_phase(state, field, deformation, opt=None, deform_lr=None):
    """
    Load a checkpoint into a freshly built `field` and `deformation`. With `opt` (resume), the optimizers and their
    moments come back too; without it both are frozen constants.
    """
    field.restore(state['field'], opt)
    deformation.load_state_dict(state['deformation'])
    deformation.cuda()
    if opt is None:
        deformation.freeze()
        return
    deformation.training_setup(deform_lr)
    if deformation.optimizer is not None and state['deformation_optimizer'] is not None:
        deformation.optimizer.load_state_dict(state['deformation_optimizer'])


def restore_rng_state(state):
    """Call after building every randomly initialized module, so that a resumed run continues the same sequence."""
    rng = state['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    torch.cuda.set_rng_state(rng['cuda'].cpu())
