"""Canonical fields, deformation networks and phase checkpoints."""

from types import SimpleNamespace

import pytest
import torch

from scene.deformation import Deformation
from scene.gaussian_model import InterfaceField, ReflectionField
from utils.checkpoint_utils import load_checkpoint, restore_phase, restore_rng_state, save_checkpoint

pytestmark = pytest.mark.gpu

OPT = SimpleNamespace(position_lr_init=1e-3, position_lr_final=1e-5, position_lr_delay_mult=0.01, position_lr_max_steps=100,
                      feature_lr=0.0025, occupancy_lr=0.025, opacity_lr=0.025, scaling_lr=0.005, rotation_lr=0.001, percent_dense=0.01)
NET = SimpleNamespace(deform_xyz_freqs=3, deform_time_freqs=2, deform_depth=2, deform_width=16)


def make(cls, n=64, **kwargs):
    field = cls(1, **kwargs)
    points = torch.rand(n, 3) * 2 - 1
    field.create_from_points(points, torch.rand(n, 3), spatial_lr_scale=1.0)
    field.training_setup(OPT)
    return field


def test_interface_renders_with_occupancy_times_opacity():
    field = make(InterfaceField)
    assert 'occupancy' in field.PARAMS and field.get_scaling.shape[1] == 2
    torch.testing.assert_close(field.get_effective_opacity, field.get_occupancy * field.get_opacity)
    assert torch.allclose(field.get_occupancy, torch.full_like(field.get_occupancy, InterfaceField.INIT_OCCUPANCY))
    plain = make(InterfaceField, factorized=False)
    assert 'occupancy' not in plain.PARAMS
    assert torch.all(plain.get_occupancy == 1) and torch.equal(plain.get_effective_opacity, plain.get_opacity)
    reflection = make(ReflectionField)
    assert reflection.get_scaling.shape[1] == 3 and 'occupancy' not in reflection.PARAMS


def test_densification_keeps_moments_of_survivors_and_respects_the_point_cap():
    field = make(ReflectionField, n=50, max_points=60)
    loss = (field.get_xyz.square().sum() + field.get_opacity.sum())
    loss.backward()
    field.optimizer.step()
    moments = field.optimizer.state[field._xyz]['exp_avg'].clone()
    field.xyz_gradient_accum[:] = 1.0
    field.denom[:] = 1.0
    field.densify(max_grad=0.5, extent=1.0)
    assert field.num_points == 60  # 10 slots: the 10 strongest clone or split, each adding one
    state = field.optimizer.state[field._xyz]['exp_avg']
    assert state.shape[0] == 60
    assert torch.count_nonzero(state.abs().sum(-1)) <= 50  # new Gaussians start with zero moments
    assert torch.isin(moments[:, 0], state[:, 0]).sum() >= 40  # survivors keep theirs


def test_presence_reset_and_pruning():
    field = make(InterfaceField, n=32)
    field.reset_presence(0.01)
    assert torch.all(field.get_occupancy <= 0.0100001)
    field.densify_and_prune(max_grad=1.0, min_presence=0.02, extent=1.0)
    assert field.num_points == 1  # everything is below min_presence; the most present Gaussian is kept
    field = make(ReflectionField, n=8)
    with torch.no_grad():
        field._scaling[0] = 3.0  # e^3 = 20: more than 10% of an extent of 100, the others are far below
    field.densify_and_prune(max_grad=1.0, min_presence=0.0, extent=100.0, prune_large=True)
    assert field.num_points == 7


def test_deformation_starts_as_identity_and_static_never_moves():
    field = make(ReflectionField, n=16)
    deformation = Deformation(3, NET, center=torch.zeros(3), extent=2.0).cuda()
    xyz, rotation, scale = deformation(field, 0.3)
    torch.testing.assert_close(xyz, field.get_xyz)
    torch.testing.assert_close(rotation, field.get_rotation)
    torch.testing.assert_close(scale, field.get_scaling)
    with torch.no_grad():
        deformation.head.bias[:3] = 0.1
    moved, _, _ = deformation(field, 0.3)
    torch.testing.assert_close(moved - field.get_xyz, torch.full_like(moved, 0.2))
    moved.sum().backward()
    assert field._xyz.grad is not None  # the center still gets its gradient through mu + dmu
    static = Deformation(3, NET, static=True).cuda()
    assert static(field, 0.7)[0] is field.get_xyz
    static.training_setup(1e-3)
    assert static.optimizer is None


def test_checkpoint_resume_continues_exactly(tmp_path):
    torch.manual_seed(1)
    field = make(InterfaceField, n=40)
    deformation = Deformation(2, NET).cuda()
    deformation.training_setup(1e-3)

    def step(f, d, time):
        f.optimizer.zero_grad()
        d.optimizer.zero_grad()
        xyz, _, scale = d(f, time)
        (xyz.square().sum() + scale.sum() + f.get_effective_opacity.sum()).backward()
        f.optimizer.step()
        d.optimizer.step()

    step(field, deformation, 0.2)
    path = str(tmp_path / 'interface' / 'latest.pth')
    save_checkpoint(path, field, deformation, iteration=1)
    step(field, deformation, 0.7)

    state = load_checkpoint(path)
    resumed = InterfaceField(1)
    resumed_deformation = Deformation(2, NET)
    restore_phase(state, resumed, resumed_deformation, opt=OPT, deform_lr=1e-3)
    restore_rng_state(state)
    step(resumed, resumed_deformation, 0.7)
    torch.testing.assert_close(resumed.get_xyz, field.get_xyz)
    torch.testing.assert_close(resumed_deformation.head.weight, deformation.head.weight)

    frozen = InterfaceField(1)
    frozen_deformation = Deformation(2, NET)
    restore_phase(state, frozen, frozen_deformation)
    assert frozen.optimizer is None and not frozen.get_xyz.requires_grad
    assert not any(p.requires_grad for p in frozen_deformation.parameters())
