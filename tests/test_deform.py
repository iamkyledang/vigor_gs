"""Module 02: the deformation MLP used by both fields."""

from argparse import Namespace

import pytest
import torch

from scene.deform_model import DeformModel, DeformNetwork, PositionalEncoding, apply_deformation

TRAIN_ARGS = Namespace(position_lr_init=1.6e-4, position_lr_final=1.6e-6, position_lr_delay_mult=0.01, deform_lr_max_steps=40_000)


def small_model(scale_dim=3):
    return DeformModel(scale_dim, depth=4, width=32, device='cpu')


def randomize_heads(network):
    for head in (network.gaussian_warp, network.gaussian_rotation, network.gaussian_scaling):
        torch.nn.init.normal_(head.weight, std=0.1)


def test_positional_encoding_layout():
    enc = PositionalEncoding(3, 4)
    x = torch.tensor([[0.0, 0.5, -1.0]])
    out = enc(x)
    assert out.shape == (1, enc.out_dim) == (1, 3 * (1 + 2 * 4))
    assert torch.equal(out[:, :3], x)
    assert torch.allclose(out[0, 3:7], torch.zeros(4))  # sin(0 * 2^k)
    assert torch.allclose(out[0, 7:11], torch.ones(4))  # cos(0 * 2^k)


@pytest.mark.parametrize('scale_dim', [2, 3])
def test_output_shapes(scale_dim):
    network = DeformNetwork(scale_dim=scale_dim, depth=4, width=32)
    d_xyz, d_rotation, d_scaling = network(torch.rand(50, 3), torch.full((50, 1), 0.3))
    assert d_xyz.shape == (50, 3)
    assert d_rotation.shape == (50, 4)
    assert d_scaling.shape == (50, scale_dim)


def test_fresh_network_is_the_identity():
    model = small_model(scale_dim=2)
    xyz = torch.rand(20, 3) * 4 - 2
    rotation = torch.nn.functional.normalize(torch.randn(20, 4), dim=-1)
    log_scaling = torch.randn(20, 2)

    offsets = model.step(xyz, 0.7)
    assert all(torch.count_nonzero(o) == 0 for o in offsets)

    new_xyz, new_rotation, new_scaling = apply_deformation(xyz, rotation, log_scaling, *offsets)
    assert torch.equal(new_xyz, xyz)
    assert torch.allclose(new_rotation, rotation, atol=1e-6)
    assert torch.allclose(new_scaling, log_scaling.exp())


def test_deformation_keeps_geometry_valid():
    model = small_model(scale_dim=3)
    randomize_heads(model.network)
    xyz = torch.rand(20, 3)
    rotation = torch.nn.functional.normalize(torch.randn(20, 4), dim=-1)

    _, new_rotation, new_scaling = apply_deformation(xyz, rotation, torch.randn(20, 3), *model.step(xyz, 0.2))
    assert torch.allclose(new_rotation.norm(dim=-1), torch.ones(20), atol=1e-6)
    assert (new_scaling > 0).all()


def test_output_depends_on_time_and_position():
    model = small_model()
    randomize_heads(model.network)
    xyz = torch.rand(20, 3)

    at_start = model.step(xyz, 0.0)[0]
    at_end = model.step(xyz, 1.0)[0]
    assert not torch.allclose(at_start, at_end, atol=1e-4)
    assert not torch.allclose(at_start[0], at_start[1], atol=1e-5)


def test_training_reaches_every_layer_but_not_the_canonical_centers():
    model = small_model()
    model.train_setting(TRAIN_ARGS, spatial_lr_scale=5.0)
    xyz = torch.rand(30, 3, requires_grad=True)
    target = torch.full((30, 3), 0.05)

    def loss_fn():
        return (model.step(xyz, 0.5)[0] - target).abs().mean()

    first = loss_fn()
    first.backward()
    assert model.network.gaussian_warp.weight.grad.abs().sum() > 0
    assert xyz.grad is None  # centers are detached before entering the network

    for _ in range(200):
        model.optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        model.optimizer.step()
    assert loss.item() < 0.5 * first.item()
    assert model.network.linear[0].weight.grad.abs().sum() > 0


def test_time_noise_jitters_the_timestamp():
    model = small_model()
    randomize_heads(model.network)
    xyz = torch.rand(10, 3)
    clean = model.step(xyz, 0.5)[0]
    assert torch.equal(clean, model.step(xyz, 0.5)[0])
    assert not torch.equal(clean, model.step(xyz, 0.5, time_noise=0.1)[0])


def test_learning_rate_decays():
    model = small_model()
    model.train_setting(TRAIN_ARGS, spatial_lr_scale=5.0)
    assert model.update_learning_rate(0) == pytest.approx(1.6e-4 * 5.0)
    assert model.update_learning_rate(40_000) == pytest.approx(1.6e-6)
    assert model.optimizer.param_groups[0]['lr'] == pytest.approx(1.6e-6)


def test_save_and_load_round_trip(tmp_path):
    model = small_model(scale_dim=2)
    randomize_heads(model.network)
    model.save_weights(str(tmp_path), 100)
    model.save_weights(str(tmp_path), 300)

    restored = small_model(scale_dim=2)
    assert restored.load_weights(str(tmp_path)) == 300
    xyz = torch.rand(10, 3)
    for a, b in zip(model.step(xyz, 0.4), restored.step(xyz, 0.4)):
        assert torch.equal(a, b)


def test_freeze_stops_gradients():
    model = small_model()
    model.train_setting(TRAIN_ARGS)
    model.freeze()
    assert all(not p.requires_grad for p in model.network.parameters())
    assert model.optimizer is None


def test_from_args_uses_model_params():
    model = DeformModel.from_args(2, Namespace(deform_depth=3, deform_width=16, deform_xyz_multires=4, deform_time_multires=2), device='cpu')
    assert len(model.network.linear) == 3
    assert model.network.linear[0].out_features == 16
    assert model.network.input_ch == 3 * 9 + 5
    assert DeformModel.from_args(3, Namespace(), device='cpu').network.linear[0].out_features == 256
