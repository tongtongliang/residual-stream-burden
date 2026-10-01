"""CPU numerical checks for the optional synthetic research example."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from research.toy.data import (
    make_batch, make_dataset, normalized_data_plane, project_to_2d,
    to_velocity, velocity_loss,
)
from research.toy.diagnostics import (
    gradient_snapshot, linear_weight_gradient, normalized_noise_variance, spectrum,
)
from research.toy.models import ModelConfig, ToyModel
from research.toy.train import DEFAULTS, load_checkpoint, save_checkpoint


def tiny(family="sihc"):
    return ModelConfig(family=family, ambient_dim=12, width=8, depth=3,
                       time_embed_dim=8, streams=3)


def randomize(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-.4, .4)


def test_dataset_and_normalized_plane():
    dataset = make_dataset(128, 16, seed=42)
    repeated = make_dataset(128, 16, seed=42)
    for name in dataset:
        np.testing.assert_array_equal(dataset[name], repeated[name])
    np.testing.assert_allclose(project_to_2d(dataset["x0"], dataset),
                               dataset["data_2d"], rtol=2e-6, atol=2e-6)
    x = dataset["x0"]
    basis = normalized_data_plane(dataset)
    np.testing.assert_allclose(x, x @ basis @ basis.T, rtol=2e-5, atol=1e-6)
    assert np.linalg.matrix_rank(x.astype(np.float64), tol=1e-5) == 2


@pytest.mark.parametrize("mode", ["x", "v", "eps"])
def test_velocity_conversion_and_loss(mode):
    torch.manual_seed(1)
    clean, noise = torch.randn(5, 8), torch.randn(5, 8)
    tau = torch.tensor([.1, .3, .5, .7, .9])
    z = (1 - tau[:, None]) * clean + tau[:, None] * noise
    prediction = {"x": clean, "v": noise - clean, "eps": noise}[mode]
    torch.testing.assert_close(to_velocity(prediction, z, tau, mode), noise - clean)
    assert velocity_loss(prediction, (clean, noise, tau, z), mode) < 1e-12


@pytest.mark.parametrize("family", ["fcn", "sihc"])
def test_zero_init_and_taps(family):
    model = ToyModel(tiny(family))
    output, taps = model(torch.randn(4, 12), torch.rand(4), return_taps=True)
    assert torch.count_nonzero(output) == 0
    assert taps["block0.fanin"].shape == (4, 8)
    if family == "sihc":
        assert taps["block0.carrier"].shape == (4, 3, 8)
        assert taps["block0.workspace"].shape == (4, 8)


def test_triangular_matches_explicit_forward_and_backward():
    torch.manual_seed(3)
    triangular = ToyModel(tiny()).double()
    randomize(triangular)  # Exercise nonzero updates and unconstrained routes.
    explicit = deepcopy(triangular)
    z1 = torch.randn(5, 12, dtype=torch.float64, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_()
    tau = torch.rand(5, dtype=torch.float64)
    out1, taps1 = triangular(z1, tau, return_taps=True)
    out2, taps2 = explicit(z2, tau, return_taps=True, schedule="explicit")
    torch.testing.assert_close(out1, out2, rtol=1e-12, atol=1e-12)
    for name in taps1:
        torch.testing.assert_close(taps1[name], taps2[name], rtol=1e-12, atol=1e-12)
    out1.square().sum().backward()
    out2.square().sum().backward()
    torch.testing.assert_close(z1.grad, z2.grad, rtol=1e-11, atol=1e-12)
    for p1, p2 in zip(triangular.parameters(), explicit.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-10, atol=1e-12)
    out_without_taps = triangular(z1.detach(), tau)
    torch.testing.assert_close(out_without_taps, out1, rtol=1e-12, atol=1e-12)


def test_spectrum_and_conditional_noise_definitions():
    metrics, energy = spectrum(np.diag([3., 1.]), center=False)
    np.testing.assert_allclose(energy, [9., 1.])
    assert metrics["stable_rank"] == pytest.approx(10 / 9)
    assert metrics["rank90"] == 1
    assert metrics["rank95"] == 2
    zeros, _ = spectrum(np.ones((3, 4)))
    assert zeros["rank90"] == 0
    h = np.array([[[1., 0.], [-1., 0.]], [[2., 0.], [-2., 0.]]])
    assert normalized_noise_variance(h) == pytest.approx(1.)
    assert normalized_noise_variance(np.ones((3, 2, 4))) == 0


def test_linear_gradient_factorization_with_leading_dimensions():
    torch.manual_seed(5)
    layer = torch.nn.Linear(7, 5).double()
    inputs = torch.randn(3, 4, 7, dtype=torch.float64)
    outputs = layer(inputs)
    outputs.retain_grad()
    outputs.square().mean().backward()
    torch.testing.assert_close(layer.weight.grad, linear_weight_gradient(inputs, outputs.grad))


@pytest.mark.parametrize("family", ["fcn", "sihc"])
def test_gradient_snapshot_and_cleanup(family):
    torch.manual_seed(6)
    model = ToyModel(tiny(family))
    randomize(model)
    generator = torch.Generator().manual_seed(4)
    batch = make_batch(torch.randn(20, 12), 5, generator)
    rows = gradient_snapshot(model, batch, "v")
    assert rows and max(row["factorization_relative_error"] for row in rows) < 1e-5
    assert all(p.grad is None for p in model.parameters())
    assert all(not module._forward_hooks for module in model.modules())


def test_checkpoint_continuation_preserves_rng_and_optimizer(tmp_path):
    torch.manual_seed(7)
    config = dict(DEFAULTS, **vars(tiny("fcn")))
    model = ToyModel(tiny("fcn"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(88)
    data = {k: torch.from_numpy(v) for k, v in make_dataset(24, 12).items()}

    def update(m, opt, rng):
        batch = make_batch(data["x0"], 5, rng)
        opt.zero_grad(set_to_none=True)
        loss = velocity_loss(m(batch[3], batch[2]), batch, "v")
        loss.backward()
        opt.step()
        return loss.detach()

    update(model, optimizer, generator)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, generator, data, config, 1, [])
    expected = update(model, optimizer, generator)
    state = load_checkpoint(path)
    restored = ToyModel(tiny("fcn"))
    restored.load_state_dict(state["model"])
    opt = torch.optim.AdamW(restored.parameters(), lr=9.0)
    opt.load_state_dict(state["optimizer"])
    rng = torch.Generator()
    rng.set_state(state["rng"])
    actual = update(restored, opt, rng)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for p1, p2 in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)
    assert opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
