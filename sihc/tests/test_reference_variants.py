from __future__ import annotations

import pytest
import torch

from sihc.models import build_model
from sihc.models.sihc import SIHC_MODELS
from sihc.models.sihc.components import (
    DirectPatchEmbedNHWC,
    VisionRotaryEmbeddingFast,
    WorkspaceJiTBranch,
)
from sihc.models.sihc.reference import (
    ReferenceSiHCModel,
    SublayerReferenceSiHCModel,
)


def _small_model(model_cls, patch_size: int):
    return model_cls(
        patch_size=patch_size,
        hidden_size=16,
        depth=2,
        num_heads=4,
        bottleneck_dim=8,
        workspace_grid=16,
        fused_stage_size=2,
        attn_backend="math",
    )


def _enable_nonzero_updates(model) -> None:
    for block in model.blocks:
        torch.nn.init.normal_(block.branch.adaLN_modulation[-1].weight, std=0.01)
        torch.nn.init.normal_(block.branch.adaLN_modulation[-1].bias, std=0.01)
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.01)


def test_reference_registry_exposes_matched_p8_models() -> None:
    with torch.device("meta"):
        fused = build_model("sihc_1x12_d768_b8", attn_backend="math")
        p8 = build_model("sihc_reference_1x12_d768_b8", attn_backend="math")
        sublayer = build_model(
            "sihc_sublayer_reference_1x12_d768_b8", attn_backend="math"
        )

    assert fused.patch_size == 8
    assert isinstance(fused.x_embedder, DirectPatchEmbedNHWC)
    assert fused.x_embedder.proj.weight.shape == (768, 3, 8, 8)
    assert fused.high_grid == 32
    assert fused.blocks[0].connection.cell_tokens == 4
    assert fused.stage_sizes == (12,)

    assert p8.patch_size == 8
    assert isinstance(p8.x_embedder, DirectPatchEmbedNHWC)
    assert p8.high_grid == 32
    assert p8.blocks[0].connection.cell_tokens == 4
    assert p8.stage_sizes == (12,)
    assert sum(
        name.endswith("connection.read_weight")
        for name, _ in p8.named_parameters()
    ) == 12

    assert sublayer.patch_size == 8
    assert isinstance(sublayer.x_embedder, DirectPatchEmbedNHWC)
    assert sublayer.high_grid == 32
    assert sublayer.blocks[0].attention_connection.cell_tokens == 4
    assert sum(
        name.endswith("read_weight") for name, _ in sublayer.named_parameters()
    ) == 24
    assert sum(
        name.endswith("write_weight") for name, _ in sublayer.named_parameters()
    ) == 24


def test_all_registered_sihc_models_default_to_direct_patch_embedding() -> None:
    for model_name in SIHC_MODELS:
        with torch.device("meta"):
            model = build_model(model_name, attn_backend="math")
        assert isinstance(model.x_embedder, DirectPatchEmbedNHWC), model_name
        parameter_names = {name for name, _ in model.x_embedder.named_parameters()}
        assert parameter_names == {"proj.weight", "proj.bias"}, model_name


def test_p8_cell_layout_round_trips_and_preserves_slot_order() -> None:
    model = _small_model(ReferenceSiHCModel, patch_size=8)
    high = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32, 1)
    cells = model._high_to_cells(high)

    assert cells.shape == (1, 256, 4, 1)
    torch.testing.assert_close(
        cells[0, 0, :, 0], torch.tensor([0.0, 1.0, 32.0, 33.0])
    )
    torch.testing.assert_close(model._cells_to_high(cells), high)


def test_identity_initialized_reference_stage_matches_dense_residual_recurrence() -> None:
    model = _small_model(ReferenceSiHCModel, patch_size=8)
    high = torch.randn(2, 32, 32, 5)
    layers = 3
    slots = 4
    alpha = torch.full((layers, 5, slots), 1.0 / slots)
    beta = torch.ones(layers, 5, slots)
    functions = [
        lambda z, scale=scale: torch.tanh(z * scale)
        for scale in (0.1, 0.2, 0.3)
    ]

    actual, _, updates = model._reference_stage(high, alpha, beta, functions)
    cells = model._high_to_cells(high)
    z = cells.mean(dim=2)
    expected_updates = []
    for function in functions:
        update = function(z)
        expected_updates.append(update)
        z = z + update
    expected_cells = cells + torch.stack(expected_updates).sum(dim=0).unsqueeze(2)

    torch.testing.assert_close(actual, model._cells_to_high(expected_cells))
    for actual_update, expected_update in zip(
        updates, expected_updates, strict=True
    ):
        torch.testing.assert_close(actual_update, expected_update)


def test_split_branch_methods_preserve_existing_block_forward() -> None:
    torch.manual_seed(7)
    branch = WorkspaceJiTBranch(32, 4, attn_backend="math")
    rope = VisionRotaryEmbeddingFast(4, 4)
    z = torch.randn(2, 16, 32)
    cond = torch.randn(2, 32)

    modulation = branch.conditioning(cond)
    dz_attention = branch.attention_update(z, modulation, rope)
    dz_mlp = branch.mlp_update(z + dz_attention, modulation)

    torch.testing.assert_close(branch(z, cond, rope), dz_attention + dz_mlp)


@pytest.mark.parametrize(
    ("model_cls", "patch_size"),
    ((ReferenceSiHCModel, 8), (SublayerReferenceSiHCModel, 8)),
)
def test_small_reference_model_forward_backward(model_cls, patch_size: int) -> None:
    torch.manual_seed(11 + patch_size)
    model = _small_model(model_cls, patch_size)
    _enable_nonzero_updates(model)
    output = model(
        torch.randn(1, 3, 256, 256),
        torch.rand(1),
        torch.zeros(1, dtype=torch.long),
    )
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (1, 3, 256, 256)
    first_connection = (
        model.blocks[0].connection
        if hasattr(model.blocks[0], "connection")
        else model.blocks[0].attention_connection
    )
    assert first_connection.read_weight.grad is not None
    assert torch.isfinite(first_connection.read_weight.grad).all()
