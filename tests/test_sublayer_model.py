import copy

import pytest
import torch

from sihc.activation_checkpointing import configure_activation_checkpointing
from sihc.checkpoint import build_model_from_checkpoint, load_checkpoint, load_model_state
from sihc.models import build_model
from sihc.models.sihc.reference import SublayerReferenceSiHCModel, enable_reference_schedule
from sihc.models.sihc.sublayer import SublayerSiHCModel


@pytest.mark.parametrize("patch", [4, 8])
def test_gemm_stem_preserves_conv_weights_and_optimizer_resume(patch):
    from sihc.models.sihc.components import DirectPatchEmbedNHWC
    from sihc.models.sihc.sublayer import _DirectPatchEmbedGEMM

    torch.manual_seed(292)
    kwargs = dict(img_size=32, patch_size=patch, embed_dim=16)
    old = DirectPatchEmbedNHWC(**kwargs)
    new = _DirectPatchEmbedGEMM(**kwargs)
    old_opt = torch.optim.AdamW(old.parameters(), lr=2e-4)
    new_opt = torch.optim.AdamW(new.parameters(), lr=2e-4)
    x = torch.randn(2, 3, 32, 32)
    old(x).square().mean().backward()
    old_opt.step()
    new.load_state_dict(copy.deepcopy(old.state_dict()), strict=True)
    new_opt.load_state_dict(copy.deepcopy(old_opt.state_dict()))
    for a, b in zip(old.parameters(), new.parameters()):
        assert a.shape == b.shape and a.stride() == b.stride()
        for key in ("exp_avg", "exp_avg_sq"):
            assert old_opt.state[a][key].stride() == new_opt.state[b][key].stride()
    old_opt.zero_grad(set_to_none=True)
    new_opt.zero_grad(set_to_none=True)
    x = torch.randn_like(x)
    expected, actual = old(x), new(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    old_opt.step()
    new_opt.step()
    for a, b in zip(old.parameters(), new.parameters()):
        torch.testing.assert_close(b, a, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("patch", [4, 8])
def test_checkpoint_roundtrip_raw_and_emas(tmp_path, patch):
    kwargs = dict(hidden_size=16, num_heads=4, depth=8, fused_stage_size=8, patch_size=patch)
    model = SublayerSiHCModel(**kwargs)
    state = model.state_dict()
    ckpt = dict(model=state, ema=dict(model.named_parameters()), ema2=dict(model.named_parameters()),
                meta=dict(model=f"sihc_sublayer_1x12_d768_b{patch}", model_kwargs=kwargs))
    path = tmp_path / "roundtrip.pt"
    torch.save(ckpt, path)
    ckpt = load_checkpoint(path)
    reference = SublayerReferenceSiHCModel(**kwargs)
    reference.load_state_dict(state, strict=True)
    for key in ("model", "ema", "ema2"):
        loaded = build_model_from_checkpoint(ckpt, attn_backend="math")
        load_model_state(loaded, ckpt, key)
        assert loaded.stage_sizes == (8,)
        for name, value in loaded.state_dict().items():
            torch.testing.assert_close(value, state[name])
        enable_reference_schedule(loaded)
        with torch.no_grad():
            assert loaded(torch.randn(1, 3, 256, 256), torch.rand(1), torch.zeros(1, dtype=torch.long)).shape == (1, 3, 256, 256)


def test_registry_initialization_and_mixed_stages():
    with torch.device("meta"):
        for patch in (4, 8):
            model = build_model(f"sihc_sublayer_1x12_d768_b{patch}")
            assert model.depth == 12 and model.stage_sizes == (12,)
            assert model.patch_embed_type == "direct"
            assert model.blocks[0].attention_connection.read_weight.shape == (768, (16 // patch) ** 2)
        mixed = SublayerSiHCModel(depth=28, stage_sizes=(8, 12, 8))
        assert mixed.stage_sizes == (8, 12, 8)
    small = SublayerSiHCModel(hidden_size=16, num_heads=4, depth=8, fused_stage_size=8)
    for block in small.blocks:
        for connection in (block.attention_connection, block.mlp_connection):
            torch.testing.assert_close(connection.read_weight, torch.full_like(connection.read_weight, 1 / 16))
            torch.testing.assert_close(connection.write_weight, torch.ones_like(connection.write_weight))
    with pytest.raises(ValueError, match="REPA"):
        SublayerSiHCModel(repa_depth=8, repa_z_dims=(768,))
    with pytest.raises(ValueError, match="sublayer"):
        configure_activation_checkpointing(small, "branch")
    keys = set(small.state_dict())
    assert configure_activation_checkpointing(small, "mlp") == 8
    assert set(small.state_dict()) == keys
