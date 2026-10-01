import torch

from sihc.models.baselines import (
    BottleneckJiTB16,
    FactorizedPatchEmbed,
    FullRankPatchEmbed,
    MHCJiTB16,
    MHCSlotJiTB16,
    PlainJiTB16,
    PlainJiTL16,
    PlainJiTXL16,
)
from sihc.objectives import flow_matching_loss, make_flow_batch


def test_plain_model_has_no_bottleneck_or_class_tokens():
    model = PlainJiTB16(
        image_size=8,
        patch_size=4,
        hidden_size=48,
        depth=2,
        num_heads=6,
        num_classes=10,
        attn_backend="math",
    )
    assert isinstance(model.x_embedder, FullRankPatchEmbed)
    names = {name for name, _ in model.named_parameters()}
    assert not any("bottleneck" in name for name in names)
    assert not any("class_token" in name for name in names)
    assert model.x_embedder.proj.weight.shape == (48, 3, 4, 4)


def test_group2_rank_bottleneck_is_linear_and_explicit():
    model = BottleneckJiTB16(
        image_size=8,
        patch_size=4,
        hidden_size=48,
        depth=1,
        num_heads=6,
        num_classes=10,
        bottleneck_dim=8,
        attn_backend="math",
    )
    assert isinstance(model.x_embedder, FactorizedPatchEmbed)
    assert model.x_embedder.proj_in.weight.shape == (8, 3, 4, 4)
    assert model.x_embedder.proj_out.weight.shape == (48, 8, 1, 1)
    assert not any(isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU))
                   for module in model.x_embedder.modules())


def test_group2_jit_l_uses_full_rank_expanding_stem():
    model = PlainJiTL16(
        image_size=8,
        patch_size=4,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=8,
        num_classes=10,
        attn_backend="math",
    )
    assert isinstance(model.x_embedder, FullRankPatchEmbed)
    assert model.x_embedder.proj.weight.shape == (64, 3, 4, 4)


def test_group2_jit_xl_defaults_match_rae_capacity():
    with torch.device("meta"):
        model = PlainJiTXL16(
            image_size=8,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            attn_backend="math",
        )
    assert isinstance(model.x_embedder, FullRankPatchEmbed)
    assert model.hidden_size == 1152
    assert model.depth == 28
    assert model.num_heads == 16


def test_initial_output_is_zero_and_shape_is_image():
    model = PlainJiTB16(
        image_size=8,
        patch_size=4,
        hidden_size=48,
        depth=2,
        num_heads=6,
        num_classes=10,
        attn_backend="math",
    )
    output = model(torch.randn(2, 3, 8, 8), torch.tensor([0.2, 0.8]), torch.tensor([1, 2]))
    assert output.shape == (2, 3, 8, 8)
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_clean_and_velocity_heads_share_velocity_loss_contract():
    torch.manual_seed(7)
    clean = torch.randn(3, 3, 4, 4)
    flow = make_flow_batch(clean, p_mean=-0.8, p_std=0.8, noise_scale=1.0, t_eps=0.05)
    velocity_prediction = torch.randn_like(clean)
    clean_prediction = flow.noisy + (1 - flow.timestep).clamp_min(0.05) * velocity_prediction
    clean_loss, _, recovered = flow_matching_loss(clean_prediction, flow, "clean", 0.05)
    velocity_loss, _, direct = flow_matching_loss(velocity_prediction, flow, "velocity", 0.05)
    torch.testing.assert_close(recovered, direct)
    torch.testing.assert_close(clean_loss, velocity_loss)


def test_forward_features_exposes_all_residual_fanins():
    model = PlainJiTB16(
        image_size=8,
        patch_size=4,
        hidden_size=48,
        depth=2,
        num_heads=6,
        num_classes=10,
        attn_backend="math",
    )
    endpoint, conditioning, fanins = model.forward_features(
        torch.randn(2, 3, 8, 8), torch.tensor([0.2, 0.8]), torch.tensor([1, 2])
    )
    assert endpoint.shape == (2, 4, 48)
    assert conditioning.shape == (2, 48)
    assert len(fanins) == 2
    assert all(tensor.shape == (2, 4, 48) for tensor in fanins)



def test_mhc_n4_reference_exposes_expanded_residual_and_finite_gradients():
    model = MHCJiTB16(
        image_size=8,
        patch_size=4,
        hidden_size=48,
        depth=2,
        num_heads=6,
        num_classes=10,
        attn_backend="math",
        mhc_n=4,
        mhc_backend="reference",
    )
    images = torch.randn(2, 3, 8, 8)
    timesteps = torch.tensor([0.2, 0.8])
    labels = torch.tensor([1, 2])
    endpoint, conditioning, fanins = model.forward_features(images, timesteps, labels)
    output = model(images, timesteps, labels)
    output.square().mean().backward()
    assert endpoint.shape == (2, 4, 48)
    assert conditioning.shape == (2, 48)
    assert all(tensor.shape == (2, 4, 4 * 48) for tensor in fanins)
    assert output.shape == (2, 3, 8, 8)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_mhc_n4_p8_uses_four_named_full_width_slots_and_direct_readout():
    model = MHCSlotJiTB16(
        image_size=32,
        patch_size=16,
        local_patch_size=8,
        hidden_size=768,
        depth=1,
        num_heads=12,
        num_classes=10,
        attn_backend="math",
        mhc_n=4,
        mhc_backend="reference",
    )
    images = torch.randn(1, 3, 32, 32)
    timesteps = torch.tensor([0.4])
    labels = torch.tensor([2])
    embedded = model._embed_tokens(images)
    state = model._slot_state(images)
    output = model(images, timesteps, labels)

    assert embedded.shape == (1, 4, 4, 768)
    assert state.shape == (4, 1, 768, 4)
    assert model.final_layer.linear.out_features == 8 * 8 * 3
    assert output.shape == images.shape
    torch.testing.assert_close(output, torch.zeros_like(output))


