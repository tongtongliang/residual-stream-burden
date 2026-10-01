"""Checkpoint discovery, reconstruction, and state-loading helpers."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch


# Checkpoints may carry either the former shc_* spelling or an earlier LTHC-v2 identifier.
# This is deliberately limited to the maintained architecture; removed LTHC v1
# and SharedAdaLN schemas are not migrated.
LEGACY_MODEL_NAMES = {
    "local_thc_v2_fused_final12_adaln_b4": "sihc_1x12_d768_b4",
    "local_thc_v2_fused_2x12_adaln_b4": "sihc_2x12_d768_b4",
    "local_thc_v2_fused_2x12_adaln_d1024_b4": "sihc_2x12_d1024_b4",
    "local_thc_v2_fused_3x8_adaln_b4": "sihc_3x8_d768_b4",
    "local_thc_v2_fused_3x8_adaln_d1024_b4": "sihc_3x8_d1024_b4",
    "local_thc_v2_fused_5x8_adaln_b4": "sihc_5x8_d768_b4",
    "local_thc_v2_fused_5x8_adaln_d1024_b4": "sihc_5x8_d1024_b4",
    "local_thc_v2_fused_5x8_adaln_d1024_b4_ctx32s4": "sihc_5x8_d1024_b4_ctx32s4",
    "local_thc_v2_direct_fused_final12_adaln_b4": "sihc_direct_1x12_d768_b4",
    "single_jit_b16": "jit_b16",
    "mhc_n4_jit_b16": "mhc_jit_b16_n4",
    "mhc_n4_p8_jit_b16": "mhc_jit_b16_n4_p8",
    "ghc_m4n16_p4_jit_b16": "fraction_hc_jit_b16_m4n16_p4",
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _checkpoint_value(ckpt: Mapping[str, Any], key: str, default: Any = None) -> Any:
    args = _as_mapping(ckpt.get("args"))
    if key in args:
        return args[key]
    meta = _as_mapping(ckpt.get("meta"))
    return meta.get(key, default)


def checkpoint_model_name(ckpt: Mapping[str, Any], default: str | None = None) -> str:
    """Return a registered model name without silently choosing another model."""
    from .models import MODEL_NAMES, canonical_model_name

    name = _checkpoint_value(ckpt, "model", default)
    if not name:
        raise KeyError("checkpoint has no model name; pass model_name explicitly")
    name = LEGACY_MODEL_NAMES.get(str(name), str(name))
    name = canonical_model_name(name)
    if name not in MODEL_NAMES:
        raise ValueError(f"checkpoint model {name!r} is not registered; available={MODEL_NAMES}")
    return name


def checkpoint_prediction(ckpt: Mapping[str, Any], default: str | None = None) -> str:
    """Return the clean/velocity training objective recorded by a checkpoint."""
    prediction = _checkpoint_value(ckpt, "prediction", default)
    if prediction not in {"clean", "velocity"}:
        raise ValueError(
            f"checkpoint prediction must be 'clean' or 'velocity', got {prediction!r}"
        )
    return str(prediction)


def _checkpoint_parameter_state(ckpt: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model", "ema", "ema2"):
        state = ckpt.get(key)
        if isinstance(state, Mapping):
            return state
    raise KeyError("checkpoint contains none of: model, ema, ema2")


def _infer_repa_dims(state: Mapping[str, torch.Tensor]) -> tuple[int | None, int | None]:
    first = state.get("repa_projectors.0.0.weight")
    last = state.get("repa_projectors.0.4.weight")
    projector_dim = int(first.shape[0]) if isinstance(first, torch.Tensor) else None
    z_dim = int(last.shape[0]) if isinstance(last, torch.Tensor) else None
    return projector_dim, z_dim


def checkpoint_model_kwargs(
    ckpt: Mapping[str, Any],
    *,
    model_name: str | None = None,
    attn_backend: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover constructor kwargs needed to reproduce the checkpoint tree.

    The frozen REPA teacher is intentionally not part of the student model and
    is not needed for inference. Only the trainable projector configuration is
    reconstructed here.
    """
    kwargs: dict[str, Any] = {}
    meta_kwargs = _as_mapping(_as_mapping(ckpt.get("meta")).get("model_kwargs"))
    kwargs.update(meta_kwargs)
    if "pool_groups" in kwargs and "connection_groups" not in kwargs:
        kwargs["connection_groups"] = kwargs.pop("pool_groups")

    config = dict(_as_mapping(ckpt.get("config")))
    config.update(_as_mapping(ckpt.get("args")))
    args = config
    state = _checkpoint_parameter_state(ckpt)
    factorized_weight = state.get("x_embedder.proj1.weight")
    if isinstance(factorized_weight, torch.Tensor):
        kwargs["patch_embed_type"] = "factorized_linear"
        kwargs["bottleneck_dim"] = int(factorized_weight.shape[0])
    canonical_name = model_name
    if canonical_name is None:
        try:
            canonical_name = checkpoint_model_name(ckpt)
        except (KeyError, ValueError):
            canonical_name = None

    if canonical_name in {
        "jit_b16",
        "mhc_jit_b16_n4",
        "mhc_jit_b16_n4_p8",
        "fraction_hc_jit_b16_m4n16_p4",
    }:
        baseline_keys = (
            "image_size",
            "patch_size",
            "hidden_size",
            "depth",
            "num_heads",
            "num_classes",
            "mhc_n",
            "mhc_backend",
            "ghc_backend",
            "local_patch_size",
        )
        kwargs = {key: args[key] for key in baseline_keys if key in args}

    has_repa_state = any(name.startswith("repa_projectors.") for name in state)
    if bool(args.get("repa")) or has_repa_state:
        inferred_projector_dim, inferred_z_dim = _infer_repa_dims(state)
        repa_depth = args.get("repa_depth", kwargs.get("repa_depth"))
        if repa_depth is None:
            raise KeyError("REPA checkpoint is missing repa_depth metadata")
        z_dim = args.get("repa_z_dim", inferred_z_dim)
        if z_dim is None:
            raise KeyError("REPA checkpoint is missing repa_z_dim metadata")
        projector_dim = args.get("repa_projector_dim", inferred_projector_dim or 2048)
        kwargs.update(
            repa_depth=int(repa_depth),
            repa_z_dims=(int(z_dim),),
            repa_projector_dim=int(projector_dim),
            repa_source=str(args.get("repa_source", kwargs.get("repa_source", "z_plus_dz"))),
        )

    if attn_backend is not None:
        kwargs["attn_backend"] = attn_backend
    if overrides:
        kwargs.update(overrides)
    return kwargs


def build_model_from_checkpoint(
    ckpt: Mapping[str, Any],
    *,
    model_name: str | None = None,
    attn_backend: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    """Construct the exact student parameter tree described by the checkpoint."""
    from .models import build_model

    name = model_name or checkpoint_model_name(ckpt)
    kwargs = checkpoint_model_kwargs(
        ckpt,
        model_name=name,
        attn_backend=attn_backend,
        overrides=overrides,
    )
    return build_model(name, **kwargs)


def _state_errors(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    parameters_only: bool,
) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    expected = (
        {name for name, _ in model.named_parameters()}
        if parameters_only
        else set(model_state)
    )
    missing = sorted(expected.difference(state))
    unexpected = sorted(set(state).difference(model_state))
    mismatched = sorted(
        name
        for name in expected.intersection(state)
        if tuple(model_state[name].shape) != tuple(state[name].shape)
    )
    return missing, unexpected, mismatched


def _error_preview(kind: str, names: list[str]) -> str:
    preview = ", ".join(names[:8])
    suffix = "" if len(names) <= 8 else f", ... ({len(names)} total)"
    return f"{kind}: {preview}{suffix}"


def _migrate_sihc_state_name(name: str) -> str:
    if name.startswith("pool."):
        name = "connection." + name.removeprefix("pool.")
    return name.replace(".pool.", ".connection.")


def resolve_model_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    parameters_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Validate a checkpoint state against a concrete SiHC schema.

    The only key migration retained is ``.pool.`` to ``.connection.`` for
    checkpoints produced by the same architecture before the SiHC rename.
    """
    selected = dict(state)
    missing, unexpected, mismatched = _state_errors(
        model,
        selected,
        parameters_only=parameters_only,
    )
    if not missing and not unexpected and not mismatched:
        return selected

    legacy_selected = {
        _migrate_sihc_state_name(name): tensor
        for name, tensor in selected.items()
    }
    if legacy_selected.keys() != selected.keys():
        missing, unexpected, mismatched = _state_errors(
            model,
            legacy_selected,
            parameters_only=parameters_only,
        )
        if not missing and not unexpected and not mismatched:
            return legacy_selected
    details = []
    if missing:
        details.append(_error_preview("missing", missing))
    if unexpected:
        details.append(_error_preview("unexpected", unexpected))
    if mismatched:
        details.append(_error_preview("shape mismatch", mismatched))
    raise RuntimeError("checkpoint state does not match model: " + "; ".join(details))


def load_model_state(
    model: torch.nn.Module,
    ckpt: Mapping[str, Any],
    key: str = "ema",
    *,
    assign: bool = False,
) -> None:
    """Strictly load raw model, primary EMA, or secondary EMA parameters."""
    if key not in {"model", "ema", "ema2"}:
        raise ValueError(f"unsupported checkpoint state key: {key}")
    if key not in ckpt:
        raise KeyError(f"checkpoint does not contain {key} weights")

    parameters_only = key != "model"
    selected = resolve_model_state(
        model,
        ckpt[key],
        parameters_only=parameters_only,
    )
    if not parameters_only:
        model.load_state_dict(selected, strict=True, assign=assign)
        return

    state = model.state_dict()
    state.update(selected)
    model.load_state_dict(state, strict=True, assign=assign)


def validate_checkpoint_model(
    model: torch.nn.Module,
    ckpt: Mapping[str, Any],
    key: str = "ema",
) -> dict[str, Any]:
    """Validate names and shapes without copying checkpoint tensors."""
    if key not in ckpt:
        raise KeyError(f"checkpoint does not contain {key} weights")
    selected = resolve_model_state(
        model,
        ckpt[key],
        parameters_only=(key != "model"),
    )
    return {
        "state_key": key,
        "state_tensors": len(selected),
        "parameters": sum(param.numel() for param in model.parameters()),
    }


def load_checkpoint(
    path: str | PathLike[str],
    *,
    mmap: bool = False,
    allow_unsafe: bool = False,
) -> Mapping[str, Any]:
    """Load a checkpoint on CPU without executable pickle by default.

    Historical third-party optimizer states may contain NumPy objects that the
    restricted loader rejects. Such a checkpoint can be loaded only through an
    explicit ``allow_unsafe=True`` opt-in after its provenance is verified.
    """
    try:
        return torch.load(
            path,
            map_location="cpu",
            mmap=mmap,
            weights_only=not allow_unsafe,
        )
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            "safe checkpoint load failed; if and only if this checkpoint is "
            "trusted, retry with allow_unsafe=True or "
            "--allow_unsafe_checkpoint"
        ) from exc
