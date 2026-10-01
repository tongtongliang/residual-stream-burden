"""Portable JSON configuration loading with strict environment expansion."""

import json
import os
import re
from pathlib import Path


VARIABLE = re.compile(r"\$\{([^}]+)\}")

ARCHITECTURE_FIELDS = {
    "input_size": "input_size",
    "image_size": "input_size",
    "patch_size": "patch_size",
    "workspace_grid": "workspace_grid",
    "hidden_size": "hidden_size",
    "depth": "depth",
    "num_heads": "num_heads",
    "num_classes": "num_classes",
    "fused_stage_size": "fused_stage_size",
    "stage_sizes": "stage_sizes",
    "in_context_len": "in_context_len",
    "in_context_start": "in_context_start",
    "local_patch_size": "local_patch_size",
    "mhc_n": "mhc_n",
}


def _expand(value):
    if isinstance(value, str):
        missing = sorted({name for name in VARIABLE.findall(value) if name not in os.environ})
        if missing:
            raise RuntimeError(f"unresolved environment variables: {', '.join(missing)}")
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict:
    return _expand(json.loads(Path(path).read_text()))


def architecture_expectations(config: dict) -> dict[str, object]:
    """Extract config fields that must agree with the selected model preset."""
    expected: dict[str, object] = {}
    for key, attribute in ARCHITECTURE_FIELDS.items():
        if key not in config:
            continue
        value = config[key]
        if attribute in expected and expected[attribute] != value:
            raise ValueError(
                f"conflicting architecture fields for {attribute}: "
                f"{expected[attribute]!r} versus {value!r}"
            )
        expected[attribute] = value
    return expected


def validate_model_architecture(model, expected: dict[str, object]) -> None:
    """Fail if descriptive config architecture disagrees with its preset."""
    for attribute, configured in expected.items():
        if not hasattr(model, attribute):
            raise ValueError(
                f"model {type(model).__name__} does not expose configured "
                f"architecture field {attribute!r}"
            )
        actual = getattr(model, attribute)
        if isinstance(actual, tuple) and isinstance(configured, list):
            configured = tuple(configured)
        if actual != configured:
            raise ValueError(
                f"model architecture mismatch for {attribute}: "
                f"config={configured!r}, preset={actual!r}"
            )

