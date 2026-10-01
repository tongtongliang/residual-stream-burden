"""SiHC model registry: fused block-wise/sublayer-wise and readable references."""
from .sihc import SIHC_MODELS
from .baselines import (
    PlainJiTB16, PlainJiTL16, PlainJiTXL16, BottleneckJiTB16,
    MHCJiTB16, MHCSlotJiTB16,
)

BASELINE_MODELS = {
    "jit_b16": PlainJiTB16,
    "jit_b16_bottleneck_r128": BottleneckJiTB16,
    "jit_l16": PlainJiTL16,
    "jit_xl16": PlainJiTXL16,
    "mhc_jit_b16_n4": MHCJiTB16,
    "mhc_jit_b16_n4_p8": MHCSlotJiTB16,
}

MODEL_NAMES = tuple(SIHC_MODELS) + tuple(BASELINE_MODELS)
MODEL_CHOICES = MODEL_NAMES
DEFAULT_MODEL = "sihc_sublayer_1x12_d768_b4_ctx32s4"
MODEL_ALIASES = {name.replace("sihc_", "shc_", 1): name for name in SIHC_MODELS}


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def build_model(name: str = DEFAULT_MODEL, **kwargs):
    name = canonical_model_name(name)
    factory = SIHC_MODELS.get(name) or BASELINE_MODELS.get(name)
    if factory is None:
        raise ValueError(f"unknown model: {name}; available={MODEL_NAMES}")
    return factory(**kwargs)


__all__ = ["MODEL_NAMES", "MODEL_CHOICES", "DEFAULT_MODEL", "MODEL_ALIASES",
           "SIHC_MODELS", "canonical_model_name", "build_model"]
