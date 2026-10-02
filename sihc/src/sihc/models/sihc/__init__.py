"""Spatially Indexed Hyperconnection models."""

from .model import (
    SIHC_MODELS as FUSED_SIHC_MODELS,
    FusedP8SiHCModel,
    SiHCModel,
    SiHCWriteOnlyModel,
)
from .reference import (
    REFERENCE_SIHC_MODELS,
    ReferenceSiHCModel,
    SublayerReferenceSiHCModel,
    enable_reference_schedule,
)

from .sublayer import SUBLAYER_SIHC_MODELS, SublayerSiHCModel, SublayerSiHCInContextModel

SIHC_MODELS = {
    **FUSED_SIHC_MODELS,
    **REFERENCE_SIHC_MODELS,
    **SUBLAYER_SIHC_MODELS,
}

__all__ = [
    "SiHCModel",
    "FusedP8SiHCModel",
    "SiHCWriteOnlyModel",
    "ReferenceSiHCModel",
    "SublayerReferenceSiHCModel",
    "SublayerSiHCModel",
    "SublayerSiHCInContextModel",
    "enable_reference_schedule",
    "SIHC_MODELS",
]
