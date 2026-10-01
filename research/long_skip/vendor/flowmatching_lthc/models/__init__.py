"""Slim registry for the Group 8 JiT-B/16 long-skip run.

The full flowmatching_lthc registry also pulls LTHC/Triton models. This
package only exposes the backbones this experiment resumes.
"""

from .jit_standard import JiT_models

MODEL_NAMES = (
    "jit_b16",
    "jit_b16_fullpatch",
    "jit_b16_fullpatch_longskip",
)


def build_model(name: str = "jit_b16_fullpatch_longskip", **kwargs):
    if name == "jit_b16":
        return JiT_models["JiT-B/16"](**kwargs)
    if name == "jit_b16_fullpatch":
        return JiT_models["JiT-B/16-FullPatch"](**kwargs)
    if name == "jit_b16_fullpatch_longskip":
        return JiT_models["JiT-B/16-FullPatch-LongSkip"](**kwargs)
    raise ValueError(f"group8 registry does not include model {name}")
