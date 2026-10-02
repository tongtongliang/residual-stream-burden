"""Named sublayer SiHC recipes; long model names remain checkpoint identities.

XL denotes the 40x1024, five-stage REPA flagship, not a generic DiT-XL.
Presets specify model/projector construction. Training hyperparameters live
in the corresponding portable configs; teacher weights remain external.
"""
from copy import deepcopy

_PRESETS = {
    'B': dict(model='sihc_sublayer_1x12_d768_b4_ctx32s4', depth=12, hidden_size=768,
              num_heads=12, stage_sizes=(12,), in_context_start=4, repa=False,
              teacher=None, config='sihc/configs/sublayer_b.json', cfg=2.9),
    'L': dict(model='sihc_sublayer_2x12_d1024_b4_ctx32s8', depth=24, hidden_size=1024,
              num_heads=16, stage_sizes=(12, 12), in_context_start=8, repa=False,
              teacher=None, config='sihc/configs/sublayer_l.json', cfg=2.4),
    'H': dict(model='sihc_sublayer_8x12x12_d1280_b4_ctx32s8', depth=32, hidden_size=1280,
              num_heads=16, stage_sizes=(8, 12, 12), in_context_start=8, repa=False,
              teacher=None, config='sihc/configs/sublayer_h.json', cfg=2.1),
    'XL': dict(model='sihc_sublayer_5x8_d1024_b4_ctx32s8', depth=40, hidden_size=1024,
               num_heads=16, stage_sizes=(8, 8, 8, 8, 8), in_context_start=8, repa=True,
               teacher='dinov3_vitl16', config='sihc/configs/sublayer_xl_repa.json', cfg=2.4),
}
PRESET_NAMES = tuple(_PRESETS)


def get_preset(size):
    """Return an independent recipe description, including explicit REPA kwargs."""
    key = size.upper()
    if key not in _PRESETS:
        raise ValueError(f'Unknown size {size!r}; choose from {PRESET_NAMES}')
    spec = deepcopy(_PRESETS[key])
    spec['model_kwargs'] = dict(patch_embed_type='direct', in_context_len=32)
    if spec['repa']:
        spec['model_kwargs'].update(repa_depth=8, repa_z_dims=(1024,),
                                    repa_projector_dim=2048, repa_source='z_plus_dz')
    spec['eval_interval'] = (0.1, 0.9)
    spec['activation_checkpoint'] = 'none'
    spec['recommended_checkpoint'] = (
        dict(epoch=520, step=650520, filename='step_00650520.pt', state_key='ema')
        if key == 'XL' else None
    )
    return spec


def build_preset(size, **kwargs):
    """Build the registered architecture, including XL's REPA projector.

    This creates initialized model weights, not a pretrained model or teacher.
    Use build_model_from_checkpoint for existing weights.
    """
    from .models import build_model
    spec = get_preset(size)
    options = spec['model_kwargs']
    options.update(kwargs)
    return build_model(spec['model'], **options)
