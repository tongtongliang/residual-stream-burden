"""Export one inference state with a strict metadata allowlist, without run provenance."""
import argparse
from pathlib import Path

import torch
from sihc.checkpoint import (
    build_model_from_checkpoint, checkpoint_model_name, checkpoint_model_kwargs,
    checkpoint_prediction, load_checkpoint, resolve_model_state, load_model_state,
)

# Architecture and validated execution fields; source args, paths, identities and optimizer state never pass through.
ARCHITECTURE_KEYS = {
    'input_size', 'patch_size', 'in_channels', 'hidden_size', 'depth', 'num_heads',
    'mlp_ratio', 'attn_drop', 'proj_drop',
    'num_classes', 'workspace_grid', 'fused_stage_size', 'stage_sizes',
    'in_context_len', 'in_context_start', 'patch_embed_type', 'bottleneck_dim',
    'repa_depth', 'repa_z_dims', 'repa_projector_dim', 'repa_source', 'kernel_backend',
}


def export_state(source, state_key='ema'):
    name = checkpoint_model_name(source)
    prediction = checkpoint_prediction(source)
    kwargs = {k: v for k, v in checkpoint_model_kwargs(source).items() if k in ARCHITECTURE_KEYS}
    for key, value in kwargs.items():
        if key == 'kernel_backend':
            valid = value in ('legacy', 'tuple')
        elif key == 'patch_embed_type':
            valid = value in ('direct', 'factorized_linear')
        elif key == 'repa_source':
            valid = value in ('dz', 'z_plus_dz')
        elif isinstance(value, (list, tuple)):
            valid = all(type(v) is int and v > 0 for v in value)
        else:
            valid = value is None or type(value) in (int, float)
        if not valid:
            raise ValueError(f'Unsupported architecture metadata value: {key}')
    clean = {'format_version': 2, 'args': {'model': name, 'prediction': prediction},
             'meta': {'model': name, 'model_kwargs': kwargs, 'prediction': prediction}}
    # Reconstruct using sanitized metadata and validate every parameter name/shape.
    clean['model'] = source[state_key]
    with torch.device('meta'):
        model = build_model_from_checkpoint(clean, attn_backend='math')
    weights = resolve_model_state(model, source[state_key], parameters_only=state_key != 'model')
    clean['model'] = {name: value.detach().cpu() for name, value in weights.items()}
    # Public inference weights can be used with either standard state selector.
    clean['ema'] = clean['model']
    load_model_state(model, clean, 'ema', assign=True)
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--state_key', choices=['model', 'ema', 'ema2'], default='ema')
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() or path.resolve() == Path(args.checkpoint).resolve():
        parser.error('Choose a new output file; the input is never overwritten')
    state = export_state(load_checkpoint(args.checkpoint, mmap=True), args.state_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    print(f'Exported {args.state_key} inference weights to {path}')


if __name__ == '__main__':
    main()
