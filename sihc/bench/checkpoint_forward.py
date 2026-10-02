"""Deterministic checkpoint forward probe; run each implementation in its own process.

This is a compatibility check, not a sampler, quality evaluation or benchmark.
Invoke this script by path with PYTHONPATH pointing at the implementation being
checked. Separate processes avoid collisions between registered custom operators.
"""
import argparse
import json
from pathlib import Path

import torch
from sihc.checkpoint import build_model_from_checkpoint, load_checkpoint, load_model_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--state_key', choices=['model', 'ema', 'ema2'], default='ema')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seed', type=int, default=932)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--reference-output', default='')
    p.add_argument('--atol', type=float, default=0.0)
    p.add_argument('--rtol', type=float, default=0.0)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch size must be positive')
    if not torch.cuda.is_available():
        p.error('Allocate a CUDA device for this forward probe')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    checkpoint = load_checkpoint(args.checkpoint, mmap=True)
    model = build_model_from_checkpoint(checkpoint, attn_backend='flash')
    load_model_state(model, checkpoint, args.state_key)
    model = model.cuda().eval()
    torch.manual_seed(args.seed)  # isolate inputs from constructor RNG consumption
    x = torch.randn(args.batch_size, model.in_channels, model.input_size, model.input_size).cuda()
    t = torch.linspace(.2, .7, args.batch_size, device='cuda')
    y = (torch.arange(args.batch_size, device='cuda') * 975 + 12) % model.num_classes
    # Also exercise trained REPA projectors, when present; no teacher is needed.
    has_repa = bool(model.repa_projectors)
    probe = torch.compile(model, mode='default', fullgraph=True) if args.compile else model
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        result = probe(x, t, y, return_repa=True) if has_repa else probe(x, t, y)
    output, features = result if has_repa else (result, ())
    tensors = {'prediction': output.cpu(), **{f'repa_{i}': value.cpu() for i, value in enumerate(features)}}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError('Nonfinite forward output')
    errors = {}
    if args.reference_output:
        reference = torch.load(args.reference_output, weights_only=True, map_location='cpu')
        if reference.keys() != tensors.keys():
            raise ValueError('Reference output keys differ')
        for key, value in tensors.items():
            torch.testing.assert_close(value, reference[key], atol=args.atol, rtol=args.rtol)
            errors[key] = float((value.float() - reference[key].float()).abs().max())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)
    report = dict(status='ok', step=int(checkpoint['step']), state_key=args.state_key,
                  batch_size=args.batch_size, seed=args.seed, compiled=args.compile,
                  dtype='bfloat16', output_shapes={k: list(v.shape) for k, v in tensors.items()},
                  comparison_max_abs_error=errors,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  torch_version=str(torch.__version__), device=torch.cuda.get_device_name())
    path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
