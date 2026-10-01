"""Pure PyTorch controls for the compiled routing benchmark.

Both controls retain the original model, context-token semantics and stage sizes.
Literal routes the dense carrier after every event. Factored implements the
appendix algebra with torch operations. Neither calls custom routing kernels.
FP32 accumulation followed by BF16 boundary casts follows the kernel contract;
literal BF16 writes introduce additional rounding, so equality is algebraic,
not bitwise, in reduced precision.
"""
from types import MethodType
import torch


def read_torch(x, alpha, grid=16):
    b, h, _, c = x.shape
    k = h // grid
    # Keep NHWC physical layout: no dense carrier permutation/copy is needed.
    cells = x.reshape(b, grid, k, grid, k, c)
    weights = alpha.T.reshape(1, 1, k, 1, k, c)
    return (cells.float() * weights.float()).sum((2, 4)).reshape(b, grid * grid, c).to(x.dtype)


def write_torch(x, updates, betas, grid=16):
    b, h, _, c = x.shape
    k = h // grid
    value = x.reshape(b, grid, k, grid, k, c).float()
    for dz, beta in zip(updates, betas):
        delta = dz.reshape(b, grid, 1, grid, 1, c).float()
        weight = beta.T.reshape(1, 1, k, 1, k, c).float()
        value = value + delta * weight
    return value.reshape_as(x).to(x.dtype)


def triangular_torch(base, updates, gamma):
    value = base.float()
    for j, dz in enumerate(updates):
        value = value + dz.float() * gamma[j].float()
    return value.to(base.dtype)


def torch_stage(self, x0, cond, context_tokens, blocks, stage_start, repa_outputs=None):
    assert repa_outputs is None, 'The controlled backbone benchmark excludes REPA.'
    sublayer = self.benchmark_frequency == 'sublayer'
    factored = self.benchmark_implementation == 'factored'
    connections = ([c for b in blocks for c in (b.attention_connection, b.mlp_connection)]
                   if sublayer else [b.connection for b in blocks])
    alpha = torch.stack([c.alpha(dtype=x0.dtype) for c in connections])
    beta = torch.stack([c.beta(dtype=x0.dtype) for c in connections])
    if factored:
        bases = [read_torch(x0, a) for a in alpha.unbind(0)]
        gamma = (alpha[:, None] * beta[None, :]).sum(-1)
    updates = []
    carrier = x0
    for i, block in enumerate(blocks):
        use_context = stage_start + i >= self.in_context_start
        rope = self.workspace_rope_incontext if use_context else self.workspace_rope
        if sublayer:
            modulation = block.branch.conditioning(cond)
        for part in range(2 if sublayer else 1):
            event = 2 * i + part if sublayer else i
            if factored:
                z = triangular_torch(bases[event], updates, gamma[event]) if updates else bases[event]
            else:
                z = read_torch(carrier, alpha[event])
            seq = torch.cat((context_tokens, z), dim=1) if use_context else z
            if not sublayer:
                dseq = block.branch_update(seq, cond, rope)
            elif part == 0:
                dseq = block.branch.attention_update(seq, modulation, rope)
            else:
                dseq = block.branch.mlp_update(seq, modulation)
            if use_context:
                context_tokens = context_tokens + dseq[:, :self.in_context_len]
                dz = dseq[:, self.in_context_len:].contiguous()
            else:
                dz = dseq.contiguous()
            if factored:
                updates.append(dz)
            else:
                carrier = write_torch(carrier, [dz], [beta[event]])
    if factored:
        # Blockwise production keeps final-write casts separate from gamma.
        write_beta = beta.unbind(0) if sublayer else [c.beta(dtype=x0.dtype) for c in connections]
        carrier = write_torch(x0, updates, write_beta)
    return carrier, context_tokens


def attach(model, implementation, frequency):
    if implementation == 'fused':
        return model
    if implementation not in ('literal', 'factored'):
        raise ValueError(implementation)
    model.benchmark_implementation = implementation
    model.benchmark_frequency = frequency
    model._run_fused_schedule_stage_incontext = MethodType(torch_stage, model)
    return model


def make_model(size, frequency, implementation, *, tiny=False):
    from sihc import get_preset
    from sihc.models.sihc.model import SiHCInContextModel
    from sihc.models.sihc.sublayer import SublayerSiHCInContextModel
    spec = get_preset(size)
    kwargs = dict(patch_size=4, hidden_size=spec['hidden_size'], depth=spec['depth'],
                  num_heads=spec['num_heads'], stage_sizes=spec['stage_sizes'],
                  in_context_len=32, in_context_start=spec['in_context_start'],
                  attn_backend='flash', repa_z_dims=(), repa_depth=None)
    if tiny:
        kwargs.update(hidden_size=64, depth=20, num_heads=4, stage_sizes=(8, 12), in_context_start=4)
    cls = SublayerSiHCInContextModel if frequency == 'sublayer' else SiHCInContextModel
    if frequency == 'block':
        kwargs['kernel_backend'] = 'tuple'
    model = cls(**kwargs)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and ('adaLN_modulation.1' in name or name == 'final_layer.linear'):
                torch.nn.init.normal_(module.weight, std=.001 if not tiny else .005)
        if tiny:
            # Nonuniform learned routing coefficients exercise all slots/paths.
            for name, p in model.named_parameters():
                if 'read_weight' in name or 'write_weight' in name:
                    p.add_(torch.randn_like(p) * .01)
    return attach(model, implementation, frequency), kwargs
