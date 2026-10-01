"""CMuon semantic chunks on the pinned GMuon backend; replicated DDP compatible.

Paper: https://arxiv.org/abs/2608.02502 (Algorithm 3 and Table 8).
Backend: nanovisionx/gmuon @ 4863cc2446f09a54d937c579d4e775c125f5ab2e.
Only workspace attention/FFN/AdaLN matrix weights qualify. Routing is AdamW.
"""
from functools import partial
import math
import re

import torch

BACKEND_COMMIT = '4863cc2446f09a54d937c579d4e775c125f5ab2e'
COEFFICIENTS = (3.4445, -4.7750, 2.0315)
CHUNKS = {'attn.qkv.weight': 3, 'attn.proj.weight': 1,
          'mlp.w12.weight': 2, 'mlp.w3.weight': 1,
          'adaLN_modulation.1.weight': 6}
HOOKS = ('param_split_fn', 'param_recombine_fn', 'adjust_lr')


def split_rows(x, *, chunks):
    if x.ndim != 2 or x.shape[0] % chunks:
        raise ValueError(f'Expected a 2D matrix with rows divisible by {chunks}')
    return list(x.chunk(chunks, dim=0))


def join_rows(xs):
    return torch.cat(xs, dim=0)


def chunk_lr(lr, shape, *, chunks):
    # GMuon passes the individual chunk shape, not the fused weight shape.
    return lr * 0.2 * math.sqrt(max(shape[-2:])) * math.sqrt(chunks)


def partition_parameters(named_parameters):
    groups = {n: [] for n in (1, 2, 3, 6)}
    aux, manifest, seen = [], [], set()
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if id(param) in seen:
            raise ValueError(f'Duplicate optimizer parameter: {name}')
        seen.add(id(param))
        match = re.fullmatch(r'blocks\.\d+\.branch\.(.+)', name)
        chunks = CHUNKS.get(match[1]) if match else None
        if chunks is not None:
            if param.ndim != 2 or param.shape[0] % chunks:
                raise ValueError(f'Invalid workspace projection shape: {name}: {param.shape}')
            groups[chunks].append(param)
        else:
            aux.append(param)
        manifest.append({'name': name, 'shape': list(param.shape),
                         'optimizer': 'gmuon' if chunks else 'adamw', 'chunks': chunks or 0})
    if not any(groups.values()) or not aux:
        raise ValueError('Expected both SiHC workspace projections and AdamW parameters')
    return groups, aux, manifest


class ChunkedGMuon:
    """Checkpoint both optimizers without pickling split/scale callables.

    State includes parameter names/shapes/order and numerical settings: changing
    partition or NS settings on resume is rejected. Execution backend (kernels)
    may change. The scheduler always sees the current, live parameter groups.
    """
    def __init__(self, named_parameters, *, lr, ns_steps=5, momentum=0.95,
                 weight_decay=0.0, use_kernels=True):
        from gram_newton_schulz import Muon
        if ns_steps not in (5, 6):
            raise ValueError('Group 7 supports 5 or 6 NS iterations')
        groups, aux, self.manifest = partition_parameters(named_parameters)
        self.settings = dict(backend_commit=BACKEND_COMMIT, ns_steps=ns_steps,
                             coefficients=list(COEFFICIENTS), momentum=momentum,
                             weight_decay=weight_decay, lr=lr, nesterov=True,
                             chunk_rescale=True, ns_epsilon=1e-7,
                             algorithm='gram_newton_schulz', restarts=[2],
                             adamw_betas=[0.9, 0.95], adamw_eps=1e-8)
        muon_groups = []
        for chunks, params in groups.items():
            if params:
                muon_groups.append(dict(params=params, chunks=chunks, lr_scale=1.0,
                    param_split_fn=partial(split_rows, chunks=chunks),
                    param_recombine_fn=join_rows,
                    adjust_lr=partial(chunk_lr, chunks=chunks)))
        self.muon = Muon(muon_groups, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=True,
                         ns_coefficients=[COEFFICIENTS] * ns_steps,
                         ns_algorithm='gram_newton_schulz', ns_epsilon=1e-7,
                         ns_use_kernels=use_kernels,
                         gram_newton_schulz_restart_iterations=[2])
        self.adamw = torch.optim.AdamW(aux, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                      weight_decay=weight_decay,
                                      fused=all(p.device.type == 'cuda' for p in aux))
        for group in self.adamw.param_groups:
            group['lr_scale'] = 1.0

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def summary(self):
        return {kind: sum(x['optimizer'] == kind for x in self.manifest)
                for kind in ('gmuon', 'adamw')}

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        state = self.muon.state_dict()
        state['param_groups'] = [{k: v for k, v in g.items() if k not in HOOKS}
                                 for g in state['param_groups']]
        return dict(format='sihc_chunked_gmuon_v1', settings=self.settings,
                    manifest=self.manifest, muon=state, adamw=self.adamw.state_dict())

    def load_state_dict(self, state):
        if (state.get('format') != 'sihc_chunked_gmuon_v1'
                or state.get('manifest') != self.manifest
                or state.get('settings') != self.settings):
            raise ValueError('GMuon resume requires matching optimizer settings and parameter manifest')
        hooks = [{k: g[k] for k in HOOKS} for g in self.muon.param_groups]
        self.muon.load_state_dict(state['muon'])
        # Pinned upstream setter replaces _muon_param_groups but leaves the
        # exposed cached list stale; refresh before zero_grad/scheduling/saving.
        self.muon._combined_param_groups = self.muon._muon_param_groups
        for group, hook in zip(self.muon.param_groups, hooks, strict=True):
            group.update(hook)
        self.adamw.load_state_dict(state['adamw'])
