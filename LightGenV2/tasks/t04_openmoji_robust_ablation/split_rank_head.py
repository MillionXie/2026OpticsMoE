"""Pre-training editor compression with the original rank64 decoder unchanged.

Task-local factory: shared modules and old checkpoint inference remain untouched.
The saved editor_rank is required when strictly rebuilding this architecture.
"""
import torch
from torch import nn
from LightGenV2.tasks.t04_semantic_interaction import training as t
from . import train as base


def build_model(cfg, device):
    rank = int(getattr(cfg, 'editor_rank', 64))
    if rank not in (32, 48, 64) or cfg.shared_readout_variant != 'lowrank64':
        raise ValueError('Only editor32/48/64 with original decoder64 supported')
    model = t.build_model(cfg, device)
    if rank != 64:
        for layer in model.shared_readout.editor:
            layer.condition = nn.Sequential(nn.Linear(192, rank, bias=False),
                                            nn.Linear(rank, 384)).to(device)
            layer.pointwise = nn.Sequential(nn.Conv2d(192, rank, 1, bias=False),
                                            nn.Conv2d(rank, 192, 1, bias=False)).to(device)
        model.shared_readout.contract += f'_editor{rank}_decoder64'
    return model


def adapted_source_state(source, target, rank):
    result = dict(source)
    for group in (0, 1):
        for suffix in ('condition', 'pointwise'):
            base.factorize_pointwise(result, f'shared_readout.editor.{group}.{suffix}', rank)
    base.factorize_pointwise(result, 'shared_readout.decoder.pre.1', 64)
    if set(result) != set(target):
        raise ValueError('Split-rank warmstart key mismatch')
    for key, value in result.items():
        if value.shape != target[key].shape:
            raise ValueError(f'Split-rank warmstart shape mismatch: {key}')
    return result
