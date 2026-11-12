"""Convert the selected model to a fused final readout without retraining."""
from dataclasses import replace

import torch

if __package__:
    from . import _paths
else:
    import _paths

if __package__ and __package__ != 'tools':
    from ..model import LightGenTemporal
else:
    from model import LightGenTemporal


def fold_final_projection(source, detector_processing=None):
    if source.cfg.fused_readout:
        raise ValueError('The source final projection has already been fused')
    processing = detector_processing or source.cfg.detector_processing
    config = replace(source.cfg, fused_readout=True, detector_processing=processing)
    target = LightGenTemporal(config)
    original = source.state_dict()
    state = target.state_dict()
    for name in state:
        if name not in ('readout.1.weight', 'readout.1.bias'):
            state[name].copy_(original[name].cpu())
    count = 4 + config.prompt_length
    blocks = original['readout.1.weight'].cpu().reshape(config.readout_hidden, count, config.width)
    projection = original['video_global.output.weight'].cpu()
    bias = original['video_global.output.bias'].cpu()
    state['readout.1.weight'].copy_(
        torch.einsum('htc,cd->htd', blocks, projection).reshape(config.readout_hidden, count * 96))
    state['readout.1.bias'].copy_(original['readout.1.bias'].cpu() + torch.einsum('htc,c->h', blocks, bias))
    target.load_state_dict(state, strict=True)
    return target.eval()


def prune_hidden_units(source, features, hidden_width):
    """Select ReLU units by TRAIN activation contribution, preserving their basis."""
    if not source.cfg.fused_readout or not 0 < hidden_width < source.cfg.readout_hidden:
        raise ValueError('Hidden-unit pruning requires a narrower fused readout')
    with torch.no_grad():
        activations = source.readout[2](source.readout[1](features.flatten(-2)))
        importance = activations.square().mean(0).sqrt() * source.readout[3].weight.square().sum(0).sqrt()
        selected = importance.topk(hidden_width).indices.sort().values
        removed = torch.ones(source.cfg.readout_hidden, dtype=torch.bool, device=selected.device)
        removed[selected] = False
        bias_correction = source.readout[3].weight[:, removed] @ activations[:, removed].mean(0)
    target = LightGenTemporal(replace(source.cfg, readout_hidden=hidden_width))
    state = target.state_dict()
    original = source.state_dict()
    for name in state:
        if not name.startswith('readout.'):
            state[name].copy_(original[name].cpu())
    state['readout.1.weight'].copy_(original['readout.1.weight'][selected].cpu())
    state['readout.1.bias'].copy_(original['readout.1.bias'][selected].cpu())
    state['readout.3.weight'].copy_(original['readout.3.weight'][:, selected].cpu())
    state['readout.3.bias'].copy_((original['readout.3.bias'] + bias_correction).cpu())
    target.load_state_dict(state, strict=True)
    return target.eval(), selected.cpu().tolist()
