"""Input-preserving controls and two-forward SAM. No added inference network.

SAM reference: Foret et al., ICLR 2021, https://arxiv.org/abs/2010.01412.
This implementation uses AdamW as the base update and one global L2 ball over
active trainable parameters. Stochastic optical noise/dropout is replayed for
the two passes. Parameters are restored exactly before the base optimizer step.
"""
from pathlib import Path
import json
import math
import torch

PROFILES = ('preserve_adam', 'preserve_sam', 'preserve_fullfield_sam', 'preserve_fullfield_both_sam',
            'regularized_control', 'regularized_phase05', 'domain_mixed', 'domain_curriculum', 'domain_target_control',
            'domain_refine_control', 'domain_refine_wide', 'domain_refine_views',
            'domain_distill_light', 'domain_distill_strong')


def overlay_config(config, profile):
    if profile.startswith('domain_distill_'):
        config=overlay_config(config,'domain_refine_wide')
        overlay=json.loads(Path(__file__).with_name('domain_distillation.json').read_text(encoding='utf-8'))
        for key,value in overlay['common'].items():
            if isinstance(value,dict) and isinstance(config.get(key),dict):config[key].update(value)
            else:config[key]=value
        config.update(overlay['profiles'][profile])
        return config
    if profile.startswith('domain_refine_'):
        config=overlay_config(config,'domain_mixed')
        overlay=json.loads(Path(__file__).with_name('domain_refinement.json').read_text(encoding='utf-8'))
        for key,value in overlay['common'].items():
            if isinstance(value,dict) and isinstance(config.get(key),dict):config[key].update(value)
            else:config[key]=value
        config.update(overlay['profiles'][profile])
        return config
    filename = 'domain_expansion.json' if profile.startswith('domain_') else 'regularization.json' if profile.startswith('regularized_') else 'generalization.json'
    overlay = json.loads(Path(__file__).with_name(filename).read_text(encoding='utf-8'))
    for key, value in overlay['common'].items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    config.update(overlay['profiles'][profile])
    return config


def apply_contract(payload, config):
    metadata = dict(payload['metadata'])
    if metadata.get('fusion_alpha_min', 0) <= .4:
        raise ValueError('Generalization controls require alpha>0.4 source weights')
    metadata.update(input_preprocessing=config['input_preprocessing'],
                    ccd_readout_modes=config['ccd_readout_modes'])
    if 'phase_dropout' in config:metadata['phase_dropout']=config['phase_dropout']
    return dict(payload, metadata=metadata)


def parameter_decay(name, parameter, kind, strength):
    # Never pull raw phase logits towards zero (pi), alpha, biases or LN scales.
    return strength if kind not in ('phase','router','alpha') and parameter.ndim>=2 and name.endswith('weight') else 0.


def backward_with_sam(closure, optimizer, rho=0.):
    """closure returns a dict containing scalar loss and detached log metrics.

    Does NOT call optimizer.step; caller clips the second gradient and steps.
    Restores all perturbed parameters even after an exception/KeyboardInterrupt.
    SAM does not perturb frozen or zero-learning-rate parameter groups.
    """
    if not math.isfinite(rho) or rho < 0:
        raise ValueError('SAM rho must be finite and nonnegative')
    active = [p for g in optimizer.param_groups if g['lr'] > 0 for p in g['params'] if p.requires_grad]
    inactive = [p for g in optimizer.param_groups if g['lr'] == 0 for p in g['params']]
    device = active[0].device if active else torch.device('cpu')
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    optimizer.zero_grad(set_to_none=True)
    result = closure()
    if not torch.isfinite(result['loss']):
        raise RuntimeError('Nonfinite nominal loss')
    result['loss'].backward()
    for p in inactive:
        p.grad = None
    if rho == 0:
        return result, dict(rho=0., loss_gap=0., perturbation_norm=0.)
    gradients = [p for p in active if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in gradients])) if gradients else torch.tensor(0.,device=device)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite SAM gradient')
    if float(norm) == 0:
        return result, dict(rho=rho, loss_gap=0., perturbation_norm=0.)
    cpu_after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    saved = []
    scale = rho / (float(norm) + 1e-12)
    try:
        with torch.no_grad():
            for p in gradients:
                saved.append((p, p.detach().clone()))
                p.add_(p.grad, alpha=scale)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, device)
        perturbed = closure()
        if not torch.isfinite(perturbed['loss']):
            raise RuntimeError('Nonfinite perturbed SAM loss')
        perturbed['loss'].backward()
        for p in inactive:
            p.grad = None
        gap = float(perturbed['loss'].detach() - result['loss'].detach())
    finally:
        with torch.no_grad():
            for p, original in saved:
                p.copy_(original)
        torch.set_rng_state(cpu_after)
        if cuda_after is not None:
            torch.cuda.set_rng_state(cuda_after, device)
    return result, dict(rho=rho, loss_gap=gap, perturbation_norm=rho)
