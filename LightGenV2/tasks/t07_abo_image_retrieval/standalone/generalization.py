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
            'domain_refine_control', 'domain_refine_wide', 'domain_refine_views', 'domain_refine_pool500_mix13', 'domain_refine_context7',
            'domain_distill_light', 'domain_distill_strong', 'domain_distill_stronger', 'domain_distill_resumeaux', 'domain_distill_resumeaux_full', 'domain_distill_sharpteacher')


def initialize_category_proxies(head, features, labels, preserve_restored=False):
    """Do not overwrite restored training proxies during epoch-zero preparation.

    The legacy resumeaux control restored optical auxiliary heads but then reset
    category proxies to target class means. Keep that control reproducible;
    resumeaux_full explicitly preserves ALL restored training-only parameters.
    """
    if preserve_restored:
        return 'preserved_checkpoint'
    from torch.nn import functional as F
    with torch.no_grad():
        head.weight.copy_(torch.stack([F.normalize(features[labels==c].mean(0),dim=0)
                                      for c in range(len(head.weight))]))
    return 'target_training_class_means'


def restore_auxiliary_head(head, payload, actual_sha256, expected_sha256):
    """Restore training-only heads from one audited same-label live checkpoint.

    EMA student snapshots can contain a live auxiliary head from another step,
    so this control deliberately accepts only the pinned live starting point.
    No optimizer state is resumed and no head is attached to student inference.
    """
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError('Auxiliary continuation requires the pinned same-label checkpoint')
    if payload.get('selection_variant') != 'live' or payload.get('auxiliary_head_not_used_at_inference') is not True:
        raise ValueError('Auxiliary continuation requires a live training-only head')
    state=payload.get('auxiliary_training_head')
    expected=head.state_dict()
    if not isinstance(state,dict) or state.keys()!=expected.keys():
        raise ValueError('Auxiliary head keys changed')
    if any(state[k].shape!=expected[k].shape or not torch.isfinite(state[k]).all() for k in expected):
        raise ValueError('Auxiliary head shapes or values invalid')
    head.load_state_dict(state,strict=True)


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
    result=dict(payload, metadata=metadata)
    if 'electronic_context_kernels' in config:
        result=expand_electronic_context(result,config['electronic_context_kernels'])
    return result


def expand_electronic_context(payload, kernels):
    """Zero-pad existing depthwise kernels, preserving the starting function.

    Vision is centered; causal language kernels align at their RIGHT edge.
    All optical tensors and other weights are retained unchanged, not copied,
    regenerated, interpolated or reinitialized. Conversion never shrinks kernels.
    """
    if set(kernels)!={'vision','language'}:raise ValueError('Specify both electronic kernel sizes')
    from torch.nn import functional as F
    metadata=dict(payload['metadata']);state=dict(payload['state_dict'])
    previous=metadata.get('electronic_context_kernels',{'vision':3,'language':5})
    for mode in ('vision','language'):
        old=previous[mode];new=kernels[mode]
        if type(new) is not int or new not in (3,5,7) or new<old:
            raise ValueError('Only nonshrinking 3/5/7 electronic kernels supported')
        for index in (0,1):
            name=f'{mode}.blocks.{index}.token_depthwise.weight'
            weight=state[name]
            shape=(192,1,old,old) if mode=='vision' else (192,1,old)
            if tuple(weight.shape)!=shape:raise ValueError('Source electronic kernel metadata mismatch')
            if new!=old:
                pad=(new-old)//2
                state[name]=F.pad(weight,(pad,pad,pad,pad) if mode=='vision' else (new-old,0))
    metadata['electronic_context_kernels']=dict(kernels)
    return dict(payload,metadata=metadata,state_dict=state)


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
