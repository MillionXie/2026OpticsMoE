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

PINNED_TEACHER_PROFILES = ('domain_distill_teacher_continue', 'domain_distill_teacher_continue_sam', 'domain_distill_teacher_continue_softgt', 'domain_distill_teacher_continue_fp32gallery', 'domain_distill_joint_curriculum', 'domain_distill_vision_patch', 'domain_distill_joint_restart', 'domain_distill_joint_restart_softgt', 'domain_distill_joint_merger', 'domain_distill_joint_categorykd', 'domain_distill_joint_routerorigin', 'domain_distill_joint_phasefirst', 'domain_distill_joint_feature8', 'domain_distill_joint_routerradian')
REFIT_TEACHER_PROFILES = ('domain_distill_refit250', 'domain_distill_refit500')

PROFILES = ('preserve_adam', 'preserve_sam', 'preserve_fullfield_sam', 'preserve_fullfield_both_sam',
            'regularized_control', 'regularized_phase05', 'domain_mixed', 'domain_curriculum', 'domain_target_control',
            'domain_refine_control', 'domain_refine_wide', 'domain_refine_views', 'domain_refine_pool500_mix13', 'domain_refine_context7', 'domain_refine_balanced',
            'domain_distill_light', 'domain_distill_strong', 'domain_distill_stronger', 'domain_distill_resumeaux', 'domain_distill_resumeaux_full', 'domain_distill_sharpteacher', 'domain_distill_teacher_agreement', 'domain_distill_aligned_feature', 'domain_distill_feature_mlp', 'domain_distill_teacher_first', 'domain_distill_bounded_aspect', 'domain_distill_position_jitter', 'domain_distill_readout256', 'domain_distill_joint_centerhalf', *PINNED_TEACHER_PROFILES, *REFIT_TEACHER_PROFILES)


def learning_rate_multiplier(config):
    value=config.get('learning_rate_multiplier',1.)
    if not math.isfinite(value) or not 0<value<=1:
        raise ValueError('Continuation learning-rate multiplier must be in (0,1]')
    return float(value)


def phase_only_group_frozen(config, epoch, kind):
    """Training-only warmup: update mask/router phases, freeze EVERYTHING else.

    Unlike legacy optical_warmup (which permits readout/auxiliary updates), this
    freezes alpha, optical electronic readout, all adapters and training heads.
    Keep their gradient paths to phases; clear parameter grads before AdamW so
    neither momentum nor decoupled weight decay changes frozen parameters.
    """
    count=config.get('phase_only_warmup_epochs',0)
    if type(count) is not int or count<0 or type(epoch) is not int or epoch<1:
        raise ValueError('Phase-only warmup requires nonnegative integer count and positive epoch')
    return epoch<=count and kind not in ('phase','router')


def supervised_loss_scale(epoch, config):
    """Optional teacher-first curriculum for final GT losses, never inference.

    Original profiles return exactly1. Optical auxiliary/physical regularizers
    remain outside this scale; GT is reduced, never removed or relabelled.
    """
    warm=config.get('supervised_warmup_epochs',0)
    recovery=config.get('supervised_recovery_epochs',0)
    floor=config.get('supervised_warmup_scale',1.)
    if type(epoch) is not int or epoch<1 or any(type(n) is not int or n<0 for n in (warm,recovery)):
        raise ValueError('Invalid supervised curriculum epoch schedule')
    if not math.isfinite(floor) or not 0<floor<=1:
        raise ValueError('Supervised curriculum scale must be in (0,1]')
    if epoch<=warm:return float(floor)
    return float(floor+(1-floor)*min(1.,(epoch-warm)/recovery)) if recovery else 1.


def merger_learning_rate_multiplier(config):
    value=config.get('merger_learning_rate_multiplier',1.)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<=1:
        raise ValueError('Existing merger learning-rate multiplier must be in (0,1]')
    return float(value)


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
    if profile=='domain_distill_joint_centerhalf':
        config=overlay_config(config,'domain_distill_joint_restart')
        config.pop('teacher_alignment_sha256')
        config.pop('teacher_alignment_origin_checkpoint_sha256')
        overlay=json.loads(Path(__file__).with_name('domain_distillation.json').read_text(encoding='utf-8'))
        config.update(overlay['profiles'][profile])
        return config
    if profile in ('domain_distill_joint_restart', 'domain_distill_joint_restart_softgt', 'domain_distill_joint_merger', 'domain_distill_joint_categorykd', 'domain_distill_joint_routerorigin', 'domain_distill_joint_phasefirst', 'domain_distill_joint_feature8', 'domain_distill_joint_routerradian'):
        parent='domain_distill_joint_curriculum' if profile=='domain_distill_joint_restart' else 'domain_distill_joint_restart'
        config=overlay_config(config,parent)
        overlay=json.loads(Path(__file__).with_name('domain_distillation.json').read_text(encoding='utf-8'))
        config.update(overlay['profiles'][profile])
        return config
    if profile == 'domain_distill_readout256':
        config=overlay_config(config,'domain_distill_refit250')
        overlay=json.loads(Path(__file__).with_name('domain_distillation.json').read_text(encoding='utf-8'))
        config.update(overlay['profiles'][profile])
        return config
    if profile in REFIT_TEACHER_PROFILES:
        config=overlay_config(config,'domain_distill_teacher_continue')
        # Fit both controls against the same original1440 train images and
        # fixed78.75% student. Old77.9167% basis is intentionally not reused.
        config.pop('teacher_alignment_sha256')
        config.pop('teacher_alignment_origin_checkpoint_sha256')
        config['adapt']['steps']=250
        overlay=json.loads(Path(__file__).with_name('domain_distillation.json').read_text(encoding='utf-8'))
        config.update(overlay['profiles'][profile])
        return config
    if profile in PINNED_TEACHER_PROFILES and profile != 'domain_distill_teacher_continue':
        config=overlay_config(config,'domain_distill_teacher_continue')
        overlay=json.loads(Path(__file__).with_name('domain_distillation.json').read_text(encoding='utf-8'))
        config.update(overlay['profiles'][profile])
        return config
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
    if 'frontend_training' in config:
        if config['frontend_training'] not in ('frozen','merger_fc2'):
            raise ValueError('Unknown compact frontend training contract')
        merger_learning_rate_multiplier(config)
        metadata['frontend_training']=config['frontend_training']
    if 'phase_dropout' in config:metadata['phase_dropout']=config['phase_dropout']
    result=dict(payload, metadata=metadata)
    if 'electronic_context_kernels' in config:
        result=expand_electronic_context(result,config['electronic_context_kernels'])
    if 'retrieval_head' in config:
        result=expand_retrieval_head(result,config['retrieval_head'])
    if 'router_initial_phase_offset_turns' in config:
        result=shift_router_phase_origin(result,config['router_initial_phase_offset_turns'])
    return result


def shift_router_phase_origin(payload, turns):
    """One-time warmstart only, preserving the physical optical implementation.

    Uniform phase over the illuminated router aperture cancels in ideal CCD
    intensity. It is NOT invariant relative to unmodulated/bypassed light;
    retain that explicit training-noise difference and re-evaluate initialization.
    Export ordinary raw sigmoid parameters; add no inference parametrization.
    """
    if isinstance(turns,bool) or not isinstance(turns,(int,float)) or not math.isfinite(turns) or not 0<turns<1:
        raise ValueError('Router origin offset must be a finite fraction of one turn')
    metadata=dict(payload['metadata']);previous=metadata.get('router_initial_phase_offset_turns')
    if previous is not None:
        if previous!=turns:raise ValueError('Refuse to stack different router initialization offsets')
        return payload
    state=dict(payload['state_dict'])
    for modality in ('vision','language'):
        name=modality+'.optics.router.raw_router_phase';raw=state.get(name)
        if not isinstance(raw,torch.Tensor) or raw.shape!=(224,224) or raw.dtype!=torch.float32 or not torch.isfinite(raw).all():
            raise ValueError('Expected finite FP32 224x224 router phase')
        # First sigmoid matches existing FP32 forward; double precision is only
        # used for the one-time wrap/inverse and never the optical propagation.
        phase_turns=torch.remainder(raw.detach().sigmoid().double()+turns,1.)
        state[name]=phase_turns.clamp(1e-7,1-1e-7).logit().float()
    metadata['router_initial_phase_offset_turns']=float(turns)
    return dict(payload,metadata=metadata,state_dict=state)


def expand_retrieval_head(payload, kind):
    """Replace final linear readout with one serial MLP, preserving its function.

    ReLU(t)-ReLU(-t)=t: initialize hidden as [W;-W],[b;-b] and
    output as [I,-I],0. No optical/frontend/electronic residual tensors change.
    Already-converted checkpoints are retained, never reinitialized on reload.
    """
    previous=payload['metadata'].get('retrieval_head','linear64')
    if kind not in ('linear64','relu128','linear256') or previous not in ('linear64','relu128','linear256'):
        raise ValueError('Unknown retrieval head contract')
    if previous==kind:return payload
    if previous!='linear64' or kind not in ('relu128','linear256'):
        raise ValueError('Only linear64 to relu128/linear256 readout expansion supported')
    state=dict(payload['state_dict'])
    w=state.pop('readout.projection.weight');b=state.pop('readout.projection.bias')
    if w.shape!=(64,384) or b.shape!=(64,) or not torch.isfinite(w).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid source linear readout')
    if kind=='linear256':
        # z -> [z,0] preserves cosine geometry before training; never duplicate
        # optical features or route around the existing optical computation.
        state['readout.projection.weight']=torch.cat((w,w.new_zeros(192,384)))
        state['readout.projection.bias']=torch.cat((b,b.new_zeros(192)))
        result=dict(payload,metadata=dict(payload['metadata'],retrieval_head=kind),state_dict=state)
        if 'auxiliary_training_head' in payload:
            auxiliary=dict(payload['auxiliary_training_head']);proxy=auxiliary['weight']
            if proxy.ndim!=2 or proxy.shape[1]!=64 or not torch.isfinite(proxy).all():
                raise ValueError('Invalid training category proxy for descriptor expansion')
            auxiliary['weight']=torch.cat((proxy,proxy.new_zeros(len(proxy),192)),dim=1)
            result['auxiliary_training_head']=auxiliary
        return result
    eye=torch.eye(64,dtype=w.dtype,device=w.device)
    state.update({'readout.projection.0.weight':torch.cat((w,-w)),
                  'readout.projection.0.bias':torch.cat((b,-b)),
                  'readout.projection.2.weight':torch.cat((eye,-eye),dim=1),
                  'readout.projection.2.bias':torch.zeros_like(b)})
    return dict(payload,metadata=dict(payload['metadata'],retrieval_head=kind),state_dict=state)


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
