"""Pure helpers for the staged fixed-bank comparison."""
import math
import random
import torch


def common_anchor(seed, size=224, dc_power=0.25):
    """Uniform physical phase about pi; expected mask DC power is sinc(a)^2.

    This initialization statistic is separate from the optical zero-order noise.
    Return sigmoid logits so the inherited PhaseLayer remains unchanged.
    """
    if not 0 < dc_power < 1:
        raise ValueError('dc_power must be in (0,1)')
    lo, hi = 0.0, math.pi
    for _ in range(60):
        mid = (lo + hi) / 2
        if (math.sin(mid) / mid) ** 2 > dc_power:
            lo = mid
        else:
            hi = mid
    noise = torch.rand(2, 4, size, size, generator=torch.Generator().manual_seed(seed))
    phase = math.pi + (2 * noise - 1) * ((lo + hi) / 2)
    return torch.logit(phase / (2 * math.pi))


def split_train_validation(samples, seed, per_class):
    training, validation = [], []
    rng = random.Random(seed)
    for label in sorted({s.sku_index for s in samples}):
        rows = sorted((s for s in samples if s.sku_index == label), key=lambda s: s.sample_id)
        if len(rows) <= per_class + 3:
            raise ValueError('Insufficient class examples after validation holdout')
        rng.shuffle(rows)
        validation.extend(rows[:per_class])
        training.extend(rows[per_class:])
    return training, validation


def stage_at(epoch, stages):
    start = 1
    for stage in stages:
        end = start + int(stage['epochs'])
        if start <= epoch < end:
            return stage, epoch - start
        start = end
    raise ValueError(f'No stage for epoch {epoch}')


def active_groups(stage_name):
    experts = {'expert', 'generator_context', 'generator_decoder'}
    if stage_name == 'experts':
        return experts
    if stage_name == 'optics':
        return experts | {'router', 'global', 'generator_global_decoder'}
    if stage_name == 'joint':
        return experts | {'router', 'global', 'generator_global_decoder', 'electronic', 'readout', 'fusion'}
    raise ValueError(stage_name)


def physical_phase(raw):
    return 2 * math.pi * raw.float().sigmoid()


@torch.no_grad()
def zero_optical_phases(surrogate):
    """Router deliberately uses a distinct raw_router_phase parameter name."""
    audit={}
    for name,module in surrogate.named_modules():
        for attribute in ('raw_phase','raw_router_phase'):
            parameter=getattr(module,attribute,None)
            if isinstance(parameter,torch.nn.Parameter):
                parameter.zero_()
                phase=module.phase()
                if float((phase-math.pi).abs().max())>1e-6:
                    raise RuntimeError('Optical zero logits must map to physical pi')
                audit[f'{name}.{attribute}']={'shape':list(parameter.shape),'raw_max_abs':float(parameter.abs().max()),
                    'physical_min_rad':float(phase.min()),'physical_max_rad':float(phase.max())}
    return audit


@torch.no_grad()
def phase_summary(raw, initial):
    phase = physical_phase(raw)
    diff = phase - physical_phase(initial)
    diff = torch.atan2(diff.sin(), diff.cos())
    return {'rms_change_rad': float(diff.square().mean().sqrt()),
            'per_expert_rms_change_rad': diff.square().mean((-2, -1)).sqrt().tolist(),
            'max_change_rad': float(diff.abs().max()),
            'dc_power': (phase.cos().mean((-2,-1)).square() + phase.sin().mean((-2,-1)).square()).tolist(),
            'sigmoid_saturated_fraction': float((raw.abs() > 4).float().mean())}
