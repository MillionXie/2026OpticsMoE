"""Training-only illumination and registration controls for the optical stages."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch

if __package__:
    from .evaluate_phase_robustness import modulation_for_condition
    from .spatial_locality import STAGES
    from ..model import translated
else:
    from evaluate_phase_robustness import modulation_for_condition
    from spatial_locality import STAGES
    from model import translated


def illumination_coverage_loss(captured, valid, minimum_fractions):
    """Discourage concentrated incident energy within the existing pixel mapping."""
    penalties = []
    for branch in STAGES:
        minimum = minimum_fractions[branch]
        if not 0 < minimum <= 1:
            raise ValueError('Minimum effective illumination area must lie in (0,1]')
        intensity = captured[branch + '_incident'].flatten(2).float().square()
        effective_area = intensity.sum(-1).square() / intensity.square().sum(-1).clamp_min(1e-20)
        fraction = effective_area / (111 * 111)
        penalties.append(((minimum - fraction) / minimum).clamp_min(0)[valid].mean())
    return torch.stack(penalties).mean()


def registration_condition(maximum_pixels):
    """Sample clean, phase-shift or CCD-shift training with equal probabilities."""
    if maximum_pixels < 0 or int(maximum_pixels) != maximum_pixels:
        raise ValueError('Registration shift must be a nonnegative integer')
    if maximum_pixels == 0:
        return {'kind': 'clean'}
    kind = ('clean', 'phase_shift', 'ccd_shift')[int(torch.randint(3, ()))]
    if kind == 'clean':
        return {'kind': kind}
    distance = int(torch.randint(1, maximum_pixels + 1, ()))
    dy, dx = ((distance, 0), (-distance, 0), (0, distance), (0, -distance))[
        int(torch.randint(4, ()))]
    return {'kind': kind, 'dy': dy, 'dx': dx}


@contextmanager
def registered_optics(model, condition):
    """Temporarily perturb registration while preserving deterministic model mode."""
    if condition['kind'] not in ('clean', 'phase_shift', 'ccd_shift'):
        raise ValueError('Unsupported training registration condition')
    with ExitStack() as stack:
        for index, attribute in enumerate(STAGES.values()):
            stage = getattr(model, attribute)
            if condition['kind'] == 'phase_shift':
                stack.enter_context(patch.object(stage, 'modulation',
                    modulation_for_condition(stage, condition, index)))
            elif condition['kind'] == 'ccd_shift':
                def shift(module, arguments, output):
                    return translated(output, condition['dy'], condition['dx'])
                handle = stage.propagation.register_forward_hook(shift)
                stack.callback(handle.remove)
        yield
