"""Opt-in bounded LSP propagation input; no old checkpoint behavior changes.

Shared numerical contract with ABO/T12: preserve complex phase and zero support,
bound amplitude before propagation, and never undo the bound by multiplying a
measured intensity by the legacy fixed-scale factor. Camera response calibration
and detector readout are separate contracts; this helper does not calibrate them.
"""
import torch


def bounded_field(field, scale=0.5, quantize=False):
    if scale <= 0 or not torch.isfinite(torch.tensor(scale)):
        raise ValueError('Positive finite amplitude scale required')
    magnitude = field.abs()
    amplitude = torch.tanh(magnitude / scale)
    if quantize:
        rounded = torch.round(255 * amplitude) / 255
        # Forward is exactly the BMP value; TRAIN gradients use the smooth map.
        amplitude = amplitude + (rounded - amplitude).detach()
    return field / magnitude.clamp_min(1e-8) * amplitude


def install_bounded_inputs(core, *, quantize=False):
    """Install once on a newly built T02 core, not a running hardware session."""
    optical = core.optical_branch.core
    propagators = (optical.router.propagator, optical.propagator)
    if any(getattr(p, '_lsp_bounded_input', False) for p in propagators):
        raise RuntimeError('Bounded input already installed')
    def encode(_module, args):
        if len(args) != 1:
            raise RuntimeError('Unexpected propagator signature')
        return (bounded_field(args[0], quantize=quantize),)
    for propagator in propagators:
        propagator.register_forward_pre_hook(encode)
        propagator._lsp_bounded_input = True
