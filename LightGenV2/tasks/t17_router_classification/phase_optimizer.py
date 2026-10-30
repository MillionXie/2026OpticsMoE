"""Training-only Adam in periodic phase coordinates; canonical raw state stays unchanged."""
import torch


class WrappedPhaseAdam(torch.optim.Adam):
    """Use angle gradients/moments, then export each step to the original sigmoid raw masks.

    Forward inference always uses the unmodified model and its original raw tensors.
    No unconstrained amplitude or extra inference parameter is introduced.
    """
    def __init__(self, parameters, lr, gradient_clip=5.):
        super().__init__(parameters, lr=lr)
        self.gradient_clip = gradient_clip

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('WrappedPhaseAdam requires an explicit canonical forward/backward')
        active = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        raw_before = [p.detach().clone() for p in active]
        try:
            for p in active:
                fraction = p.detach().double().sigmoid()
                jacobian = 2 * torch.pi * fraction * (1 - fraction)
                # The model produced dL/draw; convert by the exact chain rule.
                gradient = p.grad.double() / jacobian.clamp_min(1e-20)
                if not torch.isfinite(gradient).all():
                    raise RuntimeError('Nonfinite physical phase gradient')
                p.grad.copy_(gradient.to(p.dtype))
                p.copy_((2 * torch.pi * fraction).to(p.dtype))
            torch.nn.utils.clip_grad_norm_(active, self.gradient_clip, error_if_nonfinite=True)
            super().step()
            for p in active:
                fraction = torch.remainder(p.double(), 2 * torch.pi) / (2 * torch.pi)
                # Only a numerical seam guard, at most ~6e-7 radians of phase error.
                p.copy_(torch.logit(fraction.clamp(1e-7, 1 - 1e-7)).to(p.dtype))
                if not torch.isfinite(p).all():
                    raise RuntimeError('Nonfinite canonical raw phase after wrapped update')
        except BaseException:
            for p, raw in zip(active, raw_before):
                p.copy_(raw)
            raise
