"""Training-only Adam in periodic phase coordinates; canonical raw state stays unchanged."""
import torch


class WrappedPhaseAdam(torch.optim.Adam):
    """Use angle gradients/moments, then export each step to the original sigmoid raw masks.

    Forward inference always uses the unmodified model and its original raw tensors.
    No unconstrained amplitude or extra inference parameter is introduced.
    """
    def __init__(self, parameters, lr, gradient_clip=5.,router_parameter=None,router_step_rule='adam'):
        super().__init__(parameters, lr=lr)
        self.gradient_clip = gradient_clip
        if router_step_rule not in ('adam','normalized_angle_sgd'):raise ValueError('Unknown router step rule')
        if router_step_rule=='normalized_angle_sgd' and (router_parameter is None or not any(p is router_parameter for g in self.param_groups for p in g['params'])):
            raise ValueError('Normalized router SGD requires an optimizer-owned router parameter')
        self.router_parameter=router_parameter;self.router_step_rule=router_step_rule
        self.last_router_step_bound=0.

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('WrappedPhaseAdam requires an explicit canonical forward/backward')
        active = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        raw_before = [p.detach().clone() for p in active]
        router_gradient=None;router_angle=None;router_zero=False
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
            if self.router_step_rule=='normalized_angle_sgd' and any(p is self.router_parameter for p in active):
                router=self.router_parameter;router_gradient=router.grad
                original=next(raw for p,raw in zip(active,raw_before) if p is router)
                router_angle=2*torch.pi*original.double().sigmoid()
                maximum=router_gradient.double().abs().max();router_zero=bool(maximum==0)
                router_lr=next(g['lr'] for g in self.param_groups if any(p is router for p in g['params']))
                self.last_router_step_bound=float(router_lr) if not router_zero else 0.
                if not router_zero:router_angle=router_angle-router_lr*router_gradient.double()/maximum
                # Same shared physical-gradient clipping as before; Adam sees
                # only E/G, and never creates a router moment state.
                router.grad=None
            super().step()
            if router_gradient is not None:self.router_parameter.grad=router_gradient
            for p,raw in zip(active,raw_before):
                if p is self.router_parameter and router_angle is not None:
                    if router_zero:
                        p.copy_(raw);continue
                    angle=router_angle
                else:angle=p.double()
                fraction = torch.remainder(angle, 2 * torch.pi) / (2 * torch.pi)
                # Only a numerical seam guard, at most ~6e-7 radians of phase error.
                p.copy_(torch.logit(fraction.clamp(1e-7, 1 - 1e-7)).to(p.dtype))
                if not torch.isfinite(p).all():
                    raise RuntimeError('Nonfinite canonical raw phase after wrapped update')
        except BaseException:
            for p, raw in zip(active, raw_before):
                p.copy_(raw)
            raise
        finally:
            if router_gradient is not None:self.router_parameter.grad=router_gradient
