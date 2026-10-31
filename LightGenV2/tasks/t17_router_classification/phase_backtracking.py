"""Training-only bounded descent, always evaluated through canonical raw phases."""
import math
import torch


class CanonicalPhaseBacktracking(torch.optim.Optimizer):
    """Accept a physical-phase step only on a paired TRAIN objective decrease.

    The supplied closure must evaluate the same TRAIN views with a fixed head.
    No labels or validation metrics enter this optimizer. Rejection and failures
    restore the exact original raw tensors, including their finite precision.
    """
    def __init__(self, parameters, lr, max_trials=8, armijo=1e-4, min_decrease=1e-6):
        if lr <= 0 or max_trials < 1 or not 0 < armijo < 1 or min_decrease < 0:
            raise ValueError('Invalid bounded phase backtracking policy')
        super().__init__(parameters, dict(lr=lr))
        self.max_trials=max_trials;self.armijo=armijo;self.min_decrease=min_decrease
        self.last_step=None

    @torch.no_grad()
    def step(self, closure=None, baseline_loss=None):
        if closure is None or baseline_loss is None or not math.isfinite(float(baseline_loss)):
            raise ValueError('Explicit paired TRAIN closure and finite baseline required')
        active=[(p,group['lr']) for group in self.param_groups for p in group['params']]
        if not active or any(p.grad is None for p,_ in active):raise ValueError('Missing physical phase gradient')
        before=[p.detach().clone() for p,_ in active];angles=[];directions=[];slope=0.
        for p,lr in active:
            fraction=p.detach().double().sigmoid();angle=2*torch.pi*fraction
            g=p.grad.double()/(2*torch.pi*fraction*(1-fraction)).clamp_min(1e-20)
            if not torch.isfinite(g).all():raise RuntimeError('Nonfinite physical phase gradient')
            direction=-g/(g.abs()+1e-12)
            angles.append(angle);directions.append(lr*direction);slope+=float((g*direction).sum())*lr
        trials=[];accepted=False
        try:
            for trial in range(self.max_trials):
                scale=2.**(-trial)
                for (p,_),angle,direction in zip(active,angles,directions):
                    fraction=torch.remainder(angle+scale*direction,2*torch.pi)/(2*torch.pi)
                    p.copy_(torch.logit(fraction.clamp(1e-7,1-1e-7)).to(p.dtype))
                value=float(closure())
                required=max(self.min_decrease,-self.armijo*scale*slope)
                accepted=math.isfinite(value) and value <= float(baseline_loss)-required
                trials.append(dict(scale=scale,objective=value,required_decrease=required,accepted=accepted))
                if accepted:break
        finally:
            if not accepted:
                for (p,_),old in zip(active,before):p.copy_(old)
        self.last_step=dict(accepted=accepted,baseline_objective=float(baseline_loss),
            directional_derivative=slope,forward_trials=len(trials),trials=trials,
            accepted_scale=trials[-1]['scale'] if accepted else 0.,
            objective_after=trials[-1]['objective'] if accepted else float(baseline_loss),
            rejected_raw_restored_bitwise=not accepted)
        return self.last_step
