"""Optional exact normalization backward; original fusion forward is retained.

Motivation: Xu et al., Understanding and Improving Layer Normalization,
NeurIPS 2019, arXiv:1911.07013 (not a reproduction of AdaNorm).
No new inference operation, affine parameter, branch, or optical change.
"""
from types import MethodType
import torch


def differentiable_fusion(electronic, optical, alpha, padding_mask, epsilon):
    """Original convex/RMS forward equation, with differentiable RMS statistics.

    Clamp squared RMS before sqrt for finite derivatives at zero/padded inputs.
    Above the epsilon floor this is exactly the derivative of the old forward.
    """
    valid = ~padding_mask
    e, o = electronic.float(), optical.float()
    mask = valid.unsqueeze(-1).to(e.dtype)
    count = (valid.sum(1, keepdim=True).to(e.dtype) * e.shape[-1]).clamp_min(1)

    def rms(value):
        square_mean = (value.square() * mask).sum((1, 2))[:, None, None] / count.unsqueeze(-1)
        return square_mean.clamp_min(float(epsilon)**2).sqrt()

    re, ro = rms(e), rms(o)
    mixture = (1-alpha) * e/re + alpha * o/ro
    fused = re * mixture / rms(mixture)
    fused = torch.where(alpha.detach() == 0, e, fused)
    return fused.to(electronic.dtype).masked_fill(padding_mask.unsqueeze(-1), 0.)


def enable_exact_fusion_backward(hybrid):
    """Training-only override; deployment still executes the original method.

    The original no-grad forward supplies unchanged values and diagnostics.
    A zero-valued term substitutes only its backward with full RMS derivatives.
    """
    if getattr(hybrid, '_exact_fusion_backward_enabled', False):
        raise ValueError('Exact fusion backward is already configured')
    original = hybrid._fuse.__func__

    def fuse(self, electronic, optical, alpha, padding_mask, stage):
        if (not self.training or not torch.is_grad_enabled()
                or self.fusion_mode != 'scale_matched_convex'
                or self.fusion_ablation_mode != 'none'):
            return original(self, electronic, optical, alpha, padding_mask, stage)
        with torch.no_grad():
            reference = original(self, electronic, optical, alpha, padding_mask, stage)
        full = differentiable_fusion(electronic, optical, alpha, padding_mask, self.fusion_rms_epsilon)
        return reference + (full - full.detach())

    hybrid._fuse = MethodType(fuse, hybrid)
    hybrid._exact_fusion_backward_enabled = True
