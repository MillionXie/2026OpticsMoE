"""Adapt original visual mergers; FP32 optimizer masters are training-only."""
import torch
from torch import nn


def modules(qwen):
    visual=qwen.model.visual
    if len(visual.deepstack_merger_list)!=3:raise ValueError('Require the original three Qwen3-VL DeepStack mergers')
    return nn.ModuleDict({'merger':visual.merger,'deepstack_merger_list':visual.deepstack_merger_list})


def snapshot(qwen):return {n:v.detach().cpu().clone() for n,v in modules(qwen).state_dict().items()}


def restore(qwen,state):
    target=modules(qwen);expected=target.state_dict()
    if set(state)!=set(expected):raise ValueError('Native readout keys changed')
    if any(state[n].shape!=v.shape or not torch.isfinite(state[n]).all() for n,v in expected.items()):raise ValueError('Invalid native readout weights')
    target.load_state_dict(state,strict=True)


class ReadoutMasters(nn.Module):
    """Accumulate small Adam updates without changing BF16 forward arithmetic."""
    def __init__(self,qwen):
        super().__init__();actual=list(modules(qwen).parameters())
        # Borrow existing parameters without registering another deployed layer.
        object.__setattr__(self,'actual',actual)
        for p in actual:p.requires_grad_(True)
        self.master=nn.ParameterList([nn.Parameter(p.detach().float().clone()) for p in actual])

    def zero_native_grad(self):
        for p in self.actual:p.grad=None

    def collect(self):
        for p,master in zip(self.actual,self.master):master.grad=None if p.grad is None else p.grad.detach().float().clone()

    @torch.no_grad()
    def copy_to_native(self):
        for p,master in zip(self.actual,self.master):p.copy_(master.to(p.dtype))

    @torch.no_grad()
    def reset_from_native(self):
        for p,master in zip(self.actual,self.master):master.copy_(p.float());master.grad=None
        self.zero_native_grad()
