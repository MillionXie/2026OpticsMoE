import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
