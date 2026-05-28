import torch


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean().item()
