from typing import Dict, Optional

import torch


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        batch = targets.numel()
        total_loss += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch
    return {
        "loss": total_loss / max(1, seen),
        "acc": correct / max(1, seen),
        "samples": seen,
    }


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets_all = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        targets_all.append(targets.cpu())
    return torch.cat(preds), torch.cat(targets_all)

