from typing import Dict, Sequence

import torch


def _next_batch(iterators, loaders, task_name):
    try:
        return next(iterators[task_name])
    except StopIteration:
        iterators[task_name] = iter(loaders[task_name])
        return next(iterators[task_name])


def train_multitask_one_epoch(
    model,
    train_loaders: Dict,
    optimizer,
    device: torch.device,
    criterion,
    task_names: Sequence[str],
    loss_reduction: str = "mean",
    batches_per_update: int = 1,
    balanced_sampling: bool = True,
) -> Dict:
    """Train shared optics using one task-specific prompt per dataset batch."""

    if loss_reduction not in {"mean", "sum"}:
        raise ValueError("loss_reduction must be mean or sum.")
    model.train()
    task_names = list(task_names)
    steps = (
        max(len(train_loaders[name]) for name in task_names)
        if balanced_sampling
        else min(len(train_loaders[name]) for name in task_names)
    )
    iterators = {name: iter(train_loaders[name]) for name in task_names}
    task_loss_sums = {name: 0.0 for name in task_names}
    task_correct = {name: 0 for name in task_names}
    task_seen = {name: 0 for name in task_names}
    total_loss_sum = 0.0

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        update_losses = []
        for task_name in task_names:
            for _batch_index in range(int(batches_per_update)):
                batch = _next_batch(iterators, train_loaders, task_name)
                images, targets = batch[:2]
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images, task_name=task_name)
                loss = criterion(logits, targets)
                update_losses.append(loss)

                batch_size = targets.numel()
                task_loss_sums[task_name] += float(loss.item()) * batch_size
                task_correct[task_name] += int(
                    (logits.argmax(dim=1) == targets).sum().item()
                )
                task_seen[task_name] += batch_size

        stacked = torch.stack(update_losses)
        total_loss = stacked.mean() if loss_reduction == "mean" else stacked.sum()
        total_loss.backward()
        optimizer.step()
        total_loss_sum += float(total_loss.item())

    result = {
        "total_loss": total_loss_sum / max(steps, 1),
        "steps": steps,
    }
    for task_name in task_names:
        result[f"{task_name}_loss"] = task_loss_sums[task_name] / max(
            task_seen[task_name], 1
        )
        result[f"{task_name}_acc"] = task_correct[task_name] / max(
            task_seen[task_name], 1
        )
    return result


@torch.no_grad()
def evaluate_task(
    model,
    loader,
    device: torch.device,
    criterion,
    prompt_task: str,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    predictions = []
    targets_all = []
    for batch in loader:
        images, targets = batch[:2]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, task_name=prompt_task)
        loss = criterion(logits, targets)
        predicted = logits.argmax(dim=1)
        batch_size = targets.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predicted == targets).sum().item())
        total_seen += batch_size
        predictions.append(predicted.cpu())
        targets_all.append(targets.cpu())
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "predictions": (
            torch.cat(predictions)
            if predictions
            else torch.empty(0, dtype=torch.long)
        ),
        "targets": (
            torch.cat(targets_all)
            if targets_all
            else torch.empty(0, dtype=torch.long)
        ),
    }


def task_switching_evaluation(
    model,
    test_loaders: Dict,
    device: torch.device,
    criterion,
    task_names: Sequence[str],
):
    rows = []
    for eval_dataset in task_names:
        for prompt_task in task_names:
            result = evaluate_task(
                model,
                test_loaders[eval_dataset],
                device,
                criterion,
                prompt_task=prompt_task,
            )
            rows.append(
                {
                    "eval_dataset": eval_dataset,
                    "prompt_task": prompt_task,
                    "loss": result["loss"],
                    "accuracy": result["accuracy"],
                }
            )
    return rows
