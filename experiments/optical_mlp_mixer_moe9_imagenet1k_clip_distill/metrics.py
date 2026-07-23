from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassificationAccumulator:
    num_classes: int
    device: torch.device

    def __post_init__(self) -> None:
        self.samples = torch.zeros((), dtype=torch.float64, device=self.device)
        self.loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.correct1 = torch.zeros((), dtype=torch.float64, device=self.device)
        self.correct5 = torch.zeros((), dtype=torch.float64, device=self.device)
        self.class_total = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        self.class_correct = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        batch = labels.numel()
        top = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        predicted = top[:, 0]
        self.samples += batch
        self.loss_sum += loss.detach().double() * batch
        self.correct1 += predicted.eq(labels).sum().double()
        self.correct5 += top.eq(labels[:, None]).any(-1).sum().double()
        self.class_total += torch.bincount(labels, minlength=self.num_classes).double()
        correct_labels = labels[predicted.eq(labels)]
        self.class_correct += torch.bincount(
            correct_labels, minlength=self.num_classes
        ).double()

    def reduce(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for tensor in (
                self.samples,
                self.loss_sum,
                self.correct1,
                self.correct5,
                self.class_total,
                self.class_correct,
            ):
                torch.distributed.all_reduce(tensor)

    def compute(self) -> dict:
        samples = self.samples.clamp_min(1)
        per_class = self.class_correct / self.class_total.clamp_min(1)
        valid = self.class_total > 0
        return {
            "samples": int(self.samples.item()),
            "loss": float((self.loss_sum / samples).item()),
            "top1_accuracy": float((self.correct1 / samples).item()),
            "top5_accuracy": float((self.correct5 / samples).item()),
            "balanced_accuracy": float(per_class[valid].mean().item()) if valid.any() else 0.0,
            "per_class_accuracy": per_class.detach().cpu().tolist(),
            "per_class_samples": self.class_total.detach().cpu().long().tolist(),
        }


class ScalarAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[str, torch.Tensor] = {}
        self.weight = torch.zeros((), dtype=torch.float64, device=device)

    def update(self, values: dict[str, float | torch.Tensor], weight: int) -> None:
        self.weight += int(weight)
        for key, value in values.items():
            tensor = (
                value.detach().double()
                if isinstance(value, torch.Tensor)
                else torch.tensor(float(value), dtype=torch.float64, device=self.device)
            )
            self.values.setdefault(
                key, torch.zeros((), dtype=torch.float64, device=self.device)
            )
            self.values[key] += tensor * int(weight)

    def reduce(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.weight)
            for value in self.values.values():
                torch.distributed.all_reduce(value)

    def compute(self) -> dict[str, float]:
        divisor = self.weight.clamp_min(1)
        return {key: float((value / divisor).item()) for key, value in self.values.items()}


class RouterAccumulator:
    def __init__(self, num_blocks: int, num_experts: int, device: torch.device) -> None:
        self.selected = torch.zeros(num_blocks, num_experts, dtype=torch.float64, device=device)
        self.weight_sum = torch.zeros_like(self.selected)
        self.importance_sum = torch.zeros_like(self.selected)
        self.entropy_sum = torch.zeros(num_blocks, dtype=torch.float64, device=device)
        self.samples = torch.zeros(num_blocks, dtype=torch.float64, device=device)

    def update(self, statistics: list[dict[str, torch.Tensor]], batch_size: int) -> None:
        for index, values in enumerate(statistics):
            self.selected[index] += values["selected_count"].detach().double()
            # weights_mean includes zeros for unselected experts.
            self.weight_sum[index] += values["weights_mean"].detach().double() * batch_size
            self.importance_sum[index] += values["importance"].detach().double() * batch_size
            self.entropy_sum[index] += values["normalized_entropy"].detach().double() * batch_size
            self.samples[index] += batch_size

    def reduce(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for tensor in (
                self.selected,
                self.weight_sum,
                self.importance_sum,
                self.entropy_sum,
                self.samples,
            ):
                torch.distributed.all_reduce(tensor)

    def compute(self) -> dict:
        divisor = self.samples[:, None].clamp_min(1)
        return {
            "selection_rate": (self.selected / divisor).cpu().tolist(),
            "mean_sparse_routing_weight": (self.weight_sum / divisor).cpu().tolist(),
            "mean_dense_importance": (self.importance_sum / divisor).cpu().tolist(),
            "normalized_entropy": (
                self.entropy_sum / self.samples.clamp_min(1)
            ).cpu().tolist(),
            "samples": self.samples.long().cpu().tolist(),
        }
