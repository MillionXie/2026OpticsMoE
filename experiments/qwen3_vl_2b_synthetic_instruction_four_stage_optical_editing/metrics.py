from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .scenes import TASKS


def compose_prediction(
    palette_logits: torch.Tensor,
    edit_logits: torch.Tensor,
    source_classes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generated = palette_logits.argmax(dim=1)
    predicted_edit = torch.sigmoid(edit_logits).ge(0.5)
    output = torch.where(predicted_edit, generated, source_classes.long())
    return output, generated, predicted_edit


class MetricAccumulator:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> list[dict[str, Any]]:
        prediction, generated, predicted_edit = compose_prediction(
            outputs["palette_logits"],
            outputs["edit_logits"],
            batch["source_classes"],
        )
        target = batch["target_classes"].long()
        edit_target = batch["edit_mask"].bool()
        preserve = batch["preserve_mask"].bool()
        task_prediction = outputs["task_logits"].argmax(dim=-1)
        records: list[dict[str, Any]] = []
        for index, task_name in enumerate(batch["task"]):
            matches = prediction[index].eq(target[index])
            intersection = (predicted_edit[index] & edit_target[index]).sum().item()
            union = (predicted_edit[index] | edit_target[index]).sum().item()
            edit_iou = intersection / union if union else 1.0
            changed_count = int(edit_target[index].sum().item())
            changed_accuracy = (
                matches[edit_target[index]].float().mean().item() if changed_count else 1.0
            )
            preserve_count = int(preserve[index].sum().item())
            preserve_accuracy = (
                matches[preserve[index]].float().mean().item() if preserve_count else 1.0
            )
            pixel_accuracy = matches.float().mean().item()
            black_prediction = prediction[index].eq(7)
            black_target = target[index].eq(7)
            true_positive = (black_prediction & black_target).sum().item()
            edge_precision = true_positive / max(1, black_prediction.sum().item())
            edge_recall = true_positive / max(1, black_target.sum().item())
            edge_f1 = (
                2 * edge_precision * edge_recall / (edge_precision + edge_recall)
                if edge_precision + edge_recall
                else 0.0
            )
            task_correct = float(task_prediction[index].item() == batch["task_index"][index].item())
            success = float(
                pixel_accuracy >= 0.99
                and edit_iou >= 0.90
                and preserve_accuracy >= 0.995
            )
            values = {
                "pixel_accuracy": pixel_accuracy,
                "changed_pixel_accuracy": changed_accuracy,
                "preserved_pixel_accuracy": preserve_accuracy,
                "edit_mask_iou": edit_iou,
                "task_accuracy": task_correct,
                "edge_f1": edge_f1 if task_name == "edge" else 0.0,
                "sample_success": success,
            }
            for group in ("overall", task_name):
                self.rows[group]["count"] += 1.0
                for key, value in values.items():
                    self.rows[group][key] += value
            records.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "task": task_name,
                    "instruction": batch["instruction"][index],
                    **values,
                }
            )
        return records

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        ordered = ("overall", *TASKS)
        for group in ordered:
            values = self.rows.get(group)
            if not values:
                continue
            count = values["count"]
            result[group] = {
                "samples": int(count),
                **{
                    key: value / count
                    for key, value in values.items()
                    if key != "count"
                },
            }
        return result


__all__ = ["MetricAccumulator", "compose_prediction"]
