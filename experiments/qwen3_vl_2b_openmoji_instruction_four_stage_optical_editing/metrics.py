from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .scenes import TASKS


def compose_prediction(
    category_logits: torch.Tensor,
    edit_logits: torch.Tensor,
    source_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generated = category_logits.argmax(dim=1)
    edit = torch.sigmoid(edit_logits).ge(0.5)
    composed = torch.where(edit, generated, source_grid.long())
    return composed, generated, edit


class MetricAccumulator:
    def __init__(self) -> None:
        self.groups: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def update(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
        prediction, _, predicted_edit = compose_prediction(
            outputs["category_logits"], outputs["edit_logits"], batch["source_grid"]
        )
        target = batch["target_grid"].long()
        true_edit = batch["edit_grid"].bool()
        preserve = batch["preserve_grid"].bool()
        task_prediction = outputs["task_logits"].argmax(-1)
        records = []
        for index, task in enumerate(batch["task"]):
            correct = prediction[index].eq(target[index])
            foreground = target[index].gt(0)
            changed_accuracy = correct[true_edit[index]].float().mean().item()
            foreground_accuracy = (
                correct[foreground].float().mean().item() if foreground.any() else 1.0
            )
            preserved_accuracy = correct[preserve[index]].float().mean().item()
            intersection = (predicted_edit[index] & true_edit[index]).sum().item()
            union = (predicted_edit[index] | true_edit[index]).sum().item()
            edit_iou = intersection / union if union else 1.0
            predicted_objects = {
                (int(prediction[index, row, col]), int(row), int(col))
                for row, col in zip(*prediction[index].gt(0).nonzero(as_tuple=True))
            }
            target_objects = {
                (int(target[index, row, col]), int(row), int(col))
                for row, col in zip(*target[index].gt(0).nonzero(as_tuple=True))
            }
            matched = len(predicted_objects & target_objects)
            object_precision = matched / max(1, len(predicted_objects))
            object_recall = matched / max(1, len(target_objects))
            object_f1 = (
                2.0 * object_precision * object_recall / (object_precision + object_recall)
                if object_precision + object_recall
                else 0.0
            )
            scene_exact = float(torch.equal(prediction[index], target[index]))
            values = {
                "cell_accuracy": correct.float().mean().item(),
                "changed_cell_accuracy": changed_accuracy,
                "foreground_category_accuracy": foreground_accuracy,
                "preserved_cell_accuracy": preserved_accuracy,
                "edit_grid_iou": edit_iou,
                "object_f1": object_f1,
                "scene_exact_match": scene_exact,
                "task_accuracy": float(
                    task_prediction[index].item() == batch["task_index"][index].item()
                ),
            }
            for group in ("overall", task):
                self.groups[group]["count"] += 1.0
                for name, value in values.items():
                    self.groups[group][name] += value
            records.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "task": task,
                    "instruction": batch["instruction"][index],
                    **values,
                }
            )
        return records, prediction, predicted_edit

    def compute(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for group in ("overall", *TASKS):
            values = self.groups.get(group)
            if not values:
                continue
            count = values["count"]
            result[group] = {
                "samples": int(count),
                **{name: value / count for name, value in values.items() if name != "count"},
            }
        return result


__all__ = ["MetricAccumulator", "compose_prediction"]
