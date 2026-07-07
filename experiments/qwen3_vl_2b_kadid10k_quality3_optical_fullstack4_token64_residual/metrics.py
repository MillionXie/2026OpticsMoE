from __future__ import annotations

from typing import Any, Sequence
import torch


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, class_names: Sequence[str]) -> dict[str, Any]:
    predictions = logits.argmax(dim=1); top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    matrix = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    for truth, pred in zip(labels.cpu().tolist(), predictions.cpu().tolist()): matrix[truth, pred] += 1
    per_accuracy={}; per_precision={}; per_recall={}; per_f1={}
    for index,name in enumerate(class_names):
        tp=int(matrix[index,index]); support=int(matrix[index].sum()); predicted=int(matrix[:,index].sum())
        precision=tp/predicted if predicted else 0.0; recall=tp/support if support else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_accuracy[name]=recall; per_precision[name]=precision; per_recall[name]=recall; per_f1[name]=f1
    total=max(1,len(labels))
    return {
        "top1_accuracy": float((predictions.cpu()==labels.cpu()).sum())/total,
        "top5_accuracy": float(sum(int(int(y) in row) for y,row in zip(labels.cpu().tolist(),top5.cpu().tolist())))/total,
        "macro_f1": sum(per_f1.values())/len(class_names),
        "balanced_accuracy": sum(per_recall.values())/len(class_names),
        "per_class_accuracy": per_accuracy, "per_class_precision": per_precision,
        "per_class_recall": per_recall, "per_class_f1": per_f1,
        "confusion_matrix": matrix.tolist(), "samples": len(labels),
    }


def concatenate_metrics(logit_chunks: list[torch.Tensor], label_chunks: list[torch.Tensor], class_names: Sequence[str]) -> dict[str, Any]:
    return metrics_from_logits(torch.cat(logit_chunks), torch.cat(label_chunks), class_names)

