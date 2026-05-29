import csv
from pathlib import Path
from typing import Dict, Iterable


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "test_loss",
    "test_acc",
    "lr",
]


def init_metrics_csv(path: str, append: bool = False) -> None:
    if append and Path(path).exists():
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def append_metrics_csv(path: str, row: Dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
