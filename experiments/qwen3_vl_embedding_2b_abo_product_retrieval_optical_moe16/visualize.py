from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .datasets import ABOBundle


def visualize_results(bundle: ABOBundle, settings: Any) -> None:
    gallery_by_item: dict[str, Path] = {}
    for sample in bundle.gallery.samples:
        gallery_by_item.setdefault(sample.item_id, sample.image_path)
    for system in ("teacher", "student"):
        csv_path = (
            settings.output_dir
            / "metrics"
            / f"{system}_retrieval_results.csv"
        )
        if not csv_path.is_file():
            continue
        rows = _read_csv(csv_path)
        _render_examples(
            rows[:12],
            gallery_by_item,
            settings.output_dir / "figures" / f"{system}_retrieval_examples.png",
        )
        if system == "student":
            failures = [row for row in rows if int(row["correct_top1"]) == 0]
            _render_examples(
                failures[:12],
                gallery_by_item,
                settings.output_dir / "figures" / "student_failure_cases.png",
            )
    _plot_training(settings)


def _render_examples(
    rows: list[dict[str, str]],
    gallery_by_item: dict[str, Path],
    path: Path,
) -> None:
    if not rows:
        return
    tile = 180
    label_height = 50
    columns = 4
    canvas = Image.new(
        "RGB",
        (columns * tile, len(rows) * (tile + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        top_items = row["top10_item_ids"].split("|")[:3]
        top_scores = row["top10_similarities"].split("|")[:3]
        paths = [Path(row["query_image_path"])] + [
            gallery_by_item[item_id] for item_id in top_items
        ]
        titles = [f"Query true={row['true_item_id']}"] + [
            f"Top-{rank} {item_id}\ncos={score}"
            for rank, (item_id, score) in enumerate(
                zip(top_items, top_scores), start=1
            )
        ]
        for column, (image_path, title) in enumerate(zip(paths, titles)):
            image = _fit_image(image_path, tile, tile)
            x = column * tile
            y = row_index * (tile + label_height)
            canvas.paste(image, (x, y))
            draw.multiline_text(
                (x + 4, y + tile + 3),
                title[:90],
                fill=(0, 0, 0),
                font=font,
                spacing=2,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    rgb.thumbnail((width, height), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (width, height), (240, 240, 240))
    output.paste(rgb, ((width - rgb.width) // 2, (height - rgb.height) // 2))
    return output


def _plot_training(settings: Any) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plotted = False
    for axis, stage in zip(axes, ("stage1", "stage2")):
        path = (
            settings.output_dir
            / "metrics"
            / f"{stage}_training_history.csv"
        )
        if not path.is_file():
            axis.set_visible(False)
            continue
        rows = _read_csv(path)
        epochs = [int(row["epoch"]) for row in rows]
        for key, label in (
            ("total_loss", "total"),
            ("kd_loss", "KD"),
            ("supcon_loss", "SupCon"),
            ("identity_loss", "ID"),
        ):
            values = [float(row[key]) for row in rows]
            if any(value != 0 for value in values):
                axis.plot(epochs, values, label=label)
        axis.set_title(stage)
        axis.set_xlabel("epoch")
        axis.set_ylabel("training loss")
        axis.grid(alpha=0.25)
        axis.legend()
        plotted = True
    if plotted:
        figure.tight_layout()
        path = settings.output_dir / "figures" / "training_curves.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

