from __future__ import annotations

import argparse
import ast
import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _select_examples(
    predictions: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    successes = sorted(
        (row for row in predictions if int(row["first_positive_rank"]) == 1),
        key=lambda row: int(row["title_label"]),
    )
    quantiles = (0.15, 0.50, 0.85)
    representative = [
        successes[round((len(successes) - 1) * quantile)] for quantile in quantiles
    ]
    failures = sorted(
        (row for row in predictions if int(row["first_positive_rank"]) > 1),
        key=lambda row: (-int(row["first_positive_rank"]), int(row["title_label"])),
    )[:3]
    return representative, failures


def _draw(
    rows: list[dict[str, str]],
    *,
    sample_by_id: dict[str, dict[str, str]],
    dataset_root: Path,
    output: Path,
    heading: str,
    note: str,
) -> None:
    count = len(rows)
    fig = plt.figure(figsize=(17, 3.15 * count + 1.15), facecolor="white")
    grid = fig.add_gridspec(
        count,
        6,
        width_ratios=[2.8, 1, 1, 1, 1, 1],
        left=0.025,
        right=0.985,
        top=0.88,
        bottom=0.055,
        wspace=0.12,
        hspace=0.42,
    )
    fig.suptitle(heading, fontsize=19, fontweight="bold", y=0.965)
    fig.text(0.5, 0.915, note, ha="center", va="center", fontsize=10, color="#444444")

    for row_index, prediction in enumerate(rows):
        query_product = prediction["product_id"]
        sample_ids = ast.literal_eval(prediction["top10_sample_ids"])[:5]
        scores = ast.literal_eval(prediction["top10_scores"])[:5]
        first_rank = int(prediction["first_positive_rank"])

        text_ax = fig.add_subplot(grid[row_index, 0])
        text_ax.axis("off")
        text_ax.text(
            0,
            0.95,
            f"Query #{prediction['title_label']}\nTarget SKU: {query_product}",
            transform=text_ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        text_ax.text(
            0,
            0.67,
            textwrap.fill(prediction["title"], width=39),
            transform=text_ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        text_ax.text(
            0,
            0.06,
            f"First correct rank: {first_rank}",
            transform=text_ax.transAxes,
            ha="left",
            va="bottom",
            color="#20733b" if first_rank == 1 else "#b33a32",
            fontsize=11,
            fontweight="bold",
        )

        for rank, (sample_id, score) in enumerate(zip(sample_ids, scores), start=1):
            sample = sample_by_id[sample_id]
            correct = sample["product_id"] == query_product
            image_path = dataset_root / sample["image_path"]
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            ax = fig.add_subplot(grid[row_index, rank])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            color = "#2d8a4b" if correct else "#c64a3e"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(4)
                spine.set_edgecolor(color)
            mark = "CORRECT" if correct else "WRONG"
            ax.set_title(f"Top-{rank}  {mark}\nscore={score:.3f}", fontsize=10, color=color)
            ax.set_xlabel(sample["product_id"], fontsize=9, color=color)

    fig.savefig(output, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    predictions = _read_rows(args.run_dir / "text_to_image_predictions.csv")
    test_rows = _read_rows(args.dataset_root / "test.csv")
    sample_by_id = {row["sample_id"]: row for row in test_rows}
    successes, failures = _select_examples(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _draw(
        successes,
        sample_by_id=sample_by_id,
        dataset_root=args.dataset_root,
        output=args.output_dir / "retrieval_examples_success.png",
        heading="ABO easy100 Text-to-Image Retrieval — Representative Top-1 Successes",
        note=(
            "Final 15 cm optical Router Top-2 model; three deterministic label-quantile "
            "examples among all Top-1 successes. Green = same SKU, red = different SKU."
        ),
    )
    _draw(
        failures,
        sample_by_id=sample_by_id,
        dataset_root=args.dataset_root,
        output=args.output_dir / "retrieval_examples_failure.png",
        heading="ABO easy100 Text-to-Image Retrieval — Three Hardest Queries",
        note=(
            "Final 15 cm optical Router Top-2 model; sorted by first correct rank. "
            "These are failure analysis examples, not manually selected favorable cases."
        ),
    )


if __name__ == "__main__":
    main()
