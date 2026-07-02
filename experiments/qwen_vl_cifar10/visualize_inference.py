from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .inference_profiling import BatchTiming


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_inference_figure(
    metrics: dict[str, Any],
    timings: Sequence[BatchTiming],
    feature_audit: dict[str, Any],
    output_path: Path,
) -> None:
    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)

    summary = metrics["timing_summary"]
    phase_fields = (
        ("dataset_fetch_sec", "dataset fetch"),
        ("prompt_build_sec", "prompt build"),
        ("image_preprocess_sec", "image processor"),
        ("tokenizer_sec", "tokenizer"),
        ("processor_framework_sec", "template/framework"),
        ("host_to_device_sec", "host to device"),
        ("model_generate_sec", "model.generate"),
        ("decode_postprocess_sec", "decode/postprocess"),
    )
    phase_labels = [label for _, label in phase_fields]
    phase_ms = [summary[field]["mean_ms_per_image"] for field, _ in phase_fields]
    axes[0, 0].barh(phase_labels, phase_ms, color="#4C78A8")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Mean milliseconds per image")
    axes[0, 0].set_title("Additive inference-time decomposition")

    indices = [row.batch_index for row in timings]
    for field, label, color in (
        ("model_generate_sec", "model only", "#E45756"),
        ("complete_inference_sec", "complete inference", "#F2CF5B"),
        ("end_to_end_sec", "end to end", "#54A24B"),
    ):
        values = [1000.0 * getattr(row, field) / row.image_count for row in timings]
        axes[0, 1].plot(indices, values, label=label, linewidth=1.2, color=color)
    axes[0, 1].set_xlabel("Measured batch index")
    axes[0, 1].set_ylabel("Milliseconds per image")
    axes[0, 1].set_title("Steady-state latency trace")
    axes[0, 1].legend()

    per_class = metrics.get("per_class_accuracy", {})
    class_names = list(per_class)
    class_scores = [float(per_class[name]) for name in class_names]
    axes[1, 0].bar(class_names, class_scores, color="#72B7B2")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Zero-shot CIFAR-10 accuracy by class")
    axes[1, 0].tick_params(axis="x", rotation=35)

    layer_labels, layer_widths, layer_shapes = _layer_widths(feature_audit)
    axes[1, 1].barh(layer_labels, layer_widths, color="#B279A2")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Last-dimension feature width")
    axes[1, 1].set_title("Observed intermediate activation shapes")
    if layer_widths:
        axes[1, 1].set_xlim(0, max(layer_widths) * 1.45)
    for index, (width, shape) in enumerate(zip(layer_widths, layer_shapes)):
        axes[1, 1].text(width, index, f"  {shape}", va="center", fontsize=8)

    model_ms = summary["model_generate_sec"]["mean_ms_per_image"]
    e2e_ms = summary["end_to_end_sec"]["mean_ms_per_image"]
    figure.suptitle(
        f"{str(metrics['model_id']).split('/')[-1]} | accuracy={metrics['accuracy']:.4f} | "
        f"model={model_ms:.2f} ms/image | end-to-end={e2e_ms:.2f} ms/image",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=200)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _layer_widths(
    feature_audit: dict[str, Any]
) -> tuple[list[str], list[int], list[str]]:
    labels: list[str] = []
    widths: list[int] = []
    shapes: list[str] = []
    modules = feature_audit.get("intermediate_modules", {})
    for name, record in modules.items():
        observations = record.get("observations", [])
        if not observations:
            continue
        observation = max(observations, key=lambda value: value.get("numel", 0))
        shape = observation.get("shape", [])
        if not shape:
            continue
        labels.append(name)
        widths.append(int(shape[-1]))
        shapes.append("x".join(str(value) for value in shape))
    pooled = feature_audit.get("pooled_feature")
    if pooled and pooled.get("shape"):
        labels.append("pooled visual feature")
        widths.append(int(pooled["shape"][-1]))
        shapes.append("x".join(str(value) for value in pooled["shape"]))
    return labels, widths, shapes


def _load_batch_timings(path: Path) -> list[BatchTiming]:
    integer_fields = {"batch_index", "image_count", "input_tokens", "generated_tokens"}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            values = {
                key: int(value) if key in integer_fields else float(value)
                for key, value in raw.items()
            }
            rows.append(BatchTiming(**values))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize a Qwen3-VL inference benchmark."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    with (run_dir / "inference_metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    feature_path = run_dir / "feature_shapes.json"
    feature_audit = (
        json.loads(feature_path.read_text(encoding="utf-8"))
        if feature_path.exists()
        else {}
    )
    timings = _load_batch_timings(run_dir / "batch_timings.csv")
    write_inference_figure(
        metrics, timings, feature_audit, run_dir / "inference_summary.png"
    )
    print(f"figure={run_dir / 'inference_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
