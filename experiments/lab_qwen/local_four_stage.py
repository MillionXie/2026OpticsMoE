"""Run the four measured-CCD Qwen adaptation stages on the laboratory PC.

The wrapper owns all long paths and always uses a development split for
checkpoint selection.  The sealed Caltech101 test split is evaluated once,
after the selected checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paths(root: Path, stage: str) -> tuple[Path, Path, Path]:
    project = root / (
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_"
        "optical_retrieval_10cm_ccd_noise1"
    )
    config = project / "configs/release/stage_hardware_canonical_ccd.yaml"
    session = root / "experiments/lab_qwen/four"
    index = STAGES.index(stage)
    checkpoint = (
        root / "experiments/lab_qwen/model/ema.pt"
        if index == 0
        else session / "checkpoints" / f"after_{STAGES[index - 1]}.pt"
    )
    return config, session, checkpoint


def _reconstruct_amplitude(stage_dir: Path, settings: object) -> dict[str, object]:
    from experiments.hardware_sdk.workflows.reconstruct_slm import (
        reconstruct_directory,
    )

    return reconstruct_directory(
        stage_dir / "compact_amplitude",
        stage_dir / "amplitude_to_play",
        slm_size_wh=(
            int(settings.hardware_amplitude_slm_width),
            int(settings.hardware_amplitude_slm_height),
        ),
        scale_factor=None,
        center_xy=(
            float(settings.hardware_amplitude_slm_center_x),
            float(settings.hardware_amplitude_slm_center_y),
        ),
        logical_pixel_pitch_um=float(settings.language_optical_pixel_pitch_um),
        slm_pixel_pitch_um=float(settings.hardware_amplitude_slm_pixel_pitch_um),
    )


def run(
    *,
    stage: str,
    epochs: int,
    development_per_class: int,
    pk_classes: int,
    pk_images_per_class: int,
    inference_batch_size: int,
    early_stopping_patience: int,
    finetune_only: bool,
) -> dict[str, object]:
    # Heavy Qwen/Torch imports occur only for this explicit local-training command.
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1.modeling import (
        build_hybrid_student,
        load_backbone,
    )
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_ccd_noise1.settings import (
        load_settings,
    )
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
        hardware_bridge as bridge,
    )
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
        seed_everything,
    )

    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    root = _repo_root()
    config, session, checkpoint = _paths(root, stage)
    if not config.is_file():
        raise FileNotFoundError(f"Local four-stage config is missing: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Previous-stage checkpoint is missing: {checkpoint}\n"
            "If you are switching from a server run midway, download that exact "
            "after_<previous-stage>.pt file into experiments/lab_qwen/four/checkpoints/."
        )

    bridge.build_hybrid_student = build_hybrid_student
    bridge.load_backbone = load_backbone
    bridge.load_settings = load_settings
    settings = load_settings(config)
    seed_everything(settings.random_seed)
    bridge.finetune_stage(
        settings,
        checkpoint,
        session,
        stage,
        epochs,
        upstream_source="measured",
        selection_policy="development",
        development_per_class=development_per_class,
        pk_classes=pk_classes,
        pk_images_per_class=pk_images_per_class,
        inference_batch_size=inference_batch_size,
        early_stopping_patience=early_stopping_patience,
    )
    selected = session / "checkpoints" / f"after_{stage}.pt"
    result: dict[str, object] = {
        "status": "fine_tuned",
        "stage": stage,
        "checkpoint": str(selected),
        "metrics": str(
            session
            / f"{STAGES.index(stage) + 1:02d}_{stage}"
            / "finetune_metrics.json"
        ),
        "selection": "development_top1_then_ce; sealed_test_once_after_selection",
    }
    stage_index = STAGES.index(stage)
    if not finetune_only and stage_index + 1 < len(STAGES):
        next_stage = STAGES[stage_index + 1]
        bridge.export_stage(
            settings,
            selected,
            session,
            next_stage,
            upstream_source="measured",
            inference_batch_size=inference_batch_size,
        )
        next_dir = session / f"{stage_index + 2:02d}_{next_stage}"
        reconstruction = _reconstruct_amplitude(next_dir, settings)
        result.update(
            {
                "next_stage": next_stage,
                "next_stage_dir": str(next_dir),
                "next_amplitude_bmps": int(reconstruction["files"]),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--development-per-class", type=int, default=2)
    parser.add_argument("--pk-classes", type=int, default=3)
    parser.add_argument("--pk-images-per-class", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--finetune-only", action="store_true")
    args = parser.parse_args()
    report = run(
        stage=args.stage,
        epochs=args.epochs,
        development_per_class=args.development_per_class,
        pk_classes=args.pk_classes,
        pk_images_per_class=args.pk_images_per_class,
        inference_batch_size=args.inference_batch_size,
        early_stopping_patience=args.early_stopping_patience,
        finetune_only=args.finetune_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
