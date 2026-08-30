"""Run the four measured-CCD Qwen adaptation stages on the laboratory PC.

The wrapper owns all long paths and always uses a development split for
checkpoint selection.  The sealed Caltech101 test split is evaluated once,
after the selected checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path


STAGES = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)
OFFLINE_MODEL_RELATIVE = Path("models/Qwen3-VL-Embedding-2B")
OFFLINE_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
)
PROFILES = {
    "strong_noise": {
        "project": (
            "qwen3_vl_embedding_2b_caltech101_four_layer_"
            "optical_retrieval_10cm_ccd_noise1"
        ),
        "config": "configs/release/stage_hardware_canonical_ccd.yaml",
        "initial_checkpoint": "ema.pt",
        "session": "four",
    },
    "accuracy_first": {
        "project": (
            "qwen3_vl_embedding_2b_caltech101_four_layer_"
            "optical_retrieval_10cm_early_robust_tradeoff"
        ),
        "config": "configs/hardware/accuracy_first_quick210.yaml",
        "initial_checkpoint": "accuracy_first_ema.pt",
        "session": "four_accuracy_first",
    },
    "accuracy_first_full": {
        "project": (
            "qwen3_vl_embedding_2b_caltech101_four_layer_"
            "optical_retrieval_10cm_early_robust_tradeoff"
        ),
        "config": "configs/hardware/accuracy_first_full.yaml",
        "initial_checkpoint": "accuracy_first_ema.pt",
        "session": "four_accuracy_first_full",
    },
    "balanced": {
        "project": (
            "qwen3_vl_embedding_2b_caltech101_four_layer_"
            "optical_retrieval_10cm_early_robust_tradeoff"
        ),
        "config": "configs/hardware/balanced_quick210.yaml",
        "initial_checkpoint": "balanced_ema.pt",
        "session": "four_balanced",
    },
    "balanced_full": {
        "project": (
            "qwen3_vl_embedding_2b_caltech101_four_layer_"
            "optical_retrieval_10cm_early_robust_tradeoff"
        ),
        "config": "configs/hardware/balanced_full.yaml",
        "initial_checkpoint": "balanced_ema.pt",
        "session": "four_balanced_full",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paths(root: Path, stage: str, profile: str) -> tuple[Path, Path, Path]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown local four-stage profile {profile!r}")
    contract = PROFILES[profile]
    project = root / "experiments" / str(contract["project"])
    config = project / str(contract["config"])
    session = root / "experiments" / "lab_qwen" / str(contract["session"])
    index = STAGES.index(stage)
    checkpoint = (
        root
        / "experiments"
        / "lab_qwen"
        / "model"
        / str(contract["initial_checkpoint"])
        if index == 0
        else session / "checkpoints" / f"after_{STAGES[index - 1]}.pt"
    )
    return config, session, checkpoint


def _configure_local_backbone(
    settings: object, root: Path, model_dir: str | Path | None
) -> dict[str, object]:
    # The laboratory workflow is deliberately offline.  Set these before any
    # validation so a missing bundle can never silently fall back to the Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    candidate = (
        Path(model_dir).expanduser()
        if model_dir is not None
        else root / OFFLINE_MODEL_RELATIVE
    )
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    missing = [name for name in OFFLINE_MODEL_REQUIRED_FILES if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Offline Qwen model directory is incomplete and network fallback "
            f"is forbidden: {candidate}; missing={missing}. Copy the complete "
            "Qwen3-VL-Embedding-2B snapshot into models/Qwen3-VL-Embedding-2B."
        )
    settings.model_id = str(candidate)
    settings.cache_dir = None
    settings.local_files_only = True
    return {
        "mode": "bundled_local_snapshot",
        "model_id": str(candidate),
        "local_files_only": True,
        "offline_candidate_present": True,
    }


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
    profile: str = "strong_noise",
    model_dir: str | Path | None = None,
) -> dict[str, object]:
    # Heavy Qwen/Torch imports occur only for this explicit local-training command.
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
        hardware_bridge as bridge,
    )
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
        seed_everything,
    )

    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    if profile not in PROFILES:
        raise ValueError(f"Unknown local four-stage profile {profile!r}")
    project_module = str(PROFILES[profile]["project"])
    modeling = importlib.import_module(f"experiments.{project_module}.modeling")
    settings_module = importlib.import_module(f"experiments.{project_module}.settings")
    build_hybrid_student = modeling.build_hybrid_student
    load_backbone = modeling.load_backbone
    load_settings = settings_module.load_settings
    root = _repo_root()
    config, session, checkpoint = _paths(root, stage, profile)
    if not config.is_file():
        raise FileNotFoundError(f"Local four-stage config is missing: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Previous-stage checkpoint is missing: {checkpoint}\n"
            "If you are switching from a server run midway, download that exact "
            f"after_<previous-stage>.pt file into {session / 'checkpoints'}."
        )

    bridge.build_hybrid_student = build_hybrid_student
    bridge.load_backbone = load_backbone
    bridge.load_settings = load_settings
    settings = load_settings(config)
    backbone = _configure_local_backbone(settings, root, model_dir)
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
        "profile": profile,
        "stage": stage,
        "checkpoint": str(selected),
        "metrics": str(
            session
            / f"{STAGES.index(stage) + 1:02d}_{stage}"
            / "finetune_metrics.json"
        ),
        "selection": "development_top1_then_ce; sealed_test_once_after_selection",
        "backbone": backbone,
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
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="accuracy_first_full",
        help=(
            "Must match both the initial checkpoint and capture population. "
            "Use accuracy_first_full for the full-data physical experiment; "
            "accuracy_first is the quick210 diagnostic only."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Offline Qwen snapshot directory. By default the wrapper auto-detects "
            "models/Qwen3-VL-Embedding-2B below the extracted laboratory bundle."
        ),
    )
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
        profile=args.profile,
        model_dir=args.model_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
