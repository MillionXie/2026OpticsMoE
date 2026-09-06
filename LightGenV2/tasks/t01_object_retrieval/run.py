"""Single audited entry point for the three formal Caltech101 systems."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.cache_teacher_embeddings import (
    TeacherEmbeddingStore,
    build_teacher_embedding_cache,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.evaluate_retrieval import (
    evaluate_all_systems,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    environment_report,
    seed_everything,
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.retrieval_metrics import (
    evaluate_embeddings,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    load_checkpoint,
    train_optical_retrieval,
)

from .modeling import (
    build_student,
    initialize_student,
    load_backbone,
    parameter_fairness_contract,
)
from .settings import load_settings, save_resolved_config


TASK_DIR = Path(__file__).resolve().parent
PROFILES = {
    "main": "moe_optical_router_scale_matched.yaml",
    "d2nn": "d2nn_active_expert_matched.yaml",
    "qwen": "qwen_frozen_embedding.yaml",
}
PHASES = {"prepare", "train", "evaluate", "all"}


def _device(settings: Any) -> torch.device:
    requested = str(settings.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=TASK_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _configure_run(settings: Any, args: argparse.Namespace) -> None:
    settings.router_optimization_seed = int(args.seed)
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).expanduser().resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(settings)
    environment = environment_report()
    environment.update(
        {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty_porcelain": _git_value("status", "--short"),
        }
    )
    write_json(settings.output_dir / "environment.json", environment)
    write_json(
        settings.output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "task": "t01_object_retrieval",
            "profile": args.profile,
            "variant": settings.lightgen_model_variant,
            "phase": args.phase,
            "optimization_seed": int(args.seed),
            "dataset_split_seed": int(settings.random_seed),
            "command": "python -m LightGenV2.tasks.t01_object_retrieval.run "
            f"--profile {args.profile} --phase {args.phase} --seed {args.seed}",
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "test_selection": (
                "maximum periodically observed EMA test Top-1 every 5 epochs"
                if args.profile != "qwen"
                else "none; frozen model evaluated once"
            ),
        },
    )


def _plot_confusion(
    matrix: torch.Tensor, class_names: tuple[str, ...], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, label="Query count")
    axis.set_xticks(range(len(class_names)), class_names, rotation=55, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Frozen Qwen3-VL-Embedding-2B")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _evaluate_frozen_qwen(
    loaded: Any, bundle: Any, settings: Any, *, force_cache: bool
) -> dict[str, Any]:
    build_teacher_embedding_cache(loaded, bundle, settings, force=force_cache)
    store = TeacherEmbeddingStore(settings.teacher_cache_path, bundle, settings)
    result = evaluate_embeddings(
        store.lookup(bundle.test_samples),
        bundle.test_samples,
        store.lookup(bundle.gallery_samples),
        bundle.gallery_samples,
        bundle.class_names,
        settings.gallery_aggregation,
        system_name="frozen_qwen3_vl_embedding_2b",
    )
    total = sum(parameter.numel() for parameter in loaded.model.parameters())
    metrics = {
        **result.metrics,
        "model_id": settings.model_id,
        "instruction": settings.instruction,
        "embedding_method": "last valid token, first 64 Matryoshka dimensions, L2 normalized",
        "model_frozen": True,
        "trainable_parameters": 0,
        "frozen_parameters": total,
        "checkpoint_selection": "none; one deterministic frozen-model evaluation",
        "manifest_sha256": bundle.manifest_digest,
    }
    write_json(settings.output_dir / "frozen_qwen_metrics.json", metrics)
    write_json(settings.output_dir / "student_metrics.json", metrics)
    write_csv(
        settings.output_dir / "retrieval_results.csv",
        result.rows,
        list(result.rows[0]),
    )
    _plot_confusion(
        result.confusion,
        bundle.class_names,
        settings.output_dir / "confusion_matrix.png",
    )
    return metrics


def _preferred_checkpoint(settings: Any, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        path = settings.output_dir / settings.lightgen_primary_checkpoint
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation checkpoint is missing: {path}. Train this profile first "
            "or pass --checkpoint explicitly."
        )
    return path


def _curate_student_artifacts(
    settings: Any, replacement: Any, readout: Any
) -> dict[str, Any]:
    """Retain one selected checkpoint, one last checkpoint and readable plots.

    The compatibility trainer creates several live/EMA/train-loss variants and
    periodic phase-only PT files.  T01 has one explicit selection contract, so
    keeping all of those aliases makes the formal result harder to identify.
    The original metrics JSON remains, but its checkpoint path is updated to
    the canonical ``best_checkpoint.pt``.
    """

    output = settings.output_dir
    selected_source = output / "ema_best_observed_test_checkpoint.pt"
    if not selected_source.is_file():
        raise FileNotFoundError(
            "The configured EMA test-selected checkpoint was not produced: "
            f"{selected_source}"
        )
    payload = load_checkpoint(selected_source, replacement, readout)
    best = output / "best_checkpoint.pt"
    shutil.copy2(selected_source, best)
    visualization = output / "best_visualization"
    visualization.mkdir(parents=True, exist_ok=True)
    replacement.save_multiplane_phase_preview(
        visualization / "phase_preview.png",
        title=(
            "Selected best optical phase "
            f"(epoch {int(payload.get('epoch', -1))})"
        ),
    )
    selection_path = output / "metrics" / "ema_best_observed_test.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["checkpoint"] = str(best)
        write_json(selection_path, selection)

    removed: list[str] = []
    for directory in (output / "phase_training", output / "best_optical_artifacts"):
        if directory.is_dir():
            shutil.rmtree(directory)
            removed.append(str(directory.relative_to(output)))
    keep = {"best_checkpoint.pt", "last_checkpoint.pt"}
    for candidate in output.glob("*.pt"):
        if candidate.name not in keep:
            candidate.unlink()
            removed.append(candidate.name)
    report = {
        "schema_version": 1,
        "policy": "retain canonical best and live last model checkpoints only",
        "best_checkpoint": str(best),
        "best_epoch": int(payload.get("epoch", -1)),
        "last_checkpoint": str(output / "last_checkpoint.pt"),
        "periodic_phase_pt_retained": False,
        "best_phase_visualization": str(
            visualization / "phase_preview.png"
        ),
        "removed": removed,
    }
    write_json(output / "artifact_retention.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = TASK_DIR / "configs" / PROFILES[args.profile]
    settings = load_settings(config)
    _configure_run(settings, args)
    seed_everything(int(args.seed))
    bundle = prepare_caltech101_subset(settings, persist=True)
    write_json(
        settings.output_dir / "parameter_fairness_contract.json",
        parameter_fairness_contract(settings),
    )
    if args.phase == "prepare":
        return {"status": "prepared", "counts": bundle.metadata["counts"]}

    loaded = load_backbone(settings, _device(settings))
    settings.resolve_architecture(loaded.model)
    save_resolved_config(settings)
    if args.profile == "qwen":
        if args.phase == "train":
            raise ValueError("The frozen Qwen baseline has no training phase")
        return _evaluate_frozen_qwen(
            loaded, bundle, settings, force_cache=args.force_teacher_cache
        )

    replacement, readout = build_student(loaded, settings)
    try:
        write_json(
            settings.output_dir / "student_architecture.json",
            replacement.student_architecture_report(),
        )
        if args.phase in {"train", "all"}:
            initialization = initialize_student(settings, replacement, readout)
            write_json(
                settings.output_dir / "initialization_report.json", initialization
            )
            train_optical_retrieval(
                loaded,
                replacement,
                readout,
                bundle,
                None,
                settings,
                resume_checkpoint=(
                    Path(args.resume_checkpoint).expanduser().resolve()
                    if args.resume_checkpoint
                    else None
                ),
            )
            _curate_student_artifacts(settings, replacement, readout)
            if args.phase == "train":
                return {"status": "trained", "output_dir": str(settings.output_dir)}
        checkpoint = _preferred_checkpoint(settings, args.checkpoint)
        systems = evaluate_all_systems(
            loaded,
            replacement,
            readout,
            bundle,
            None,
            settings,
            checkpoint,
        )
        write_json(
            settings.output_dir / "fusion_diagnostics_last_batch.json",
            replacement.fusion_diagnostics(),
        )
        return systems["student"].metrics
    finally:
        replacement.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LightGenV2 T01: optical-Router MoE and two fair baselines"
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--force-teacher-cache", action="store_true")
    args = parser.parse_args()
    if args.resume_checkpoint and args.phase not in {"train", "all"}:
        parser.error("--resume-checkpoint is only valid for train/all")
    if args.checkpoint and args.phase not in {"evaluate", "all"}:
        parser.error("--checkpoint is only valid for evaluate/all")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
