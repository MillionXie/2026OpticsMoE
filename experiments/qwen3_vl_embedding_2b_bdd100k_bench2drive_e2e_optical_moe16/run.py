from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .behavior_cloning import (
    build_policy,
    evaluate_bc,
    train_behavior_cloning,
)
from .datasets_bdd100k import build_bdd_records
from .datasets_bench2drive import build_bench2drive_splits
from .io_utils import (
    atomic_torch_save,
    ensure_output_dirs,
    seed_everything,
    torch_load,
    write_json,
)
from .modeling import (
    BDDPretrainModel,
    NativeVisionFeatureExtractor,
    build_backbone,
    load_vision_backbone,
)
from .pca import fit_bdd_pca, load_pca
from .pretraining import bdd_loader, export_backbone, train_bdd_backbone
from .sac import train_sac
from .settings import load_settings, save_resolved_config
from .smoke import run_smoke


PHASES = (
    "prepare_data",
    "fit_pca",
    "bdd_pretrain",
    "export_backbone",
    "prepare_bench2drive",
    "bc_stage1",
    "bc_stage2",
    "bc_all",
    "bc_evaluate",
    "sac_train",
    "all",
    "smoke",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BDD100K-pretrained Optical MoE16 for Bench2Drive"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    arguments = parser.parse_args(argv)
    settings = load_settings(arguments.config)
    ensure_output_dirs(settings.output_dir)
    save_resolved_config(settings)
    seed_everything(settings.random_seed)
    phase = arguments.phase
    if phase == "smoke":
        result = run_smoke(settings)
        print(f"Smoke passed: {result}", flush=True)
        return 0
    if phase in {"prepare_data", "prepare_bench2drive"}:
        _prepare(settings, include_bdd=phase == "prepare_data")
        return 0
    if phase == "fit_pca":
        _fit_pca(settings)
        return 0
    if phase == "bdd_pretrain":
        _bdd_pretrain(settings)
        return 0
    if phase == "export_backbone":
        _export(settings)
        return 0
    if phase in {"bc_stage1", "bc_stage2", "bc_all", "bc_evaluate"}:
        _behavior_cloning(settings, phase)
        return 0
    if phase == "sac_train":
        _sac(settings)
        return 0
    if phase == "all":
        use_bdd_pretraining = settings.bc_backbone_initialization == "bdd_pretrained"
        _prepare(settings, include_bdd=use_bdd_pretraining)
        if use_bdd_pretraining:
            _fit_pca(settings)
            _bdd_pretrain(settings)
        else:
            print(
                "[all] Scratch Optical Backbone selected: skipping BDD100K, PCA, "
                "and BDD feature pretraining.",
                flush=True,
            )
        _behavior_cloning(settings, "bc_all")
        if settings.sac_env_factory:
            _sac(settings)
        else:
            print(
                "[all] Offline BDD pretraining and behavior cloning completed. "
                "SAC was not started because sac.env_factory is empty.",
                flush=True,
            )
        return 0
    raise AssertionError(phase)


def _device(settings: Any) -> torch.device:
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("qwen.device requests CUDA, but CUDA is unavailable")
    return torch.device(settings.device)


def _prepare(settings: Any, *, include_bdd: bool) -> None:
    payload: dict[str, Any] = {}
    if include_bdd:
        train = build_bdd_records(settings, settings.bdd_train_split)
        test = build_bdd_records(settings, settings.bdd_test_split)
        payload["bdd100k"] = {"train": len(train), "test": len(test)}
    bench_train, bench_validation = build_bench2drive_splits(settings)
    payload["bench2drive"] = {
        "train": len(bench_train),
        "offline_validation": len(bench_validation),
    }
    write_json(settings.output_dir / "metrics" / "data_summary.json", payload)
    print(f"Prepared data: {payload}", flush=True)


def _fit_pca(settings: Any) -> None:
    device = _device(settings)
    records = build_bdd_records(settings, settings.bdd_train_split)
    loaded = load_vision_backbone(settings, device)
    teacher = NativeVisionFeatureExtractor(loaded)
    metrics = fit_bdd_pca(
        teacher,
        loaded.processor,
        bdd_loader(records, settings, training=False),
        settings,
    )
    teacher.close()
    print(f"Fitted BDD PCA: {metrics}", flush=True)


def _bdd_pretrain(settings: Any) -> None:
    device = _device(settings)
    train_records = build_bdd_records(settings, settings.bdd_train_split)
    test_records = build_bdd_records(settings, settings.bdd_test_split)
    loaded = load_vision_backbone(settings, device)
    teacher = NativeVisionFeatureExtractor(loaded)
    backbone = build_backbone(loaded, settings)
    model = BDDPretrainModel(backbone).to(device)
    projection, _ = load_pca(settings.pca_path, device)
    train_bdd_backbone(
        model,
        teacher,
        projection,
        loaded.processor,
        train_records,
        test_records,
        settings,
        device,
    )
    teacher.close()


def _export(settings: Any) -> None:
    device = _device(settings)
    loaded = load_vision_backbone(settings, device)
    source = settings.output_dir / "checkpoints" / "bdd_pretrain_best_full.pt"
    if source.is_file():
        payload = torch_load(source)
        source_state = payload["backbone"]
        temporary = settings.output_dir / "checkpoints" / ".export_source.pt"
        atomic_torch_save(temporary, source_state)
        backbone = build_backbone(loaded, settings, temporary)
        temporary.unlink(missing_ok=True)
    else:
        backbone = build_backbone(
            loaded, settings, settings.pretrained_backbone_checkpoint
        )
    report = export_backbone(backbone, settings)
    print(f"Exported backbone: {report}", flush=True)


def _behavior_cloning(settings: Any, phase: str) -> None:
    device = _device(settings)
    train_records, validation_records = build_bench2drive_splits(settings)
    loaded = load_vision_backbone(settings, device)
    backbone = build_backbone(loaded, settings, _initial_backbone_checkpoint(settings))
    policy = build_policy(backbone, settings)
    if phase in {"bc_stage1", "bc_all"}:
        train_behavior_cloning(
            policy,
            loaded.processor,
            train_records,
            validation_records,
            settings,
            device,
            stage=1,
        )
    if phase in {"bc_stage2", "bc_all"}:
        train_behavior_cloning(
            policy,
            loaded.processor,
            train_records,
            validation_records,
            settings,
            device,
            stage=2,
        )
    if phase == "bc_evaluate":
        checkpoint = settings.output_dir / "checkpoints" / "bc_policy_best.pt"
        result = evaluate_bc(
            policy,
            loaded.processor,
            validation_records,
            settings,
            device,
            checkpoint,
        )
        print(f"BC evaluation: {result}", flush=True)


def _sac(settings: Any) -> None:
    device = _device(settings)
    loaded = load_vision_backbone(settings, device)
    backbone = build_backbone(loaded, settings, _initial_backbone_checkpoint(settings))
    policy = build_policy(backbone, settings)
    result = train_sac(policy, loaded.processor, settings, device)
    print(f"SAC complete: {result}", flush=True)


def _initial_backbone_checkpoint(settings: Any) -> Path | None:
    if settings.bc_backbone_initialization == "scratch":
        return None
    checkpoint = settings.pretrained_backbone_checkpoint
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            "BDD-pretrained behavior cloning requires the exported Optical "
            f"Backbone checkpoint, but it is missing: {checkpoint}. Run "
            "--phase fit_pca and --phase bdd_pretrain, or use a scratch config."
        )
    return checkpoint


if __name__ == "__main__":
    raise SystemExit(main())
