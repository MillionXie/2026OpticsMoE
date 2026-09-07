"""Portable preparation and checkpoint audit; never overwrite historical runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
CONFIG = TASK / "configs/handoff_balance.yaml"
CHECKPOINT = TASK / "runs/simulation/optical_router_moe_dc20_kd1_balance1_seed42/best_checkpoint.pt"


def verify_package(root: Path) -> dict:
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes package")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Package SHA mismatch: {entry['path']}")
    return {"verified_files": len(manifest["files"]), "source_commit": manifest["source_commit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("verify", "prepare", "check", "smoke", "evaluate"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "verify":
        print(json.dumps(verify_package(ROOT), indent=2))
        return
    if args.phase == "prepare":
        import yaml
        model = args.model_path.resolve() if args.model_path else None
        if model is None or not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            raise ValueError("--model-path must contain local Qwen3-VL-Embedding-2B config and safetensors")
        # Fixed placement preserves the inherited relative-path contract.
        if args.config.resolve().parent != TASK / "configs":
            raise ValueError("Place handoff config inside the task configs directory")
        # Model ID is checkpoint identity, not a machine-specific path. Reuse
        # the backend's HF-cache resolver without relaxing metadata validation.
        cache = ROOT / ".handoff_hf"
        snapshot = cache / "models--Qwen--Qwen3-VL-Embedding-2B/snapshots/local"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() or snapshot.is_symlink():
            if snapshot.resolve() != model:
                raise ValueError("Existing handoff model link points to a different model")
        else:
            snapshot.symlink_to(model, target_is_directory=True)
        config = {"base_config": "optical_router_moe_dc20_kd1_balance1.yaml",
                  "qwen": {"model_id": "Qwen/Qwen3-VL-Embedding-2B",
                           "cache_dir": str(cache), "local_files_only": True},
                  "batching": {"num_workers": 0},
                  "output_dir": "../runs/simulation/handoff_retrain"}
        with args.config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        print(json.dumps({"config": str(args.config), "model_path": str(model)}, indent=2))
        return
    from . import optical_moe as task
    import torch
    settings = task.load_settings(args.config.resolve())
    raw = task._read_config(args.config.resolve())
    contract = task.load_contract(task._resolve_from_config(args.config.resolve(),
                                                          str(task._nested(raw, "dataset.dataset_root"))))
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import resolve_cached_model_source
    source = resolve_cached_model_source(settings.model_id, settings.cache_dir)
    if not Path(source).is_dir():
        raise FileNotFoundError(f"Local Qwen model missing: {source}")
    if not settings.router_source_checkpoint.is_file():
        raise FileNotFoundError(settings.router_source_checkpoint)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    report = {"train": len(contract.train), "test": len(contract.test),
              "titles": len(contract.titles), "model": settings.model_id, "model_source": source,
              "checkpoint": str(args.checkpoint), "phase": args.phase}
    if args.phase == "check":
        print(json.dumps(report, indent=2))
        return
    if args.run_dir is None:
        raise ValueError("smoke/evaluate require a NEW --run-dir")
    settings.output_dir = args.run_dir.resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    task.seed_everything(42)
    loaded = task.load_backbone(settings, torch.device(args.device))
    replacement, readout = task.build_student(loaded, settings)
    try:
        report["initialization"] = task.initialize_student(settings, replacement, readout)
        task.load_checkpoint(args.checkpoint, replacement, readout)
        if args.phase == "smoke":
            contract = replace(contract, test=contract.test[:2])
        report["evaluated_test_samples"] = len(contract.test)
        report["metrics"] = task.evaluate(loaded, replacement, readout, contract, settings, write_outputs=True)
        report["warning"] = "Smoke is not full-test performance" if args.phase == "smoke" else "Checkpoint was selected on periodic test"
        task.write_json(settings.output_dir / "handoff_report.json", report)
        print(json.dumps(report, indent=2))
    finally:
        replacement.close()


if __name__ == "__main__":
    main()
