"""Offline, test-only verification of the exact selected OpenMoji e45 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


CHECKPOINT_SHA = "03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--boundary", action="store_true")
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root / "runtime_exact"))
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
        OpenMojiEditingDataset, collate_samples, load_prompt_cache,
    )
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator

    config = root / "runtime_exact/LightGenV2/tasks/t04_semantic_interaction/configs/layered_scene_exp05_dc30_ccdsmall.yaml"
    cfg = load_settings(config)
    cfg.data_dir = root / "data"
    cfg.prompt_cache_path = cfg.data_dir / "token_embeddings_v1.pt"
    cfg.output_dir = root / "offline_verify"
    legacy = root.parent / "OpenMoji_Lab_SHS_8um"
    cfg.qwen_checkpoint = legacy / "frontend"
    cfg.asset_dir = legacy / "assets"
    if cfg.layout_version != "layered_anchor6_svg_v3" or cfg.test_samples != 1000:
        raise RuntimeError("Wrong data layout or test population")
    checkpoint = root / "weights/selected_checkpoint.pt"
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("Selected checkpoint SHA mismatch")
    records = [json.loads(line) for line in cfg.test_manifest.read_text(encoding="utf-8").splitlines()]
    if len(records) != 1000:
        raise RuntimeError("Expected 1000 test records")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["architecture"] != model.checkpoint_architecture or payload["epoch"] != 45:
        raise RuntimeError("Selected architecture/epoch mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for path in model._optical_paths():
        path.set_phase_dropout_active(False)
    data = OpenMojiEditingDataset(cfg.test_manifest, cfg, load_prompt_cache(cfg.prompt_cache_path))
    if not 1 <= args.limit <= len(data):
        raise ValueError("limit outside test set")
    acc = MetricAccumulator()
    boundary_rows = []
    with torch.inference_mode():
        for index in range(args.limit):
            batch = collate_samples([data[index]])
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            output = model(batch["source_image"], batch["prompt_hidden"])
            if args.boundary:
                from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary, STAGES
                with OpticalBoundary(model) as tap:
                    replayed = model(batch["source_image"], batch["prompt_hidden"])
                error = max(float((output[name] - replayed[name]).abs().max())
                            for name in ("category_logits", "edit_logits", "task_logits"))
                if tuple(tap.amplitudes) != STAGES or error > 2e-4:
                    raise RuntimeError(f"Six optical boundary replay failed: {error}")
                boundary_rows.append({"sample_id": data.records[index]["sample_id"],
                                      "max_logit_error": error,
                                      "stages": list(tap.amplitudes)})
            acc.update(output, batch)
            if (index + 1) % 100 == 0:
                print(f"VERIFIED {index + 1}/{args.limit}", flush=True)
    metrics = acc.compute()
    report = {"status": "complete", "checkpoint_sha256": CHECKPOINT_SHA,
              "layout_version": cfg.layout_version, "epoch": 45, "samples": args.limit,
              "device": str(device), "metrics": metrics}
    if args.boundary:
        report["boundary_replay"] = boundary_rows
    out = root / f"offline_verify_{args.limit:04d}{'_boundary' if args.boundary else ''}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"]["overall"], ensure_ascii=False), flush=True)
    if args.limit == 1000 and abs(metrics["overall"]["changed_cell_accuracy"] - 0.8765) > 0.002:
        raise RuntimeError("Selected-checkpoint simulation differs by more than cross-GPU tolerance")


if __name__ == "__main__":
    main()
