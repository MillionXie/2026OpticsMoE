"""Adapt only the final decoder from independent TRAIN CCD, then score fixed TEST CCD.

The checkpoint, optical masks, routers, upstream electronic path and task head
are protected. TEST is not loaded until the validation-selected epoch is sealed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary, STAGES
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
    OpenMojiEditingDataset, collate_samples, load_prompt_cache,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
from .lab_shs_capture import GROUPS, sha, write
from .profiles import install


def protected_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("shared_readout.decoder."):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def config(project: Path, device: torch.device) -> tuple[Settings, torch.nn.Module]:
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((project / "resolved_config.json").read_text(encoding="utf-8")))
    original_lab = project.parent / "OpenMoji_Lab_SHS_8um"
    for key in ("config_path", "data_dir", "asset_dir", "output_dir", "qwen_checkpoint", "prompt_cache_path", "optical_base_config", "legacy_warmstart_checkpoint"):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.config_path = project / "source/LightGenV2/tasks/t04_semantic_interaction/configs/routerfill_shared.yaml"
    cfg.data_dir = original_lab / "data"
    cfg.prompt_cache_path = cfg.data_dir / "token_embeddings_v1.pt"
    cfg.asset_dir = original_lab / "assets"
    cfg.svg_asset_dir = cfg.asset_dir / "openmoji-17.0.0-svg"
    cfg.qwen_checkpoint = original_lab / "frontend"
    cfg.optical_base_config = project / "source/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml"
    cfg.shared_readout_variant = "lowrank16"
    cfg.num_workers = 0
    name, expected = GROUPS["g5"]
    checkpoint = project / "weights" / name
    assert sha(checkpoint) == expected
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload.get("settings", {}).get("shared_readout_variant") == "lowrank16"
    model = t.build_model(cfg, device)
    model.load_state_dict(payload["model"], strict=True)
    install(model, "r0_base")
    model.eval().requires_grad_(False)
    for optic in model._optical_paths():
        optic.set_phase_dropout_active(False)
    return cfg, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--test-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    project, train_run, test_run, output = (p.resolve() for p in
                                            (args.project, args.train_run, args.test_run, args.output))
    if (output / "report.json").exists():
        raise RuntimeError("Completed adaptation must not be restarted")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(927)
    expected = GROUPS["g5"][1]
    for folder, scope in ((train_run, "train"), (test_run, "test")):
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        contract = report["contract"]
        assert report["status"] == "complete" and report["ccd_count"] == 6000
        assert contract.get("scope", "test") == scope and contract["count"] == 1000
        assert contract["checkpoint_sha256"] == expected
        assert contract["exposure_us"] == 2000 and contract["gain"] == "Gain_X4"
        assert contract["wait_ms"] == 240 and contract["camera_orientation"] == "flip_v"
        for stage in STAGES:
            assert len(list((folder / "ccd" / stage).glob("*.png"))) == 1000
            assert len(list((folder / "ccd" / stage).glob("*.json"))) == 1000
    phase_sha = {stage: sha(test_run / "phase" / f"{stage}.bmp") for stage in STAGES}
    assert all(sha(train_run / "phase" / f"{stage}.bmp") == phase_sha[stage] for stage in STAGES)
    audit_path = project / "data_train_adapt1000/split_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fit, val = set(audit["fit_ids"]), set(audit["validation_ids"])
    assert len(fit) == 800 and len(val) == 200 and not fit & val
    output.mkdir(parents=True, exist_ok=True)
    cfg, model = config(project, device)
    decoder = model.shared_readout.decoder
    initial = copy.deepcopy(decoder.state_dict())
    upstream_sha = protected_sha(model)
    write(output / "execution.json", {"status": "running", "checkpoint_sha256": expected,
          "trainable_prefix": "shared_readout.decoder", "trainable_parameters": sum(p.numel() for p in decoder.parameters()),
          "fit_count": 800, "validation_count": 200, "test_count": 1000,
          "selection": "TRAIN validation changed-cell accuracy only", "protected_before": upstream_sha,
          "split_audit_sha256": sha(audit_path)})

    def cache(folder: Path, data_dir: Path, manifest: Path, scope: str) -> dict:
        cfg.data_dir = data_dir
        dataset = OpenMojiEditingDataset(manifest, cfg, load_prompt_cache(data_dir / "token_embeddings_v1.pt"))
        assert len(dataset) == 1000
        ids = [r["sample_id"] for r in dataset.records]
        if scope == "train":
            assert set(ids) == fit | val
        else:
            assert not set(ids) & (fit | val)
        captured = []
        hook = decoder.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach().cpu().clone()))
        features, rows = [], []
        try:
            with torch.no_grad():
                for index in range(1000):
                    sid = ids[index] if scope == "train" else f"test_{index:05d}"
                    batch = collate_samples([dataset[index]])
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    frames = {}
                    for stage in STAGES:
                        base = folder / "ccd" / stage / sid
                        receipt = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
                        assert receipt["sample_id"] == sid and receipt["stage"] == stage
                        assert receipt["phase_sha256"] == phase_sha[stage]
                        assert receipt["exposure"]["exposure_us"] == 2000
                        assert receipt["exposure"]["gain"] == "Gain_X4"
                        assert receipt["wait_ms"] == 240 and receipt["canonical_orientation"] == "flip_v"
                        frames[stage] = torch.from_numpy(np.asarray(Image.open(base.with_suffix(".png")), np.float32).copy()[None] / 255)
                    with OpticalBoundary(model, frames):
                        result = model(batch["source_image"], batch["prompt_hidden"])
                    assert len(captured) == 1
                    features.append(captured.pop())
                    row = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()
                           if k not in ("source_image", "prompt_hidden")}
                    row["task_logits"] = result["task_logits"].detach().cpu()
                    rows.append(row)
                    if index % 100 == 0:
                        write(output / "progress.json", {"status": "caching", "scope": scope, "completed": index + 1})
        finally:
            hook.remove()
        value = {"features": torch.cat(features), "rows": rows, "ids": ids}
        torch.save(value, output / f"{scope}_features.pt")
        return value

    train_data = project / "data_train_adapt1000"
    train = cache(train_run, train_data, train_data / "capture_train.jsonl", "train")
    fit_idx = [i for i, sid in enumerate(train["ids"]) if sid in fit]
    val_idx = [i for i, sid in enumerate(train["ids"]) if sid in val]
    assert len(fit_idx) == 800 and len(val_idx) == 200

    def mini(data: dict, indices: list[int]):
        rows = [data["rows"][i] for i in indices]
        batch = {key: (torch.cat([row[key] for row in rows]).to(device) if torch.is_tensor(rows[0][key])
                       else sum([row[key] for row in rows], [])) for key in rows[0]}
        return data["features"][indices].to(device), batch

    def evaluate(data: dict, indices: list[int], export: bool = False) -> dict:
        meter = MetricAccumulator()
        decoder.eval()
        samples = []
        with torch.no_grad():
            for start in range(0, len(indices), 32):
                x, batch = mini(data, indices[start:start + 32])
                cat, edit = decoder(x)
                row, _prediction, _ = meter.update({"category_logits": cat, "edit_logits": edit,
                                                    "task_logits": batch["task_logits"]}, batch)
                if export:
                    samples.extend(row)
        if export:
            write(output / "test_samples.json", samples)
        return meter.compute()

    baseline_fit = evaluate(train, fit_idx)
    baseline_val = evaluate(train, val_idx)
    best_score = baseline_val["overall"]["changed_cell_accuracy"]
    selected = 0
    best = copy.deepcopy(initial)
    decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-5, weight_decay=.01)
    history = []
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        losses = []
        order = torch.randperm(len(fit_idx)).tolist()
        for start in range(0, len(order), 32):
            indices = [fit_idx[i] for i in order[start:start + 32]]
            x, batch = mini(train, indices)
            cat, edit = decoder(x)
            weight = 1 + 11 * batch["edit_grid"] + 2 * batch["target_grid"].gt(0)
            ce = F.cross_entropy(cat, batch["target_grid"].long(), reduction="none")
            loss = (ce * weight).sum() / weight.sum() + 1.25 * F.binary_cross_entropy_with_logits(
                edit, batch["edit_grid"], pos_weight=edit.new_tensor(8.))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        fit_metrics = evaluate(train, fit_idx)
        val_metrics = evaluate(train, val_idx)
        score = val_metrics["overall"]["changed_cell_accuracy"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "fit": fit_metrics, "validation": val_metrics})
        write(output / "history.json", history)
        write(output / "progress.json", {"status": "training", "epoch": epoch, "validation": score,
                                          "selected_epoch": selected})
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "validation": score}), flush=True)
        if score > best_score:
            best_score, selected, best = score, epoch, copy.deepcopy(decoder.state_dict())
    last = copy.deepcopy(decoder.state_dict())
    assert protected_sha(model) == upstream_sha
    checkpoint = torch.load(project / "weights" / GROUPS["g5"][0], map_location="cpu", weights_only=False)
    decoder.load_state_dict(best)
    chosen = copy.deepcopy(checkpoint)
    chosen["model"] = model.state_dict()
    chosen["physical_adaptation"] = {"selected_epoch": selected, "validation_score": best_score,
                                      "fit_ids": sorted(fit), "validation_ids": sorted(val),
                                      "test_used_for_selection": False, "prefix": "shared_readout.decoder"}
    torch.save(chosen, output / "best.pt")
    decoder.load_state_dict(last)
    last_payload = copy.deepcopy(checkpoint)
    last_payload["model"] = model.state_dict()
    torch.save(last_payload, output / "last.pt")
    decoder.load_state_dict(best)
    decoder.requires_grad_(False)
    assert protected_sha(model) == upstream_sha
    test_data = project.parent / "OpenMoji_Lab_SHS_8um/data"
    test = cache(test_run, test_data, test_data / "test.jsonl", "test")
    physical = evaluate(test, list(range(1000)), export=True)
    decoder.load_state_dict(initial)
    baseline = evaluate(test, list(range(1000)))
    decoder.load_state_dict(best)
    expected_baseline = json.loads((test_run / "report.json").read_text())["physical_metrics"]["overall"]["changed_cell_accuracy"]
    assert abs(baseline["overall"]["changed_cell_accuracy"] - expected_baseline) < 1e-8
    assert protected_sha(model) == upstream_sha
    write(output / "report.json", {"status": "complete", "selected_epoch": selected,
          "validation_best": best_score, "baseline_fit": baseline_fit, "baseline_validation": baseline_val,
          "baseline_test": baseline, "physical_test": physical, "best_sha256": sha(output / "best.pt"),
          "protected_unchanged": True, "selection": "800 TRAIN FIT / 200 TRAIN VAL; original TEST after selection only"})
    write(output / "progress.json", {"status": "complete", "selected_epoch": selected})


if __name__ == "__main__":
    main()
