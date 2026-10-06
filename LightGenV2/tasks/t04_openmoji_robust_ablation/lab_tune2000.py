"""Adapt only the final decoder from independent TRAIN CCD, then score fixed TEST CCD.

The checkpoint, optical masks, routers, upstream electronic path and task head
are protected. User-authorized TEST development selection, never TEST gradients.
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
from .lab_shs_capture import checkpoint_identity, sha, write
from .profiles import install


def protected_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("shared_readout.decoder."):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def group_checkpoint(group: str) -> tuple[str, str]:
    identity = checkpoint_identity(group, Path(__file__).parent / "configs/lab/rank64_20261002.json")
    return identity["filename"], identity["sha256"]


def config(project: Path, device: torch.device, group: str = "g5") -> tuple[Settings, torch.nn.Module]:
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((project / "resolved_config.json").read_text(encoding="utf-8")))
    original_lab = project.parent / "OpenMoji_Lab_SHS_8um"
    for key in ("config_path", "data_dir", "asset_dir", "output_dir", "qwen_checkpoint", "prompt_cache_path", "optical_base_config", "legacy_warmstart_checkpoint"):
        setattr(cfg, key, Path(getattr(cfg, key)))
    source_root = Path(__file__).resolve().parents[3]
    cfg.config_path = source_root / "LightGenV2/tasks/t04_semantic_interaction/configs/routerfill_shared.yaml"
    cfg.data_dir = original_lab / "data"
    cfg.prompt_cache_path = cfg.data_dir / "token_embeddings_v1.pt"
    cfg.asset_dir = original_lab / "assets"
    cfg.svg_asset_dir = cfg.asset_dir / "openmoji-17.0.0-svg"
    cfg.qwen_checkpoint = original_lab / "frontend"
    cfg.optical_base_config = source_root / "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml"
    cfg.shared_readout_variant = "lowrank64"
    cfg.num_workers = 0
    name, expected = group_checkpoint(group)
    checkpoint = project / "weights" / name
    assert sha(checkpoint) == expected
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload.get("settings", {}).get("shared_readout_variant") == "lowrank64"
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
    parser.add_argument("--group", choices=("g2", "g5"), default="g5")
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--test-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-run", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    project, train_run, test_run, output = (p.resolve() for p in
                                            (args.project, args.train_run, args.test_run, args.output))
    if (output / "report.json").exists():
        raise RuntimeError("Completed adaptation must not be restarted")
    extra_run = args.extra_run.resolve()
    base_cache = args.base_cache.resolve()
    extra_data = project / "data_train_extra1000"
    extra_audit = json.loads((extra_data / "split_audit.json").read_text(encoding="utf-8"))
    extra_ids = set(extra_audit["fit_ids"])
    assert len(extra_ids) == 1000
    assert all(extra_audit[key] == 0 for key in ("test_ids_overlap", "test_source_overlap", "prior_train_ids_overlap", "prior_train_source_overlap", "internal_duplicate_sources"))
    assert sha(extra_data / "capture_train.jsonl") == "427e9fc203dc2c38bd3d1336ee881f9447ca07923a3a9aab05b3c67bf82db403"
    training_device = torch.device(args.device)
    # Reconstruct the captured CPU-model deployment before moving only the decoder.
    # This preserves exact baseline alignment and does not use a CUDA frontend cache.
    device = torch.device("cpu")
    cache_device = device
    torch.set_num_threads(4)
    torch.manual_seed(927)
    expected = group_checkpoint(args.group)[1]
    for folder, scope in ((train_run, "train"), (extra_run, "train"), (test_run, "test")):
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
    assert all(sha(folder / "phase" / f"{stage}.bmp") == phase_sha[stage]
               for folder in (train_run, extra_run) for stage in STAGES)
    audit_path = project / "data_train_adapt1000/split_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fit, val = set(audit["fit_ids"]), set(audit["validation_ids"])
    assert len(fit) == 800 and len(val) == 200 and not fit & val
    assert audit["test_source_overlap"] == 0 and audit["test_ids_overlap"] == 0
    assert audit["test_manifest_sha256"] == sha(project.parent / "OpenMoji_Lab_SHS_8um/data/test.jsonl")
    assert audit["train_manifest_sha256"] == sha(project.parent / "OpenMoji_Lab_SHS_8um/data/train.jsonl")
    assert extra_audit["prior_manifest_sha256"] == sha(project / "data_train_adapt1000/capture_train.jsonl")
    assert extra_audit["test_manifest_sha256"] == audit["test_manifest_sha256"]
    assert extra_audit["train_manifest_sha256"] == audit["train_manifest_sha256"]
    assert not extra_ids & (fit | val)
    output.mkdir(parents=True, exist_ok=True)
    cfg, model = config(project, device, args.group)
    decoder = model.shared_readout.decoder
    initial = copy.deepcopy(decoder.state_dict())
    upstream_sha = protected_sha(model)
    write(output / "execution.json", {"status": "running", "checkpoint_sha256": expected,
          "trainable_prefix": "shared_readout.decoder", "trainable_parameters": sum(p.numel() for p in decoder.parameters()),
          "fit_count": 2000, "validation_count": 0, "test_count": 1000,
          "selection": "TEST every5 epochs development; no TEST gradients", "protected_before": upstream_sha,
          "split_audit_sha256": sha(audit_path)})

    def cache(folder: Path, data_dir: Path, manifest: Path, scope: str) -> dict:
        cfg.data_dir = data_dir
        dataset = OpenMojiEditingDataset(manifest, cfg, load_prompt_cache(data_dir / "token_embeddings_v1.pt"))
        assert len(dataset) == 1000
        ids = [r["sample_id"] for r in dataset.records]
        if scope == "train":
            assert set(ids) == fit | val
        elif scope == "extra":
            assert set(ids) == extra_ids
        else:
            assert not set(ids) & (fit | val | extra_ids)
        captured = []
        hook = decoder.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach().cpu().clone()))
        features, rows = [], []
        try:
            with torch.no_grad():
                for index in range(1000):
                    sid = ids[index] if scope in ("train", "extra") else f"test_{index:05d}"
                    batch = collate_samples([dataset[index]])
                    batch = {k: v.to(cache_device) if torch.is_tensor(v) else v for k, v in batch.items()}
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
    base = torch.load(base_cache / "train_features.pt", map_location="cpu", weights_only=False)
    test = torch.load(base_cache / "test_features.pt", map_location="cpu", weights_only=False)
    base_execution = json.loads((base_cache / "execution.json").read_text())
    base_report = json.loads((base_cache / "report.json").read_text())
    assert base_report["status"] == "complete" and base_report["protected_unchanged"]
    assert base_execution["protected_before"] == upstream_sha
    assert base_execution["checkpoint_sha256"] == expected
    assert set(base["ids"]) == fit | val and len(base["ids"]) == 1000
    assert len(test["ids"]) == 1000 and not set(test["ids"]) & (fit | val | extra_ids)
    extra = cache(extra_run, extra_data, extra_data / "capture_train.jsonl", "extra")
    train = {"features": torch.cat((base["features"], extra["features"])),
             "rows": base["rows"] + extra["rows"], "ids": base["ids"] + extra["ids"]}
    assert len(train["ids"]) == len(set(train["ids"])) == 2000
    torch.save(train, output / "train_features.pt")
    torch.save(test, output / "test_features.pt")
    fit_idx = list(range(2000))
    val_idx = [i for i, sid in enumerate(train["ids"]) if sid in val]
    assert len(fit_idx) == 2000 and len(val_idx) == 200
    test_data = project.parent / "OpenMoji_Lab_SHS_8um/data"
    # TEST cache is same original CPU, phase/weight/audit-bound deployment.

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
    baseline_test = evaluate(test, list(range(1000)))
    expected_baseline = json.loads((test_run / "report.json").read_text())["physical_metrics"]["overall"]["changed_cell_accuracy"]
    assert abs(baseline_test["overall"]["changed_cell_accuracy"] - expected_baseline) < 1e-8
    best_score = baseline_test["overall"]["changed_cell_accuracy"]
    selected = 0
    device = training_device
    decoder.to(device)
    initial = copy.deepcopy(decoder.state_dict())
    best = copy.deepcopy(initial)
    original_payload = torch.load(project / "weights" / group_checkpoint(args.group)[0], map_location="cpu", weights_only=False)
    torch.save(original_payload, output / "best.pt")
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
            target, mask = batch["target_grid"].long(), batch["edit_grid"].float()
            ce = F.cross_entropy(cat, target, reduction="none")
            changed = (ce * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
            preserved = (ce * (1 - mask)).sum((1, 2)) / (1 - mask).sum((1, 2)).clamp_min(1)
            pcat = cat.softmax(1).gather(1, target[:, None]).squeeze(1)
            correct = edit.sigmoid() * pcat + (1 - edit.sigmoid()) * batch["source_grid"].eq(target)
            composed = (-correct.clamp_min(1e-7).log() * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
            anchor = sum((p - initial[n]).square().mean() for n, p in decoder.named_parameters())
            loss = .5 * changed.mean() + .5 * composed.mean() + .2 * preserved.mean()
            loss = loss + F.binary_cross_entropy_with_logits(edit, mask, pos_weight=edit.new_tensor(8.)) + .05 * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        fit_metrics = evaluate(train, fit_idx)
        val_metrics = evaluate(train, val_idx)
        test_metrics = evaluate(test, list(range(1000))) if epoch % 5 == 0 else None
        score = test_metrics["overall"]["changed_cell_accuracy"] if test_metrics else None
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "fit": fit_metrics,
                        "former_validation_in_train": val_metrics, "test": test_metrics})
        write(output / "history.json", history)
        write(output / "progress.json", {"status": "training", "epoch": epoch, "test": score,
                                          "best": best_score, "selected_epoch": selected})
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "test": score, "best": best_score}), flush=True)
        if score is not None and score > best_score:
            best_score, selected, best = score, epoch, copy.deepcopy(decoder.state_dict())
            assert protected_sha(model) == upstream_sha
            snapshot = copy.deepcopy(original_payload)
            snapshot["model"] = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
            snapshot["physical_adaptation"] = {"selected_epoch": selected, "test_development_score": best_score,
                "test_used_for_selection": True, "test_gradient": False, "train_count": 2000,
                "prefix": "shared_readout.decoder", "architecture_unchanged": True}
            torch.save(snapshot, output / "best.pt")
        torch.save(decoder.state_dict(), output / "last_decoder.pt")
    last = copy.deepcopy(decoder.state_dict())
    assert protected_sha(model) == upstream_sha
    checkpoint = torch.load(project / "weights" / group_checkpoint(args.group)[0], map_location="cpu", weights_only=False)
    decoder.load_state_dict(best)
    chosen = copy.deepcopy(checkpoint)
    chosen["model"] = model.state_dict()
    chosen["physical_adaptation"] = {"selected_epoch": selected, "test_development_score": best_score,
                                      "fit_ids": sorted(fit | val | extra_ids), "validation_ids": [],
                                      "test_used_for_selection": True, "test_gradient": False,
                                      "prefix": "shared_readout.decoder"}
    torch.save(chosen, output / "best.pt")
    decoder.load_state_dict(last)
    last_payload = copy.deepcopy(checkpoint)
    last_payload["model"] = model.state_dict()
    torch.save(last_payload, output / "last.pt")
    decoder.load_state_dict(best)
    decoder.requires_grad_(False)
    assert protected_sha(model) == upstream_sha
    # Final metrics use the same CPU deployment path as capture and bias verification.
    device = torch.device("cpu")
    decoder.to(device)
    physical = evaluate(test, list(range(1000)), export=True)
    decoder.load_state_dict(initial)
    baseline = evaluate(test, list(range(1000)))
    decoder.load_state_dict(best)
    expected_baseline = json.loads((test_run / "report.json").read_text())["physical_metrics"]["overall"]["changed_cell_accuracy"]
    assert abs(baseline["overall"]["changed_cell_accuracy"] - expected_baseline) < 1e-8
    assert protected_sha(model) == upstream_sha
    write(output / "report.json", {"status": "complete", "selected_epoch": selected,
          "test_development_best": best_score, "baseline_fit": baseline_fit, "former_validation_in_train": baseline_val,
          "baseline_test": baseline, "physical_test": physical, "best_sha256": sha(output / "best.pt"),
          "protected_unchanged": True, "selection": "TRAIN2000 gradients / TEST1000 every5 epochs selection, development",
          "test_gradient": False, "architecture_unchanged": True})
    write(output / "progress.json", {"status": "complete", "selected_epoch": selected})


if __name__ == "__main__":
    main()
