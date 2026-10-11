"""Read-only counterfactual feedback audit at a retained P12 phase-only endpoint.

This does not train FFA to that endpoint or resume an old experiment.  The four
gradient rules are evaluated on the same retained weights and validation batch.
The pilot supplies a bounded validation subset and the verified P11 assets.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

from FFA import run_pilot as pilot
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.settings import (
    load_settings,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "bp_current", "fa_source_connector", "source_full_response", "random_full_response"
)
RETENTION = ROOT / "FixedFeedbackSFT/evidence/worktree_cleanup_20260904/retention_manifest.json"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _require_identity(value: dict, expected: dict, name: str) -> None:
    mismatch = {
        key: {"actual": value.get(key), "expected": target}
        for key, target in expected.items() if value.get(key) != target
    }
    if mismatch:
        raise RuntimeError(f"{name} identity mismatch: {mismatch}")


def validate_parent_checkpoint(checkpoint: Path, profile: dict) -> tuple[dict, dict]:
    """Verify exact archived bytes plus the completed phase-only campaign identity."""

    registry = _read_json(RETENTION)
    records = [
        record for record in registry.get("checkpoint_files", [])
        if record.get("name") == "p12_phase_lsp"
    ]
    if len(records) != 1:
        raise RuntimeError("Retention registry must identify exactly one p12_phase_lsp checkpoint")
    record = records[0]
    checkpoint_sha = pilot.p12.sha256_file(checkpoint)
    if checkpoint_sha != record.get("sha256"):
        raise RuntimeError("Requested endpoint does not match the retained phase-only LSP checkpoint SHA")
    if checkpoint.stat().st_size != record.get("size_bytes"):
        raise RuntimeError("Retained checkpoint size differs from its registry")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("P12 checkpoint must be a mapping")
    method = payload.get("method")
    if method not in {"bp", "fa_pretrained"}:
        raise RuntimeError(f"Endpoint must be a BP or source-connector phase-only run, got {method}")
    _require_identity(payload, {
        "format": pilot.p12.CHECKPOINT_FORMAT,
        "task": "lsp", "seed": profile["seed"], "run_epochs": 50,
    }, "Parent checkpoint")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or not 0 < epoch <= 50 or payload.get("best_epoch") != epoch:
        raise RuntimeError("Parent must carry a selected adaptation-best epoch in [1, 50]")
    groups = payload.get("optimizer", {}).get("param_groups", [])
    if [group.get("group_name") for group in groups] != ["phase", "head"]:
        raise RuntimeError("Parent optimizer does not prove the phase-and-head-only policy")
    # This archived terminal receipt distinguishes a best checkpoint at epoch
    # 45, for example, from an interrupted run configured to target 50 epochs.
    evidence = ROOT / "FixedFeedbackSFT/evidence/p12_phase_only_fa_50e"
    receipt_path = evidence / "results/lsp" / method / f"seed_{payload['seed']}" / "result.json"
    receipt = _read_json(receipt_path)
    _require_identity(receipt, {
        "format": pilot.p12.RESULT_FORMAT, "status": "complete",
        "task": "lsp", "method": method, "seed": payload["seed"],
        "epochs_completed_this_run": 50, "adaptation_epochs": 50,
        "best_epoch": epoch,
        "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
        "source_phase_sha256": payload.get("source_phase_sha256"),
        "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
        "config_digest": payload.get("config_digest"),
        "implementation_sha256": payload.get("implementation_sha256"),
        "common_start_sha256": payload.get("common_start_sha256"),
        "frozen_stem_state_sha256": payload.get("frozen_stem_state_sha256"),
    }, "Parent completed-run receipt")
    panel = receipt.get("phase_only_panel", {})
    _require_identity(panel, {
        "format": "p12-phase-only-adaptation-panel-v1",
        "adaptation_scope": "phase_and_head", "effective_scope": "phase_and_head",
        "frozen_electronic_backbone_mode": "eval", "task_head_gradient": "exact_bp",
    }, "Parent phase-only panel")
    return payload, {
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "retention_manifest_path": str(RETENTION),
        "retention_manifest_sha256": pilot.p12.sha256_file(RETENTION),
        "retention_record": record,
        "terminal_receipt_path": str(receipt_path),
        "terminal_receipt_sha256": pilot.p12.sha256_file(receipt_path),
        "task": "lsp", "method": method, "seed": payload["seed"],
        "checkpoint_best_epoch": epoch, "completed_adaptation_epochs": 50,
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "source_phase_sha256": payload["source_phase_sha256"],
        "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "config_digest": payload["config_digest"],
        "implementation_sha256": payload["implementation_sha256"],
        "common_start_sha256": payload["common_start_sha256"],
        "git_commit": payload.get("git_commit"),
    }


def load_verified_endpoint(model, payload: dict) -> dict:
    """Strict-load the old endpoint while preserving and proving its source identity."""

    source_phase = model.source_phases.detach().cpu().clone()
    source_phase_sha = _tensor_sha256(source_phase)
    _require_identity(payload, {
        "source_checkpoint_sha256": model.source_manifest["sha256"],
        "source_phase_sha256": source_phase_sha,
        "frozen_stem_state_sha256": pilot.p12.module_state_sha256(model.backbone.stem),
    }, "Parent/P11 assets")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise TypeError("Parent checkpoint has no model state mapping")
    signature = state.get("backbone.p11_separable_architecture_signature")
    if signature is None or not torch.equal(
        signature.detach().cpu(), torch.tensor([11, 1, 2, 4], dtype=torch.int64)
    ):
        raise RuntimeError("Parent checkpoint lacks the exact P11 architecture signature")
    stored_source = state.get("source_phases")
    if stored_source is None or not torch.equal(stored_source.detach().cpu(), source_phase):
        raise RuntimeError("Parent persistent source phases differ from the verified P11 source")
    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        raise RuntimeError("Parent model tensor keys differ from the strict P12 model")
    for name, tensor in state.items():
        expected = expected_state[name]
        if not torch.is_tensor(tensor) or tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise RuntimeError(f"Parent model tensor shape/dtype mismatch: {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Nonfinite tensor in parent model: {name}")
    electronic_source_sha = pilot.electronic_hash(model)
    model.load_state_dict(state, strict=True)
    electronic_endpoint_sha = pilot.electronic_hash(model)
    if electronic_source_sha != electronic_endpoint_sha:
        raise RuntimeError("Old endpoint electronics differ from P11 source; this is not a frozen-body endpoint")
    if _tensor_sha256(model.source_phases) != source_phase_sha:
        raise RuntimeError("Persistent source phases changed while loading endpoint")
    return {
        "source_electronic_sha256": electronic_source_sha,
        "endpoint_electronic_sha256": electronic_endpoint_sha,
        "source_and_endpoint_electronics_identical": True,
        "source_phase_sha256": source_phase_sha,
        "endpoint_state_sha256": pilot.p12.module_state_sha256(model),
        "strict_state_load": True,
    }


def _check_gpu(gpu_uuid: str) -> dict:
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("--gpu must name the authorized physical GPU UUID")
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    candidates = [
        [item.strip() for item in row]
        for row in csv.reader(io.StringIO(rows)) if row and row[0].strip() == gpu_uuid
    ]
    if len(candidates) != 1:
        raise RuntimeError("Requested GPU UUID is absent or ambiguous")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader",
    ], text=True)
    occupied = [
        row for row in csv.reader(io.StringIO(apps))
        if row and row[0].strip() == gpu_uuid
    ]
    if occupied:
        raise RuntimeError(f"Requested GPU has another compute application: {occupied}")
    row = candidates[0]
    return {"gpu_uuid": row[0], "gpu_name": row[1],
            "memory_used_mib_before": row[2], "memory_total_mib": row[3],
            "compute_app_inventory_before": apps}


def _small_batch(raw: dict, sample_count: int) -> dict:
    return {
        key: value[:sample_count] if (
            (torch.is_tensor(value) and value.ndim > 0)
            or isinstance(value, (list, tuple))
        ) else value
        for key, value in raw.items()
    }


def gauge_aligned_phase_drift(model) -> tuple[dict, list[dict]]:
    """Remove each independent optical line's unobservable circular offset.

    Token propagation has one line per bank/feature column; channel propagation
    has one line per bank/token row, including the padded rows.  The circular
    resultant records when a line's mean offset is poorly defined.
    """

    current = model.phase_snapshot().double()
    source = model.source_phases.detach().cpu().double()
    per_stage = []
    all_absolute = all_square = 0.0
    count = 0
    for index, stage in enumerate(model.backbone.stages):
        dimension = -2 if stage.optical_axis == "token" else -1
        displacement = current[index] - source[index]
        mean_sin = displacement.sin().mean(dim=dimension, keepdim=True)
        mean_cos = displacement.cos().mean(dim=dimension, keepdim=True)
        offset = torch.atan2(mean_sin, mean_cos)
        resultant = torch.sqrt(mean_sin.square() + mean_cos.square())
        aligned = torch.atan2(
            torch.sin(displacement - offset), torch.cos(displacement - offset)
        )
        per_line_mae = aligned.abs().mean(dim=dimension)
        per_line_rms = aligned.square().mean(dim=dimension).sqrt()
        per_stage.append({
            "stage": index + 1, "axis": stage.optical_axis,
            "mean_absolute_rad": float(aligned.abs().mean()),
            "rms_rad": float(aligned.square().mean().sqrt()),
            "fraction_over_0p1_rad": float((aligned.abs() > 0.1).double().mean()),
            "circular_resultant_min": float(resultant.min()),
            "fraction_ambiguous_lines_resultant_under_1e_minus_6": float((resultant < 1e-6).double().mean()),
            "per_bank_line_mean_absolute_rad": per_line_mae.tolist(),
            "per_bank_line_rms_rad": per_line_rms.tolist(),
            "per_bank_line_removed_circular_offset_rad": offset.squeeze(dimension).tolist(),
            "per_bank_line_circular_resultant": resultant.squeeze(dimension).tolist(),
        })
        all_absolute += float(aligned.abs().sum())
        all_square += float(aligned.square().sum())
        count += aligned.numel()
    summary = {
        "definition": "wrap(current-source minus each independent line's circular-mean offset)",
        "mean_absolute_rad": all_absolute / count,
        "rms_rad": (all_square / count) ** 0.5,
        "per_stage_mean_absolute_rad": [row["mean_absolute_rad"] for row in per_stage],
        "per_stage_rms_rad": [row["rms_rad"] for row in per_stage],
        "line_records_file": "phase_drift.json",
    }
    return summary, per_stage


def _aggregate(rows: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        values = [row["methods"][method] for row in rows]
        stage_count = len(values[0]["per_stage"])
        result[method] = {
            "phase_cosine_mean": sum(v["phase"]["cosine"] for v in values) / len(values),
            "phase_cosine_min": min(v["phase"]["cosine"] for v in values),
            "phase_relative_l2_mean": sum(v["phase"]["relative_l2"] for v in values) / len(values),
            "per_stage_cosine_mean": [
                sum(v["per_stage"][j]["cosine"] for v in values) / len(values)
                for j in range(stage_count)
            ],
            "per_stage_cosine_min": [
                min(v["per_stage"][j]["cosine"] for v in values)
                for j in range(stage_count)
            ],
            "forward_max_abs": max(v["forward_max_abs"] for v in values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing FFA pilot identity/data contract")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Read-only retained P12 phase-only LSP best")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Authorized physical GPU UUID; must have no compute application")
    parser.add_argument("--samples-per-batch", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    run, checkpoint, out = args.run.resolve(), args.checkpoint.resolve(), args.out.resolve()
    runs_root = (ROOT / "FFA/runs").resolve()
    if runs_root not in run.parents or not run.is_dir():
        raise ValueError("--run must be an existing pilot directory inside FFA/runs")
    if runs_root not in out.parents or out == run or run in out.parents:
        raise ValueError("Use a separate fresh audit directory inside FFA/runs")
    if out.exists():
        raise FileExistsError("Audit output already exists; no overwrite/resume")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    identity_path = run / "identity.json"
    pilot_identity = _read_json(identity_path)
    _require_identity(pilot_identity, {"format": "ffa-intensity-response-pilot-v1"}, "Pilot")
    profile = pilot_identity["profile"]
    if profile.get("task") != "lsp" or profile.get("seed") != 2026:
        raise RuntimeError("This retained-endpoint audit requires the LSP seed-2026 pilot contract")
    payload, parent_identity = validate_parent_checkpoint(checkpoint, profile)
    gpu = _check_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    out.mkdir(parents=True)
    started = time.time()
    try:
        source_reconstruction_threads = torch.get_num_threads()
        pilot.p12.seed_everything(profile["seed"])
        device = torch.device("cuda:0")
        base = ROOT / "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml"
        settings = load_settings(
            base, task="lsp", method="bp", seed=profile["seed"], output_dir=out / "data",
            limits={"max_train_samples": profile["train_samples"],
                    "max_validation_samples": profile["validation_samples"], "max_test_samples": 1},
        )
        settings = replace(settings, task_settings=replace(
            settings.task_settings, train_batch_size=profile["train_batch_size"],
            evaluation_batch_size=profile["evaluation_batch_size"], num_workers=0,
        ))
        contract_path = run / "resolved_p12_contract.json"
        original_contract = _read_json(contract_path)
        reconstructed = settings.to_dict()
        _require_identity(original_contract, {
            "task": "lsp", "seed": profile["seed"],
            "model": reconstructed["model"], "limits": reconstructed["limits"],
        }, "Pilot architecture/data contract")
        task_original = dict(original_contract["task_settings"])
        task_current = dict(reconstructed["task_settings"])
        task_original.pop("num_workers", None)
        task_current.pop("num_workers", None)
        if task_original != task_current:
            raise RuntimeError("Audit task contract differs from the pilot (except worker count)")
        for relative, expected_sha in pilot_identity["implementation_files"].items():
            if not relative.startswith("FFA/") and pilot.p12.sha256_file(ROOT / relative) != expected_sha:
                raise RuntimeError(f"Shared P12 implementation changed since the pilot: {relative}")
        # Recreate the same architecture and source before loading the adapted
        # endpoint; retaining source_phases is essential to this counterfactual.
        model = pilot.p12.build_model(settings)
        endpoint_proof = load_verified_endpoint(model, payload)
        # Preserve the original CPU phase-snapshot arithmetic during identity
        # reconstruction. Float32 sigmoid can differ by an ULP when thread
        # partitioning changes; the archived reference must match byte-for-byte.
        # Only after strict source verification do we bound CPU worker threads.
        torch.set_num_threads(4)
        del payload
        model = model.to(device)
        pilot.policy(model, True)
        model.eval()
        _, bundle, loaders = pilot.p12.build_data(settings)
        del loaders["train"], loaders["test"]
        asset_path = run / "asset_identity.json"
        pilot_assets = _read_json(asset_path)
        if bundle.manifest_sha256 != pilot_assets.get("dataset_manifest_sha256"):
            raise RuntimeError("Reconstructed bounded validation subset differs from the pilot asset identity")
        if model.source_manifest["sha256"] != pilot_assets.get("source", {}).get("sha256"):
            raise RuntimeError("Audit source backbone differs from the pilot source")
        audit_identity = {
            "format": "ffa-retained-endpoint-counterfactual-audit-v1",
            "scope": "read-only gradient rules at an old phase-only endpoint; FFA has not trained to this endpoint",
            "parent": parent_identity, "endpoint_load": endpoint_proof,
            "pilot_run": str(run), "pilot_identity_sha256": pilot.p12.sha256_file(identity_path),
            "pilot_asset_identity_sha256": pilot.p12.sha256_file(asset_path),
            "pilot_resolved_contract_sha256": pilot.p12.sha256_file(contract_path),
            "shared_p12_implementation_matches_pilot": True,
            "profile": profile, "validation_batches": 3,
            "samples_per_batch": args.samples_per_batch,
            "audit_dataset_manifest_sha256": bundle.manifest_sha256,
            "old_endpoint_dataset_manifest_sha256": parent_identity["dataset_manifest_sha256"],
            "different_full_vs_pilot_subset_manifest_expected": True,
            "source": model.source_manifest, "gpu": gpu,
            "source_reconstruction_cpu_threads": source_reconstruction_threads,
            "audit_cpu_threads": torch.get_num_threads(),
            "git_commit": pilot.p12.git_commit(ROOT),
            "implementation_files": pilot.source_manifest(),
            "test_evaluated": False, "optimizer_steps": 0,
            "hardware_experiment": False, "started_unix": started,
        }
        pilot.write_json(out / "identity.json", audit_identity)
        pilot.write_json(out / "resolved_p12_contract.json", settings.to_dict())
        phase_drift = model.phase_report()
        phase_drift["current_physical_phase_sha256"] = _tensor_sha256(model.phase_snapshot())
        phase_drift["source_physical_phase_sha256"] = _tensor_sha256(model.source_phases)
        gauge_summary, gauge_lines = gauge_aligned_phase_drift(model)
        phase_drift["per_line_gauge_aligned"] = gauge_summary
        pilot.write_json(out / "phase_drift.json", {
            **phase_drift, "per_line_gauge_aligned_records": gauge_lines,
        })
        rows = []
        for batch_index, raw in enumerate(loaders["val"]):
            if batch_index == 3:
                break
            selected = _small_batch(raw, args.samples_per_batch)
            if selected["image"].shape[0] != args.samples_per_batch:
                raise RuntimeError("Validation loader does not contain three full diagnostic batches")
            batch = pilot.p12._move_batch(selected, device)
            row = {"batch_index": batch_index, "samples": args.samples_per_batch,
                   "sample_identity": {k: v for k, v in selected.items() if isinstance(v, (list, tuple))},
                   "methods": {}}
            for method in METHODS:
                row["methods"][method] = pilot.diagnose(
                    model, batch, settings, profile["seed"], method,
                    include_legacy=method == "bp_current",
                )
                row["methods"][method]["feedback_identity"] = pilot.feedback_identity(
                    model, method, profile["seed"]
                )
                print(f"audit batch={batch_index} method={method} cosine={row['methods'][method]['phase']['cosine']:.6f}", flush=True)
            rows.append(row)
            pilot.write_json(out / "batch_diagnostics.json", rows)
        if len(rows) != 3:
            raise RuntimeError("Fewer than three validation batches were available")
        unchanged = pilot.p12.module_state_sha256(model) == endpoint_proof["endpoint_state_sha256"]
        if not unchanged:
            raise RuntimeError("Endpoint tensors changed during the read-only gradient audit")
        if pilot.p12.sha256_file(checkpoint) != parent_identity["checkpoint_sha256"]:
            raise RuntimeError("Parent checkpoint bytes changed during the audit")
        report = {
            "format": audit_identity["format"], "status": "complete",
            "scope": audit_identity["scope"], "parent_method": parent_identity["method"],
            "parent_checkpoint_best_epoch": parent_identity["checkpoint_best_epoch"],
            "parent_completed_adaptation_epochs": 50, "phase_drift": phase_drift,
            "gradient_comparison": _aggregate(rows), "validation_batches": 3,
            "diagnostic_samples": 3 * args.samples_per_batch,
            "test_evaluated": False, "optimizer_steps": 0,
            "endpoint_tensors_unchanged": unchanged, "parent_checkpoint_unchanged": True,
            "source_phases_preserved": _tensor_sha256(model.source_phases) == endpoint_proof["source_phase_sha256"],
            "seconds": time.time() - started,
            "parent_checkpoint_sha256": parent_identity["checkpoint_sha256"],
            "source_checkpoint_sha256": parent_identity["source_checkpoint_sha256"],
            "audit_dataset_manifest_sha256": bundle.manifest_sha256,
            "git_commit": audit_identity["git_commit"], "gpu_uuid": args.gpu,
        }
        pilot.write_json(out / "result.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception:
        pilot.write_json(out / "failure.json", {
            "status": "failed", "traceback": traceback.format_exc(),
            "seconds": time.time() - started,
        })
        raise
    finally:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
