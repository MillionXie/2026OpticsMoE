from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa import (
    queue as p12queue,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa import (
    summarize as p12summary,
)


def test_job_matrix_builds_all_common_starts_but_only_selected_adaptations() -> None:
    jobs = p12queue.build_job_matrix(
        (2026, 2027, 2028), adaptation_seeds=(2026,)
    )
    assert len(jobs) == 18
    assert all(job.method == "noft" for job in jobs[:9])
    assert {(job.task, job.seed) for job in jobs[:9]} == {
        (task, seed)
        for task in ("caltech101", "isic2016", "lsp")
        for seed in (2026, 2027, 2028)
    }
    assert {job.seed for job in jobs[9:]} == {2026}
    assert {job.method for job in jobs[9:]} == {
        "bp",
        "fa_pretrained",
        "fa_random",
    }
    with pytest.raises(ValueError, match="subset"):
        p12queue.build_job_matrix((2026,), adaptation_seeds=(2027,))


def test_parse_int_list_rejects_empty_negative_and_duplicate_values() -> None:
    assert p12queue.parse_int_list("1, 2,5") == (1, 2, 5)
    for value in ("", "1,-2", "1,1", "gpu1"):
        with pytest.raises(Exception):
            p12queue.parse_int_list(value)


def test_gpu_inventory_uses_uuid_compute_apps_and_filters_exactly() -> None:
    outputs = iter(
        (
            "0, GPU-zero\n1, GPU-one\n2, GPU-two\n",
            "GPU-one, 9123, python, 7000\nGPU-other, 42, worker, 10\n",
        )
    )

    def runner(command, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return SimpleNamespace(stdout=next(outputs))

    devices, apps = p12queue.query_gpu_inventory(runner=runner)
    free = p12queue.free_requested_gpus(
        (0, 1, 2), devices, apps, reserved_indices=(2,)
    )
    assert [gpu.index for gpu in free] == [0]
    assert apps["GPU-one"][0]["pid"] == 9123
    with pytest.raises(RuntimeError, match="do not exist"):
        p12queue.free_requested_gpus((7,), devices, apps)


class _FakeSettings:
    def __init__(self, root: Path, method: str = "noft") -> None:
        self.output_dir = root / method
        self.task = "caltech101"
        self.method = method
        self.seed = 2026
        self.paths = SimpleNamespace(
            common_start_checkpoint=root / "common" / "common_start.pt",
            source_backbone=root / "source.pt",
            source_backbone_sha256="",
        )
        self.task_settings = SimpleNamespace(primary_metric="top1")

    def to_dict(self):  # type: ignore[no-untyped-def]
        return {
            "task": self.task,
            "method": self.method,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
        }


def test_only_strict_result_and_common_start_are_complete(tmp_path: Path) -> None:
    settings = _FakeSettings(tmp_path)
    settings.output_dir.mkdir(parents=True)
    # A last checkpoint or a status word alone must never satisfy the queue.
    (settings.output_dir / "result.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    complete, reason = p12queue.completion_reason(settings)  # type: ignore[arg-type]
    assert complete is False
    assert "mismatch" in reason

    common = settings.paths.common_start_checkpoint
    common.parent.mkdir(parents=True)
    common.write_bytes(b"complete common endpoint")
    common_hash = hashlib.sha256(common.read_bytes()).hexdigest()
    settings.paths.source_backbone.write_bytes(b"locked P11 source")
    settings.paths.source_backbone_sha256 = hashlib.sha256(
        settings.paths.source_backbone.read_bytes()
    ).hexdigest()
    implementation_digest = p12queue._implementation_sha256()
    manifest_dir = settings.output_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "caltech101_splits.json").write_text(
        json.dumps({"manifest_sha256": "b" * 64}), encoding="utf-8"
    )
    result = {
        "format": p12queue.RESULT_FORMAT,
        "status": "complete",
        "task": settings.task,
        "method": settings.method,
        "seed": settings.seed,
        "head_only_epochs": 50,
        "epochs_completed_this_run": 50,
        "inherited_pipeline_epochs": 50,
        "adaptation_epochs": 0,
        "config_digest": p12queue._json_digest(
            {
                "settings": settings.to_dict(),
                "implementation_sha256": implementation_digest,
            }
        ),
        "implementation_sha256": implementation_digest,
        "common_start_sha256": common_hash,
        "best_epoch": 50,
        "primary_metric": "top1",
        "test": {"normal": {"top1": 0.75}},
        "source_checkpoint_sha256": settings.paths.source_backbone_sha256,
        "dataset_manifest_sha256": "b" * 64,
    }
    (settings.output_dir / "result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    assert p12queue.completion_reason(settings) == (True, "complete")  # type: ignore[arg-type]

    common.write_bytes(b"corrupted")
    complete, reason = p12queue.completion_reason(settings)  # type: ignore[arg-type]
    assert complete is False
    assert "checksum" in reason


def test_paired_contrast_and_bp_recovery_use_identical_seeds() -> None:
    rows = [
        {"task": "caltech101", "method": method, "seed": seed, "test_primary": value}
        for seed, values in {
            1: {"noft": 0.50, "bp": 0.70, "fa_pretrained": 0.65},
            2: {"noft": 0.55, "bp": 0.75, "fa_pretrained": 0.67},
            3: {"noft": 0.60, "bp": 0.80},  # deliberately unpaired FA seed
        }.items()
        for method, value in values.items()
    ]
    contrast = p12summary._paired_contrast(
        rows,
        task="caltech101",
        left_method="fa_pretrained",
        right_method="noft",
        bootstrap_samples=200,
    )
    assert contrast["paired_seeds"] == [1, 2]
    assert contrast["mean"] == pytest.approx(0.135)
    assert contrast["bootstrap_ci95_low"] is not None
    recovery = p12summary._bp_recovery_summary(
        rows, task="caltech101", method="fa_pretrained"
    )
    assert recovery["paired_seeds"] == [1, 2]
    assert recovery["ratio_of_paired_mean_gains"] == pytest.approx(0.675)


def test_single_seed_summary_does_not_claim_variance_or_bootstrap_ci() -> None:
    assert p12summary._summary([0.4]) == {
        "n": 1,
        "mean": 0.4,
        "std": None,
        "min": 0.4,
        "max": 0.4,
    }
    assert p12summary._bootstrap_mean_ci(
        [0.4], samples=100, identity="single"
    ) == (None, None)
