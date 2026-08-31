from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from ..gpu_engineering_sweep import (
    CLAIM_SCOPE,
    RESULT_FORMAT,
    atomic_write_json,
    parse_args,
    parse_depths,
    summarize_phase_gradients,
    validate_result_fields,
)


def test_cli_accepts_single_and_comma_depths_with_batch_one_default(
    tmp_path: Path,
) -> None:
    assert parse_depths("64") == (64,)
    assert parse_depths("16,32,64,100,64") == (16, 32, 64, 100)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_depths("8,64")

    args = parse_args(
        [
            "--stem-checkpoint",
            str(tmp_path / "stem.pt"),
            "--p11-checkpoint",
            str(tmp_path / "p11.pt"),
            "--output-directory",
            str(tmp_path / "output"),
            "--depths",
            "100",
        ]
    )
    assert args.depths == (100,)
    assert args.batch_size == 1
    assert args.activation_checkpointing is True
    assert args.alpha_epsilon > 0.0
    no_checkpoint_args = parse_args(
        [
            "--stem-checkpoint",
            str(tmp_path / "stem.pt"),
            "--p11-checkpoint",
            str(tmp_path / "p11.pt"),
            "--output-directory",
            str(tmp_path / "output"),
            "--no-activation-checkpointing",
        ]
    )
    assert no_checkpoint_args.activation_checkpointing is False


def test_cpu_gradient_summary_checks_every_added_phase() -> None:
    present = torch.nn.Parameter(torch.ones(2))
    zero = torch.nn.Parameter(torch.ones(2))
    missing = torch.nn.Parameter(torch.ones(2))
    present.grad = torch.tensor([3.0, 4.0])
    zero.grad = torch.zeros(2)
    report = summarize_phase_gradients(
        [("present", present), ("zero", zero), ("missing", missing)]
    )

    assert report["new_phase_count"] == 3
    assert report["gradient_present_count"] == 2
    assert report["gradient_finite_count"] == 2
    assert report["gradient_nonzero_count"] == 1
    assert report["every_gradient_present"] is False
    assert report["every_gradient_finite"] is False
    assert report["every_gradient_nonzero"] is False
    assert report["missing_gradient_names"] == ["missing"]
    assert report["zero_gradient_names"] == ["zero"]
    assert report["maximum_gradient_norm"] == pytest.approx(5.0)


def test_result_contract_and_atomic_json_are_cpu_only(tmp_path: Path) -> None:
    payload = {
        "format": RESULT_FORMAT,
        "status": "failed_oom",
        "claim_scope": CLAIM_SCOPE,
        "formal_training_started": False,
        "depth": 100,
        "configuration": {"batch_size": 1},
        "source": {
            "p11_checkpoint_sha256": "abc",
            "stem_checkpoint_sha256": "def",
        },
        "device": {"gpu_uuid": "GPU-unit-test"},
        "alpha": {"configured_epsilon": 0.01},
        "migration": {},
        "parameters": {},
        "checks": {},
        "measurement": {},
    }
    validate_result_fields(payload)
    destination = tmp_path / "depth_100" / "result.json"
    atomic_write_json(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert not list(destination.parent.glob("*.tmp"))
    with pytest.raises(ValueError, match="missing fields"):
        validate_result_fields({"format": RESULT_FORMAT})
