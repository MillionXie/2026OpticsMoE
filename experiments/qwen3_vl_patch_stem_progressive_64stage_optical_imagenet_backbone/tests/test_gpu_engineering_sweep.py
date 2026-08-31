from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from ..gpu_engineering_sweep import (
    ALPHA_MODES,
    CLAIM_SCOPE,
    RESULT_FORMAT,
    _one_step,
    atomic_write_json,
    build_campaign_contract,
    canonical_json_sha256,
    combination_identity,
    effective_new_stage_alpha,
    implementation_manifest,
    load_matching_result,
    parse_args,
    parse_depths,
    parse_feedback_methods,
    result_relative_path,
    summarize_input_amplitude_gradient,
    summarize_phase_gradients,
    validate_result_fields,
)


class _TinyAuditStage(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.raw_phase = nn.Parameter(torch.tensor([value]))


class _TinyAuditSlot(nn.Module):
    def __init__(self, index: int, value: float) -> None:
        super().__init__()
        self.stage_index = index
        self.stage = _TinyAuditStage(value)


class _TinyAuditModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slots = nn.ModuleList(
            [_TinyAuditSlot(0, 0.2), _TinyAuditSlot(1, 0.3)]
        )

    def forward_field(self, amplitude: torch.Tensor):
        output = amplitude
        for slot in self.slots:
            output = output * (1.0 + slot.stage.raw_phase.mean())
        return output, ()


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
    assert args.alpha_mode == "epsilon_probe"
    assert args.alpha_epsilon > 0.0
    assert effective_new_stage_alpha(args) == pytest.approx(args.alpha_epsilon)
    assert args.feedback_methods == ("bp_current", "fa_source", "fa_random")
    assert args.feedback_random_seed == 20260901
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
    full_depth_args = parse_args(
        [
            "--stem-checkpoint",
            str(tmp_path / "stem.pt"),
            "--p11-checkpoint",
            str(tmp_path / "p11.pt"),
            "--output-directory",
            str(tmp_path / "full-depth"),
            "--alpha-mode",
            "full_depth",
        ]
    )
    assert tuple(ALPHA_MODES) == ("epsilon_probe", "full_depth")
    assert effective_new_stage_alpha(full_depth_args) == 1.0


def test_feedback_method_cli_and_combination_identity_are_unambiguous(
    tmp_path: Path,
) -> None:
    assert parse_feedback_methods("fa_source") == ("fa_source",)
    assert parse_feedback_methods(
        "bp_current,fa_source,fa_random,fa_source"
    ) == ("bp_current", "fa_source", "fa_random")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_feedback_methods("bp,fa_source")

    args = parse_args(
        [
            "--stem-checkpoint",
            str(tmp_path / "stem.pt"),
            "--p11-checkpoint",
            str(tmp_path / "p11.pt"),
            "--output-directory",
            str(tmp_path / "output"),
            "--depths",
            "64,100",
            "--feedback-methods",
            "fa_source,fa_random",
            "--feedback-random-seed",
            "73",
            "--resume-existing",
        ]
    )
    assert args.feedback_methods == ("fa_source", "fa_random")
    assert args.feedback_random_seed == 73
    assert args.resume_existing is True
    campaign = build_campaign_contract(
        args=args,
        source_sha256="p11-sha",
        stem_sha256="stem-sha",
        gpu_uuid="GPU-test",
        implementation_sha256="implementation-sha",
        torch_version="torch-test",
        torch_cuda_version="cuda-test",
    )
    campaign_sha = canonical_json_sha256(campaign)
    code_manifest = implementation_manifest()
    assert len(code_manifest["combined_sha256"]) == 64
    assert any(path.endswith("/model.py") for path in code_manifest["files"])
    source_identity = combination_identity(
        campaign_sha256=campaign_sha,
        depth=64,
        feedback_method="fa_source",
        feedback_random_seed=73,
    )
    random_identity = combination_identity(
        campaign_sha256=campaign_sha,
        depth=64,
        feedback_method="fa_random",
        feedback_random_seed=73,
    )
    assert source_identity["feedback_random_seed"] is None
    assert random_identity["feedback_random_seed"] == 73
    assert source_identity["combination_sha256"] != random_identity[
        "combination_sha256"
    ]
    assert result_relative_path(64, "fa_source") == Path(
        "depth_064/feedback_fa_source/result.json"
    )


def test_cpu_gradient_summary_checks_every_phase_and_input_amplitude() -> None:
    present = torch.nn.Parameter(torch.ones(2))
    zero = torch.nn.Parameter(torch.ones(2))
    missing = torch.nn.Parameter(torch.ones(2))
    present.grad = torch.tensor([3.0, 4.0])
    zero.grad = torch.zeros(2)
    report = summarize_phase_gradients(
        [("present", present), ("zero", zero), ("missing", missing)]
    )

    assert report["scope"] == "all_phase_parameters_carried_and_new"
    assert report["phase_parameter_count"] == 3
    assert report["gradient_present_count"] == 2
    assert report["gradient_finite_count"] == 2
    assert report["gradient_nonzero_count"] == 1
    assert report["every_gradient_present"] is False
    assert report["every_gradient_finite"] is False
    assert report["every_gradient_nonzero"] is False
    assert report["missing_gradient_names"] == ["missing"]
    assert report["zero_gradient_names"] == ["zero"]
    assert report["maximum_gradient_norm"] == pytest.approx(5.0)

    amplitude = torch.ones(1, 3, 2, 2, requires_grad=True)
    amplitude.square().sum().backward()
    amplitude_report = summarize_input_amplitude_gradient(amplitude)
    assert amplitude_report == {
        "name": "input_amplitude",
        "gradient_present": True,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "gradient_norm": pytest.approx(float(torch.full_like(amplitude, 2.0).norm())),
    }
    missing_amplitude = summarize_input_amplitude_gradient(
        torch.ones(1, requires_grad=True)
    )
    assert missing_amplitude["gradient_present"] is False


def test_one_step_audits_all_phases_and_input_amplitude() -> None:
    model = _TinyAuditModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    field = torch.linspace(0.1, 1.0, 12).reshape(1, 3, 2, 2)
    loss, phases, amplitude, stepped = _one_step(
        model=model,
        optimizer=optimizer,
        field_template=field,
        audit_gradients=True,
    )

    assert loss > 0.0
    assert stepped is True
    assert phases is not None
    assert phases["phase_parameter_count"] == 2
    assert phases["every_gradient_present"] is True
    assert phases["every_gradient_finite"] is True
    assert phases["every_gradient_nonzero"] is True
    assert amplitude is not None
    assert amplitude["gradient_present"] is True
    assert amplitude["gradient_finite"] is True
    assert amplitude["gradient_nonzero"] is True


def test_full_depth_feedback_command_locks_engineering_matrix() -> None:
    experiment = Path(__file__).resolve().parents[1]
    sweep_command = (
        experiment / "commands" / "03_gpu_engineering_sweep_bs1.sh"
    ).read_text(encoding="utf-8")
    command = (
        experiment / "commands" / "05_gpu_full_depth_feedback_cuda_audit_bs1.sh"
    ).read_text(encoding="utf-8")
    assert 'source "$(dirname "$0")/_training_common.sh"' in sweep_command
    assert (
        'CUDA_VISIBLE_DEVICES="$(visible_gpu_uuids "${P13_GPU}")"'
        in sweep_command
    )
    assert 'CUDA_VISIBLE_DEVICES="${P13_GPU}"' not in sweep_command
    assert 'DEPTHS="${DEPTHS:-64,100}"' in command
    assert (
        'FEEDBACK_METHODS="${FEEDBACK_METHODS:-bp_current,fa_source,fa_random}"'
        in command
    )
    assert 'BATCH_SIZE="${BATCH_SIZE:-1}"' in command
    assert 'if [[ "${ALPHA_MODE:-full_depth}" != "full_depth" ]]' in command
    assert 'ALPHA_MODE="full_depth"' in command
    assert "03_gpu_engineering_sweep_bs1.sh" in command
    assert "train.py" not in command


def test_result_contract_and_atomic_json_are_cpu_only(tmp_path: Path) -> None:
    identity = combination_identity(
        campaign_sha256="campaign",
        depth=100,
        feedback_method="fa_random",
        feedback_random_seed=91,
    )
    payload = {
        "format": RESULT_FORMAT,
        "status": "failed_oom",
        "claim_scope": CLAIM_SCOPE,
        "formal_training_started": False,
        "depth": 100,
        "combination": identity,
        "feedback": {
            "method": "fa_random",
            "random_seed": 91,
            "initial_manifest": {},
            "initial_manifest_sha256": None,
            "final_manifest": {},
            "final_manifest_sha256": None,
        },
        "configuration": {
            "batch_size": 1,
            "alpha_mode": "full_depth",
            "effective_new_stage_alpha": 1.0,
        },
        "source": {
            "p11_checkpoint_sha256": "abc",
            "stem_checkpoint_sha256": "def",
        },
        "device": {"gpu_uuid": "GPU-unit-test"},
        "alpha": {
            "mode": "full_depth",
            "configured_new_stage_alpha": 1.0,
            "configured_epsilon": None,
            "report": None,
            "all_stages_exactly_one": None,
            "interpretation": "full_depth_backward_connectivity_audit",
        },
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
    loaded = load_matching_result(
        destination,
        expected_combination_sha256=identity["combination_sha256"],
    )
    assert loaded["feedback"]["method"] == "fa_random"
    with pytest.raises(RuntimeError, match="another combination identity"):
        load_matching_result(
            destination,
            expected_combination_sha256="another-combination",
        )
    passed = json.loads(json.dumps(payload))
    passed["status"] = "passed_engineering"
    passed["alpha"]["report"] = {
        "all_full_depth": True,
        "minimum": 1.0,
        "maximum": 1.0,
    }
    passed["alpha"]["all_stages_exactly_one"] = True
    passed["feedback"].update(
        {
            "initial_manifest_sha256": "initial-manifest",
            "final_manifest_sha256": "final-manifest",
            "method_unchanged_during_run": True,
        }
    )
    passed["migration"] = {"source_checkpoint_sha256": "abc"}
    passed["parameters"] = {"optical_phase_parameters": 15_052_800}
    passed["checks"] = {
        "loss_finite_every_step": True,
        "every_phase_gradient_present": True,
        "every_phase_gradient_finite": True,
        "every_phase_gradient_nonzero": True,
        "input_amplitude_gradient_present": True,
        "input_amplitude_gradient_finite": True,
        "input_amplitude_gradient_nonzero": True,
    }
    passed["measurement"] = {
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 2,
        "samples_per_second": 3.0,
    }
    validate_result_fields(passed)
    mismatched_alpha = json.loads(json.dumps(passed))
    mismatched_alpha["configuration"]["effective_new_stage_alpha"] = 0.01
    with pytest.raises(ValueError, match="alpha result disagrees"):
        validate_result_fields(mismatched_alpha)
    passed["feedback"]["method"] = "fa_source"
    with pytest.raises(ValueError, match="feedback method"):
        validate_result_fields(passed)
    with pytest.raises(ValueError, match="missing fields"):
        validate_result_fields({"format": RESULT_FORMAT})
