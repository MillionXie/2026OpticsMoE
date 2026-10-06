import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "LightGenV2/scripts/check_environment.py"
spec = importlib.util.spec_from_file_location("environment_gate", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def valid():
    return {"backend_config_present": True, "canonical_checkpoint_present": True,
            "canonical_checkpoint_hash_matches": True, "backend_integrity": {"model.py": {"matches": True}},
            "resolved_inputs": {k: {"path": "/asset", "present": True} for k in (
                "dataset_root", "manifest", "vision_cache", "language_cache")}}


def test_checked_scope_passes():
    assert module.assess_t06_report(valid())["passed"]


@pytest.mark.parametrize("key", ["backend_config_present", "canonical_checkpoint_present", "canonical_checkpoint_hash_matches"])
def test_missing_or_unverified_identity_fails(key):
    report = valid()
    del report[key]
    assert key in module.assess_t06_report(report)["failures"]


def test_hash_mismatch_not_waived():
    report = valid()
    report["backend_integrity"]["model.py"]["matches"] = False
    assert "backend:model.py" in module.assess_t06_report(report)["failures"]


@pytest.mark.parametrize("key", ["dataset_root", "manifest", "vision_cache", "language_cache"])
def test_required_inputs_must_exist(key):
    report = valid()
    report["resolved_inputs"][key]["present"] = False
    assert "input:" + key in module.assess_t06_report(report)["failures"]


def test_optional_null_not_missing_but_configured_path_is():
    report = valid()
    report["resolved_inputs"]["training_soft_targets"] = {"path": None, "present": None}
    assert module.assess_t06_report(report)["passed"]
    report["resolved_inputs"]["training_soft_targets"] = {"path": "/missing", "present": False}
    assert not module.assess_t06_report(report)["passed"]


def test_inspection_exception_cannot_be_success():
    assert not module.assess_t06_report({"error": "missing profile"})["passed"]
