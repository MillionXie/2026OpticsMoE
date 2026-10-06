"""Pure protocol checks without loading Qwen, datasets, CUDA or task modules."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "LightGenV2/tasks/t03_saliency/aligned_baseline.py"


def helpers(staged=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"_resolve_baseline_protocol", "_baseline_epoch_stage", "_baseline_should_evaluate"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"math": math, "staged_epoch": staged}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), env)
    return env


def test_default_keeps_config_interval_and_first_epoch():
    h = helpers()
    p = h["_resolve_baseline_protocol"](SimpleNamespace(test_interval_epochs=5))
    assert p["staged_schedule"] and p["fixed_learning_rate"] is None
    assert [e for e in range(1, 13) if h["_baseline_should_evaluate"](e, 12, p)] == [1, 5, 10, 12]


def test_explicit_historical_interval_excludes_first_epoch():
    h = helpers()
    p = h["_resolve_baseline_protocol"](SimpleNamespace(test_interval_epochs=5), 1e-4, 10)
    assert not p["staged_schedule"]
    assert [e for e in range(1, 13) if h["_baseline_should_evaluate"](e, 12, p)] == [10, 12]


@pytest.mark.parametrize("lr", [0, -1, float("nan"), float("inf")])
def test_invalid_learning_rate_rejected(lr):
    with pytest.raises(ValueError):
        helpers()["_resolve_baseline_protocol"](SimpleNamespace(test_interval_epochs=5), lr)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_interval_rejected(interval):
    with pytest.raises(ValueError):
        helpers()["_resolve_baseline_protocol"](SimpleNamespace(test_interval_epochs=5), None, interval)


def test_fixed_lr_does_not_call_staged_schedule():
    def fail(*args):
        raise AssertionError("must not overwrite fixed LR")
    h = helpers(fail)
    p = h["_resolve_baseline_protocol"](SimpleNamespace(test_interval_epochs=5), 1e-4, 10)
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}, {"lr": 0.2}])
    stage = h["_baseline_epoch_stage"](optimizer, None, 11, p)
    assert stage["stage"] == "fixed_low_lr"
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, 1e-4]


def test_default_calls_original_schedule():
    calls = []
    def staged(*args):
        calls.append(args)
        return {"original": True}
    h = helpers(staged)
    settings = SimpleNamespace(test_interval_epochs=5)
    p = h["_resolve_baseline_protocol"](settings)
    assert h["_baseline_epoch_stage"]("optimizer", settings, 3, p) == {"original": True}
    assert calls == [("optimizer", settings, 3)]
