import pytest

from LightGenV2.tasks.t24_continuous_spatial_routing.stability import (
    assess_cohort, assess_run, load_protocol, run_names,
)


def history(accuracies):
    return [dict(epoch=i, validation=dict(accuracy=p, cross_entropy=1.0))
            for i, p in enumerate(accuracies, 1)]


def test_common_budget_requires_all_arms_and_keeps_nonplateau_at_cap():
    cfg = load_protocol()
    plateau = history([.68] * 200)
    runs = {key: plateau for key in run_names(cfg)}
    assert assess_cohort(runs, epoch=99)["next_epoch"] == 100
    assert assess_cohort(runs, epoch=100)["decision"] == "stop_plateau"
    assert assess_cohort(runs, epoch=101)["next_epoch"] == 120
    key = run_names(cfg)[0]
    runs[key] = history([.68] * 190 + [.72] * 10)
    cap = assess_cohort(runs, epoch=200)
    assert cap["decision"] == "stop_cap" and cap["runs_without_plateau"] == [key]
    with pytest.raises(ValueError, match="all 12"):
        assess_cohort({key: plateau}, epoch=200)


def test_oscillatory_tail_and_prior_vs_current_gain_prevent_plateau():
    series = history([.64] * 80 + ([.64] * 5 + [.72] * 5) * 2)
    run = assess_run(series)
    assert run["tail_delta_pp"] == pytest.approx(0)
    assert run["recent_smoothed_range_pp"] == pytest.approx(8)
    assert not run["plateau"]
    jump = assess_run(history([.65] * 80 + [.68] * 20))
    assert jump["best5_gain_pp"] == pytest.approx(3)
    assert not jump["plateau"]


def test_inclusive_thresholds_and_no_test_data_dependency():
    rows = history([.64] * 80 + [.64] * 10 + [.65] * 10)
    result = assess_run(rows)
    assert result["tail_delta_pp"] == pytest.approx(1) and result["plateau"]
    for row in rows:
        row["test"] = dict(accuracy=-999)
    assert assess_run(rows) == result
    assert not assess_run(rows, epoch=30)["eligible"]
    with pytest.raises(ValueError, match="consecutive"):
        assess_run(rows[1:])
