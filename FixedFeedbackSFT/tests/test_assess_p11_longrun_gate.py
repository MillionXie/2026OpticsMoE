from __future__ import annotations

from FixedFeedbackSFT.tools.assess_p11_longrun_gate import assess


def rows(values: list[float]) -> list[dict]:
    return [
        {
            "epoch": index,
            "validation_raw": {"top1_accuracy": value},
            "validation_ema": {"top1_accuracy": value - 0.001},
        }
        for index, value in enumerate(values, 1)
    ]


def test_waits_before_first_gate() -> None:
    result = assess(rows([0.47 + 0.002 * index for index in range(9)]))
    assert result["decision"] == "wait_for_epoch_10"
    assert result["automatic_process_action"] == "none"


def test_flags_below_floor_without_stopping_process() -> None:
    result = assess(rows([0.47 + 0.001 * index for index in range(10)]))
    assert result["decision"] == "manual_review_stop_candidate"
    assert result["active_gate_epoch"] == 10
    assert result["automatic_process_action"] == "none"


def test_allows_positive_recovery_above_gate() -> None:
    result = assess(rows([0.47 + 0.0025 * index for index in range(10)]))
    assert result["latest_raw_top1"] >= 0.49
    assert result["recent_raw_slope_per_epoch"] > 0.0
    assert result["decision"] == "continue_to_next_gate"
