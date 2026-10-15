"""Finite candidate-selection contracts, no hardware or model access."""
from LightGenV2.tasks.t06_video_quality_assessment import run_resident_utilization_suite as suite


def test_historical_initials_and_bounded_budget():
    assert suite.INITIAL == dict(qwen=16, clip=10, yolo=4, deepseek=8)
    assert suite.MAX_NEW_GROUPS == 15
    assert suite.TARGET_PERCENT == 98 and suite.SATURATED_PERCENT == 99.5


def test_initial_saturation_adds_adjacent_lower_only():
    assert suite.next_batch("yolo", {}) == 4
    assert suite.next_batch("yolo", {4: 100}) == 3
    assert suite.next_batch("yolo", {4: 100, 3: 91}) is None


def test_near_full_stops_without_scanning():
    assert suite.next_batch("clip", {10: 98.7}) is None


def test_larger_then_adjacent_lower_if_saturated():
    assert suite.next_batch("deepseek", {8: 97.8}) == 12
    assert suite.next_batch("deepseek", {8: 97.8, 12: 100}) == 11
    assert suite.next_batch("deepseek", {8: 97.8, 12: 100, 11: 99}) is None


def test_under_target_at_budget_limit_retains_result():
    assert suite.next_batch("qwen", {16: 96.5, 24: 97.3}) is None
