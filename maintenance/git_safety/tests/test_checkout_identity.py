import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_checkout_identity import classify


def test_eol_only_is_not_scientific_edit():
    assert classify(b"x\r\n", b"x\n", b"y\n") == "head_identical_after_eol_normalization"


def test_published_change_is_not_lost():
    assert classify(b"new\r\n", b"old\n", b"new\n") == "already_matches_published_main"


def test_unique_change_and_missing_main_stay_protected():
    assert classify(b"new", b"old", None) == "unique_working_change_preserve"
    assert classify(b"new", b"old", b"other") == "unique_working_change_preserve"
