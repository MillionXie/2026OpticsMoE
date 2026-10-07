"""Keep the historic report immutable while distinguishing the current entry."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_original_historical_report_is_unchanged():
    report = ROOT / 'LightGenV2/reports/20260923_t12_three_task_large_small/README.md'
    assert hashlib.sha256(report.read_bytes()).hexdigest() == (
        'ab615c3e7fc6804eed542058492a020463fcf3548fbdf7dff5b19ba363aa0565')


def test_current_entry_explains_historical_candidate_and_timing():
    task = ROOT / 'LightGenV2/tasks/t12_text_to_image'
    text = (task / 'README.md').read_text(encoding='utf8')
    target = '../../reports/20260923_t12_three_task_large_small/HISTORICAL_IDENTITY_20261004.md'
    assert target in text and (task / target).is_file()
    assert '仅指2026-09-27历史续训轮次' in text
    assert '不能将其旧模型测速套用到当前17.03M实拍版本' in text
