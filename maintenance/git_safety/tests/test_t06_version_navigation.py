"""Version-label clarity; not a model or scientific-performance evaluation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / 'LightGenV2/tasks/t06_video_quality_assessment'


def test_current_summary_binds_existing_fixed_model_reports():
    entry = (TASK / 'README.md').read_text(encoding='utf-8')
    authoritative = (TASK / 'CURRENT_VERSION_20261004.md').read_text(encoding='utf-8')
    current = entry.split('## 历史仿真结论')[0]
    for value in ('6710968960','5786364901','8043868643','7977138739'):
        assert value in authoritative
        assert value in current
    assert '95e12397' in current and '5303b574' in current
    assert '不是当前Temporal实拍模型' in current
    assert '预检未通过' in current


def test_original_historical_metrics_retained_without_current_version_label():
    entry = (TASK / 'README.md').read_text(encoding='utf-8')
    history = entry.split('## 历史仿真结论')[1].split('## 不可静默改变')[0]
    assert '当前主版本' not in history
    assert 'Spatial 的当前正式归档' not in history
    assert '历史 Temporal-36 对照版本' in history
    assert '历史 Spatial-4 对照归档' in history
    for value in ('0.6393','0.8454','0.8082','0.8044'):
        assert value in history
