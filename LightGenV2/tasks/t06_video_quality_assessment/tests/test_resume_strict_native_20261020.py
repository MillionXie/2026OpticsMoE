from unittest.mock import patch
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import resume_strict_native_20261020 as entry

def test_fixed_native_plan():
    assert entry.PLAN==[('deepseek','temporal',100,2),('kimi','mps',100,1),('kimi','mps',50,1),('kimi','temporal',100,6)]
    assert entry.LEASE_END-entry.DEADLINE==387

def test_deadline_only_explicit_authorization():
    with patch.object(entry.time,'time',return_value=entry.DEADLINE-600):
        entry.validate(entry.DEADLINE)
        entry.validate(entry.EXTENDED_DEADLINE)
        with pytest.raises(ValueError):entry.validate(entry.DEADLINE+1)
    with patch.object(entry.time,'time',return_value=entry.DEADLINE):
        with pytest.raises(ValueError):entry.validate(entry.DEADLINE)

def test_extended_admission_only_preserves_measurement_engine():
    from pathlib import Path
    text=Path(entry.__file__).read_text()
    assert "'--continuation',str(args.continuation)" in text
    assert 'old.worker(args)' in text
    assert "state.setdefault('preserved_complete',[])" in text
    assert entry.EXTENDED_LEASE_END-entry.EXTENDED_DEADLINE==420
