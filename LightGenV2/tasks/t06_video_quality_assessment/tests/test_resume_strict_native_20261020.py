from unittest.mock import patch
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import resume_strict_native_20261020 as entry

def test_fixed_native_plan():
    assert entry.PLAN==[('deepseek','temporal',100,2),('kimi','mps',100,1),('kimi','mps',50,1),('kimi','temporal',100,6)]
    assert entry.LEASE_END-entry.DEADLINE==387

def test_deadline_only_explicit_authorization():
    with patch.object(entry.time,'time',return_value=entry.DEADLINE-600):
        entry.validate(entry.DEADLINE)
        with pytest.raises(ValueError):entry.validate(entry.DEADLINE+1)
    with patch.object(entry.time,'time',return_value=entry.DEADLINE):
        with pytest.raises(ValueError):entry.validate(entry.DEADLINE)
