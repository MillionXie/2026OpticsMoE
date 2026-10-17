from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import rental_qwen_device_control as q

def test_original_identity_and_no_anchor():
    assert q.HEAD_SHA.startswith("e102adc6") and len(q.REVISION)==40
    text=Path(q.__file__).read_text()
    assert 'original.load_runtime(config)' in text
    assert 'telemetry.workload_cost' in text and 'historical_anchor_not_substituted=True' in text
    assert '.backward(' not in text and 'check(target,spec)' in text
    assert 'range(3)' in text and 'status_code==429' in text

def test_separate_qwen_utilization_and_unchanged_costs():
    text=Path(q.__file__).read_text()
    assert 'native_batch01_resident' in text and 'native_batch01_cost' in text
    assert 'phase("warmup",5)' in text and 'phase("measured",30)' in text
    assert 'range(8)' in text and 'range(1,4)' in text
    assert 'costs_not_derived_from_resident_power=True' in text
