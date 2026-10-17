from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import rental_qwen_device_control as q

def test_original_identity_and_no_anchor():
    assert q.HEAD_SHA.startswith("e102adc6") and len(q.REVISION)==40
    text=Path(q.__file__).read_text()
    assert 'original.load_runtime(config)' in text
    assert 'telemetry.workload_cost' in text and 'historical_anchor_not_substituted=True' in text
    assert '.backward(' not in text and 'check(target,spec)' in text
    assert 'range(3)' in text and 'status_code==429' in text
