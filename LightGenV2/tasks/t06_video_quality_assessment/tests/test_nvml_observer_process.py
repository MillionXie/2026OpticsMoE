"""CPU-only protocol checks. Synthetic rows never enter measurement runs."""
import sys
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler

FAKE='''import json,sys
print(json.dumps(dict(ready=dict(version="synthetic_cpu_test"))),flush=True)
for line in sys.stdin:
 c=json.loads(line)
 if c["op"]=="phase":
  s=dict(host_time_unix=1000+c["id"],host_monotonic_s=120+c["id"],phase=c["phase"],watts=100.,utilization_percent=50.,memory_used_mib=10.)
  print(json.dumps(dict(sample=s)),flush=True)
 print(json.dumps(dict(ack=c["id"])),flush=True)
 if c["op"]=="stop":break
'''


def test_original_observer_timestamps_are_not_retimed_on_delivery(tmp_path):
    s=ProcessTelemetrySampler('synthetic',tmp_path/'not_measured.jsonl',command=[sys.executable,'-u','-c',FAKE])
    try:
        s.start();s.set_phase('test1');s.set_phase('test2')
    finally:rows=s.stop()
    assert [r.host_monotonic_s for r in rows]==[121,122]
    assert [r.phase for r in rows]==['test1','test2']
    assert s.process.returncode==0 and not s.thread.is_alive()


def test_observer_errors_are_exposed_not_skipped(tmp_path):
    program='import json;print(json.dumps(dict(ready={})),flush=True);print(json.dumps(dict(error="bad NVML row")),flush=True)'
    s=ProcessTelemetrySampler('synthetic',tmp_path/'not_measured.jsonl',command=[sys.executable,'-u','-c',program])
    try:
        with pytest.raises(RuntimeError):
            s.start();s.thread.join(3);s.check()
    finally:s.stop(raise_on_error=False)


def test_same_query_receiver_no_cuda_and_no_interpolation():
    from LightGenV2.tasks.t06_video_quality_assessment import nvml_observer_process as module
    text=Path(module.__file__).read_text()
    assert 'sampler=TelemetrySampler(gpu,raw_path)' in text
    assert 'cuda_context_created=False' in text
    assert 'torch' not in text
    assert 'sample=asdict(sampler.samples[index])' in text
