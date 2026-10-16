"""CPU protocol contracts for diagnostic-only residual synchronization traces."""
import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import diagnose_residual_sync as m


def test_finite_cases_no_b1_and_no_measurement_replacement():
    c=m.config()
    assert c['cases']=={'qwen':[2,4],'clip':[4,8,10]}
    assert c['no_B1_execution_or_replacement'] and c['profile_is_diagnostic_not_utilization_or_cost']
    assert c['maximum_suite_seconds']==3600 and c['maximum_worker_seconds']==900


def test_native_paths_same_measurement_worker_unmodified():
    text=Path(m.__file__).read_text();tree=ast.parse(text)
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'backward','CUDAGraph','compile','load_state_dict','set_device'}.intersection(calls)
    assert 'execution.make_gpu_forward' in text and 'metadata.strict_pair' in text
    assert 'previous.trace_summary' in text and 'torch.equal' in text
    assert any(isinstance(n,ast.keyword) and n.arg=='no_automatic_retry' for n in ast.walk(tree))
    assert 'LOCK_EX|fcntl.LOCK_NB' in text
