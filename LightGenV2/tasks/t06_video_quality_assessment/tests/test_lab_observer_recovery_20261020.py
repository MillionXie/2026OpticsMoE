"""The receiver change cannot silently change model calls or TF accounting."""
import ast
from pathlib import Path
import subprocess
from LightGenV2.tasks.t06_video_quality_assessment import lab_busy_resource_20261020 as lab
from LightGenV2.tasks.t06_video_quality_assessment import strict_tf_window_20261020 as strict


def test_only_two_missing_temporal_conditions_not_completed_mps():
    assert lab.OBSERVER_RECOVERY=='glm_obs01'
    assert lab.point_plan(lab.OBSERVER_RECOVERY)==[('fp32','glm','temporal',100)]


def test_first_block_hook_and_gpu_forward_loop_are_unchanged():
    path='LightGenV2/tasks/t06_video_quality_assessment/strict_tf_window_20261020.py'
    old=ast.parse(subprocess.check_output(['git','show','f79ec1bc3:'+path],text=True))
    current=ast.parse(Path(strict.__file__).read_text())
    def functions(tree):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(tree)
                if isinstance(node,ast.FunctionDef) and node.name in ('first_input','group','idle','warm','interval_energy','multi_gpu_energy')}
    assert functions(current)==functions(old)


def test_observer_backend_is_explicit_and_threshold_is_not_relaxed():
    text=Path(strict.__file__).read_text()
    assert 'ProcessTelemetrySampler' in text and 'telemetry_receiver' in text
    assert "if age>.25:raise ValueError('Power observation older than250ms')" in text
    assert 'args.isolated_telemetry=args.session==OBSERVER_RECOVERY' in Path(lab.__file__).read_text()
