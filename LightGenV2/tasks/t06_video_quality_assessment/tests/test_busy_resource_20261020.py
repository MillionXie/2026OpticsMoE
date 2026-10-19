import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as r


def test_distinct_controls_and_b1():
    source=Path(r.__file__).read_text()
    assert "batch=1 if method=='mps'" in source
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap)" in source
    assert 'visible_sm_ratio_to100' in source
    assert 'first_tf_to_final_gpu_output=True' in source
    assert 'terminal_wait_excluded=True' in source
    assert 'accuracy_not_evaluated_in_repeated_cost_window=True' in source


def test_finite_delay_compensation():
    # G compute intervals but only G-1 internal waits; no final idle tail.
    for groups in (2,3,8,100):
        c=1.25;d=.5
        wait=c*(1/d-1)*groups/(groups-1)
        assert abs((groups*c)/(groups*c+(groups-1)*wait)-d)<1e-12


def test_no_global_power_or_mig_mutation():
    source=Path(r.__file__).read_text();ast.parse(source)
    assert 'nvidia-smi -pl' not in source and 'nvidia-smi -mig' not in source
    assert 'Same prepared first-B inputs' in source
    assert 'cost_and_utilization_same_window=True' in source


def test_resume_only_after_idle_handover_and_complete_receipt():
    source=Path(r.__file__).read_text()
    tree=ast.parse(source)
    suite=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='suite')
    text=ast.get_source_segment(source,suite)
    assert "previous['status']!='held_for_authorized_extension'" in text
    assert "receipt['status']!='complete'" in text
    assert text.index('if output.exists():')<text.index("if active and method!='mps'")


def test_fp32_accuracy_is_independent_and_never_substitutes_pure_cost():
    text=Path(r.__file__).with_name('rental_busy_accuracy_20261020.py').read_text()
    ast.parse(text)
    assert "args.precision='fp32';args.batch=1" in text
    assert 'base.evaluate(args,compute,prepare,transfer,first' in text
    assert 'costs_are_audit_not_complete_window_comparison=True' in text
    assert "torch.set_float32_matmul_precision('highest')" in text
    assert "test_videos=558" in text
