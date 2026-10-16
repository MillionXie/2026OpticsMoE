import ast
from LightGenV2.tasks.t06_video_quality_assessment import measure_yolo_clip_recheck as m


def test_fixed_steps_no_qwen_no_large_scan():
    c=m.config()
    assert c['cases']=={'yolo':[3,4],'clip':[10,11]}
    assert [c[k] for k in ['warmup_s','measurement_s','repeats','queue_depth']]==[5,30,3,8]
    assert c['maximum_suite_seconds']==3600


def test_only_full_pooled_mean_controls_extra():
    assert m.needs_extra(97.999999)
    assert not m.needs_extra(98)
    assert not m.needs_extra(99)


def test_worker_reuses_native_without_model_changes():
    tree=ast.parse(open(m.__file__,encoding='utf-8').read())
    calls={ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)}
    assert {'costs.worker','uniform.worker','reproduction.cuda_guard'}<=calls
    assert not any(x.endswith(('set_device','set_num_threads','set_float32_matmul_precision')) for x in calls)
