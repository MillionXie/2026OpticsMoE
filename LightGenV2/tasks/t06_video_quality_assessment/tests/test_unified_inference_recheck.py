"""CPU-only approval/boundary checks; no model loading or GPU work."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import measure_unified_inference_recheck as m


def test_exact_points_no_qwen_b1_or_rental():
    points=m.points(m.config())
    assert len(points)==10
    assert points[:6]==[('clip',b,'native',r) for b in (1,10) for r in (1,2,3)]
    assert points[6:]==[('qwen',3,'paced',1),('glm',2,'paced',1),('clip',10,'paced',1),('yolo',3,'paced',1)]
    with pytest.raises(ValueError,match='Qwen B1'):
        m.require_point(SimpleNamespace(baseline='qwen',batch=1,kind='native',repeat=1),m.config())


def test_wait_not_added_to_formal_cost():
    assert m.excluded_wait_seconds(20,.5)==pytest.approx(.02)
    assert m.excluded_wait_seconds(20,1)==0
    rows=[[i,end-start,True,20.] for i,(start,end) in enumerate(m.costs.plan(10)) if end<=200]
    result=m.telemetry.workload_cost(rows,100)
    assert result['latency_16_videos_ms']==32
    assert result['energy_16_videos_j']==pytest.approx(438.2*.032)
    assert sum(m.excluded_wait_seconds(r[3],.5) for r in rows)==pytest.approx(.4)


def test_native_output_timing_retains_original_and_wait_is_after():
    text=Path(m.__file__).read_text();tree=ast.parse(text)
    runtime=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_runtime'))
    worker=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker'))
    assert 'frozen.load_runtime(args)' in runtime and 'native(inputs)' in runtime
    assert 'core.BoundaryTimer' in runtime and 'FirstBlockTimer' in runtime
    assert worker.index('values, wall, cuda = run(cpu, inputs)')<worker.index('requested = excluded_wait_seconds')
    assert "explicit_warmup_forwards=0" in worker
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'backward','CUDAGraph','compile','set_per_process_memory_fraction'}.intersection(calls)
    assert 'service_cost(' not in text


@pytest.mark.parametrize('baseline,batch,kind,repeat',[('clip',10,'native',3),('qwen',3,'paced',1),('glm',2,'paced',1)])
def test_commands_fresh_original_env_and_new_output(baseline,batch,kind,repeat):
    args=SimpleNamespace(manifest=Path('manifest.csv'),deadline=123)
    cmd=m.child_command(args,baseline,batch,kind,repeat,Path('out'))
    assert cmd[3].endswith('measure_unified_inference_recheck')
    assert cmd[cmd.index('--kind')+1]==kind and cmd[cmd.index('--repeat')+1]==str(repeat)
    assert '--telemetry' not in cmd


def test_sources_bind_implementation_and_protocol():
    values=m.source_identity()
    assert len(values)>15
    assert values['LightGenV2/tasks/t06_video_quality_assessment/measure_unified_inference_recheck.py']==m.frozen.sha256(Path(m.__file__))


def test_authorized_continuation_is_only_remaining_three_original_deadline():
    from LightGenV2.tasks.t06_video_quality_assessment import resume_unified_inference_recheck as r
    assert r.REMAINING==[('glm',2,'paced',1),('clip',10,'paced',1),('yolo',3,'paced',1)]
    assert len(r.EXPECTED)==7 and r.DEADLINE==1792312442.3183658
    text=Path(r.__file__).read_text(encoding='utf-8')
    assert 'original.child_command' in text and 'original.costs.wait_for_idle' in text
    assert 'protected_files(args.base,args.final_two)!=protected' in text
    assert "'resume03_20261018'" in text and "original16:34" in text
    assert 'REMAINING[1:] if args.final_two else REMAINING' in text
    assert "'resume02_20261018'" in text and 'Exactly8 complete points required' in text
