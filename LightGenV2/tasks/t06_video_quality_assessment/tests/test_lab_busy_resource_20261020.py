import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import lab_busy_resource_20261020 as m


def test_strict_tf_engine_and_fixed_mps_batch():
    text=Path(m.__file__).read_text();ast.parse(text)
    assert 'strict.measure(args,compute,inputs,first' in text
    assert "/strict.PROTOCOL" in text
    assert "Path(strict.__file__)" in text
    assert "conditions=[('unrestricted',1),('busy50',.5)]" in text
    assert "previous.CONTROL[args.model]" not in text
    assert "args.method=='mps' and args.batch!=1" in text
    assert "('mps',100),('mps',50),('temporal',100)" in text
    assert 'frozen.preflight_files(pre)' in text
    assert 'with torch.inference_mode():return compute(x),None' in text
    assert 'visible_sm_ratio_to100' in text


def test_cpu_idle_guard_and_no_global_mutation():
    text=Path(m.__file__).read_text()
    assert 'while stable<3:' in text and 'time.sleep(10)' in text
    assert 'if ROOT.exists():raise RuntimeError' in text
    assert 'no automatic retry' in text
    assert 'nvidia-smi -pl' not in text and 'nvidia-smi -mig' not in text
    assert "for precision in ('native','fp32')" in text


def test_exact_entry_and_fp32_policy():
    text=Path(m.__file__).read_text()
    assert "command[3]=__package__+'.lab_busy_resource_20261020'" in text
    assert "torch.set_float32_matmul_precision('highest')" in text
    assert 'allow_tf32=False' in text
    assert "args.method=='temporal' and args.batch!=expected" in text
