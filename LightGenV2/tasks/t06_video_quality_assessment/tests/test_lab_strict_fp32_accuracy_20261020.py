from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import lab_strict_fp32_accuracy_20261020 as m

def test_only_fixed_matching_fp32_accuracy_references():
    assert m.PLAN==[('clip',1),('clip',10),('yolo',1),('yolo',3),('qwen',1),('qwen',3),('glm',1)]
    text=Path(m.__file__).read_text()
    assert 'lab.fp32.runtime(args,guard,identity)' in text
    assert 'range(0,558,args.batch)' in text
    assert 'not_fresh_accuracy_at_each_resource_cap=True' in text
    assert 'model_hashes(args.model_path)!=assets' in text
    assert 'no_training=True' in text
    assert 'strict.measure' not in text and 'optimizer' not in text
