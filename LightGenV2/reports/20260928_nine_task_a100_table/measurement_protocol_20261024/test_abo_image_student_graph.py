"""Check pinned forward semantics without loading any neural model."""
import copy
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from abo_image_student_graph import inference_signature,BODY_SHA


def test_training_asset_overrides_not_renamed_reverse_graph():
    base=dict(qwen=dict(model_id='old',dtype='bfloat16'),training=dict(epochs=40,amp_enabled=True),
        optical=dict(distance=.1),batching=dict(inference_batch_size=4))
    changed=copy.deepcopy(base);changed['qwen']['model_id']='new';changed['training']['epochs']=30
    changed['training_optimization']={'schedule':'unused'}
    assert inference_signature(base)==inference_signature(changed)
    changed['training']['amp_enabled']=False
    assert inference_signature(base)!=inference_signature(changed)


@pytest.mark.parametrize('key,value',[('distance',.08),('expert_stages',4)])
def test_graph_changes_not_hidden_by_training_exception(key,value):
    base=dict(optical=dict(distance=.1));changed=copy.deepcopy(base);changed['optical'][key]=value
    assert inference_signature(base)!=inference_signature(changed)


def test_real_original_graph_no_teacher_or_training_calls():
    source=(HERE/'abo_image_student_graph.py').read_text()
    assert BODY_SHA=='a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'
    assert 'optical_moe as original' in source and 'from abo_student_graph import patch' in source
    assert 'original.encode_student_samples' in source and 'original._encode_titles_student' in source
    assert 'original.teacher_cache(' not in source and 'original.train(' not in source
    assert 'for index in (200,2399)' in source and 'for index in (0,99)' in source


def test_finite_predecessor_wait_does_not_repeat_GPU_or_CPU_proofs():
    from run_abo_image_student_contract import predecessor_ready
    assert predecessor_ready(dict(status='running')) is False
    assert predecessor_ready(dict(status='complete',completed=['qwen_repeat01','qwen_repeat02','qwen_repeat03'],mps_released=True))
    with pytest.raises(ValueError):predecessor_ready(dict(status='complete',completed=['qwen_repeat03'],mps_released=True))
    with pytest.raises(RuntimeError):predecessor_ready(dict(status='stopped'))
