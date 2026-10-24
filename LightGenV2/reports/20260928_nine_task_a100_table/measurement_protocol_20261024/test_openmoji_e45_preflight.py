"""Pure CPU static/admission tests; never load a real model or CUDA."""
import ast
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import prepare_openmoji_e45 as prepared
from run_openmoji_e45_preflight import predecessor_action


def test_uses_amended_laboratory_weight_not_old_fullhead():
    r=prepared.row_identity()
    assert r['ours_target']==.8765 and r['ours_lab_target']==.8655
    assert r['ours_pt_sha256'].startswith('03cb861c') and r['ours_epoch']==45


def test_predecessor_must_be_complete_and_released():
    for status in ('waiting_foreign_process','running','idle_guard'):
        assert predecessor_action({'status':status})=='wait'
    for status in ('stopped','failed'):
        with pytest.raises(RuntimeError):predecessor_action({'status':status})
    with pytest.raises(ValueError):predecessor_action({'status':'complete','completed':[]})
    assert predecessor_action({'status':'complete','completed':['ours_repeat01','ours_repeat02','ours_repeat03'],'mps_released':True})=='admit'


def test_online_forward_not_cached_feature_and_no_target_model_input():
    text=(HERE/'prepare_openmoji_e45.py').read_text(encoding='utf8')
    assert "tokenizer.encode(record['instruction'],add_special_tokens=False)" in text
    assert 'nn.Embedding.from_pretrained(weight,freeze=True)' in text
    assert 'actual=model(image_tensor[None],[lookup])' in text
    assert "expected=model(original_input['source_image'][None],[original_input['prompt_hidden']])" in text
    assert "for index in (200,999)" in text
    assert "model.load_state_dict(payload['model'],strict=True)" in text
    assert "payload['epoch']!=45" in text
    tree=ast.parse(text)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model':
            assert all('target' not in ast.unparse(a) for a in n.args)


def test_no_cuda_mps_or_training_in_preflight_controller():
    text=(HERE/'run_openmoji_e45_preflight.py').read_text(encoding='utf8')
    assert 'import torch' not in text and 'engine.daemon(' not in text
    assert "CUDA_VISIBLE_DEVICES=''" in text and 'fcntl.LOCK_EX|fcntl.LOCK_NB' in text
    assert 'engine.base.guard_records(GPU)' in text


def test_declares_not_a_stage_instrumentation_or_timing_result():
    text=(HERE/'prepare_openmoji_e45.py').read_text(encoding='utf8')
    assert 'selected_NN_stage_instrumentation_not_yet_tested=True' in text
    assert 'not_a_new_timing_or_energy_or_full_TEST_result=True' in text


def test_all_inherited_config_bytes_are_enrolled(tmp_path):
    a=tmp_path/'a.yaml';b=tmp_path/'b.yaml';c=tmp_path/'c.yaml'
    a.write_text('base_config: b.yaml\n',encoding='utf8')
    b.write_text('base_config: [c.yaml]\n',encoding='utf8')
    c.write_text('value: 1\n',encoding='utf8')
    assert prepared.config_chain(a)=={a.resolve(),b.resolve(),c.resolve()}
