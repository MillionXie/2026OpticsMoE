"""Pinned actual source graph and CPU-only ordering; no model/data/GPU load."""
import ast
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from t12_live_frontend import BODY_SHA,BASELINE_SHA,WEIGHT_SHA,PINS,TEST_SHA
from run_t12_live_contract import predecessor_ready,predecessor_file_ready


def test_exact_adopted_bodies_and_native_data():
    assert BODY_SHA.startswith('5b4f9a37') and BASELINE_SHA.startswith('599805ad')
    assert all(len(v)==64 and set(v)<=set('0123456789abcdef') for v in (BODY_SHA,BASELINE_SHA,WEIGHT_SHA,TEST_SHA,*PINS.values()))


def test_waits_for_all_qualified_previous_rounds_and_MPS_release():
    assert not predecessor_ready({'status':'running'})
    done=dict(status='complete',mps_released=True,completed=['offline_title_bank','ours_repeat01','ours_repeat02','ours_repeat03'])
    assert predecessor_ready(done)
    assert not predecessor_ready(dict(done,mps_released=False))
    assert not predecessor_ready(dict(done,completed=['ours_repeat01']))
    with pytest.raises(RuntimeError):predecessor_ready({'status':'stopped'})
    source=(HERE/'run_t12_live_contract.py').read_text()
    assert "'--previous',type=Path,default=PREVIOUS" in source
    assert "predecessor_file_ready(previous/'suite_status.json')" in source


def test_predecessor_not_yet_created_or_partial_status_is_bounded_pending(tmp_path):
    import json
    path=tmp_path/'suite_status.json'
    assert predecessor_file_ready(path) is False
    path.write_text('{')
    assert predecessor_file_ready(path) is False
    path.write_text(json.dumps(dict(status='running')))
    assert predecessor_file_ready(path) is False
    path.write_text(json.dumps(dict(status='complete',mps_released=True,completed=['offline_title_bank','ours_repeat01','ours_repeat02','ours_repeat03'])))
    assert predecessor_file_ready(path) is True


def test_live_input_not_cache_or_target_and_original_quantization():
    source=(HERE/'t12_live_frontend.py').read_text();tree=ast.parse(source)
    reference=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference_input'))
    assert '_load_product(source,' in reference and '_load_product(design,' not in reference
    online=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='online_embeddings'))
    assert '_qwen_prompts(processor,[prompt])' in online and '.half().float()' in online and '.batch(' not in online and 'torch.load(' not in online
    assert "for i in (200,2303)" in source and 'target_passed_to_model=False' in source
    assert '.backward(' not in source and 'torch.cuda.is_initialized()' in source
