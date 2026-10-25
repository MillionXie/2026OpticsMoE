import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import t12_baseline_live as live
from finite_queue_budget import valid_window,point_deadline


def packet():
    return dict(status='passed',no_gpu=True,baseline_sha256=live.BASELINE_SHA,sources={'original.py':'a'*64},
        CPU_platform=live.CPU_PLATFORM,check=dict(test_index=200,bitwise_equal=True,
        original_reference_bitwise_equal=True,target_passed_to_model=False),actual_NN_shapes=[dict(test_index=200)])


def test_explicit_finite_wait_window_and_unchanged_GPU_point_cap():
    assert valid_window(100+7200,7200,100)
    assert not valid_window(100+7201,7200,100)
    assert valid_window(100+21600,21600,100)
    assert not valid_window(100+21601,21600,100)
    assert not valid_window(float('inf'),21600,100)
    assert not valid_window(100,21600,100)
    assert point_deadline(100+21600,100)==100+7199
    assert point_deadline(100+50,100)==150


def test_case_is_immutable_SHA_bound_and_source_bound(tmp_path):
    p=packet();live.save_case(tmp_path,p)
    result,reference=live.retained_case(tmp_path,200,p['sources'])
    assert result==p and len(reference['sha256'])==64
    with pytest.raises(FileExistsError):live.save_case(tmp_path,p)
    with pytest.raises(ValueError):live.retained_case(tmp_path,200,{'changed.py':'b'*64})
    (tmp_path/'passed_case_200.json').write_text('{}')
    with pytest.raises(ValueError):live.retained_case(tmp_path,200,p['sources'])


@pytest.mark.parametrize('field,value',[('status','loading'),('no_gpu',False),('baseline_sha256','wrong'),('actual_NN_shapes',[])])
def test_unpassed_or_wrong_case_cannot_be_retained(tmp_path,field,value):
    p=packet();p[field]=value;live.save_case(tmp_path,p)
    with pytest.raises(ValueError):live.retained_case(tmp_path,200,p['sources'])


def test_original_loader_tokenizer_oracle_and_full_forward_AST_unchanged():
    name='LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024/t12_baseline_live.py'
    old=ast.parse(subprocess.check_output(['git','show','0f92114f3:'+name],cwd=live.REPO))
    new=ast.parse(Path(live.__file__).read_text())
    for function in ('dependency_records','bind_dependencies','original_statements','load_runtime','inputs','forward','gold_forward'):
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==function)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==function)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
