"""Pure-CPU, no-model tests of original image-query contracts."""
import copy
import hashlib
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from measure_abo_image_qwen import (original_source,ORIGINAL_SHA,validate_bank,
    MODEL_CONFIG_SHA,MODEL_WEIGHTS_SHA,DOCUMENT_INSTRUCTION)
from audit_abo_image_qwen import validate_calls


def fixture():
    rows=[dict(sample_id=str(i),product_id=str(i//24),label=i//24,image_path=f'test/{i}.jpg') for i in range(2400)]
    calls=[dict(index=i,**row,input_start=i*2.,prepare_end=i*2.+.2,model_start=i*2.+.3,cpu_result_return=i*2.+1,
        input_ids=[[1,2,3]],attention_mask=[[1,1,1]],valid_input_tokens=3,
        input_shapes=dict(input_ids=[1,3],attention_mask=[1,3],image_grid_thw=[1,3],pixel_values=[784,1536]))
        for i,row in enumerate(rows[:200])]
    return calls,rows


def test_first200_all199_gaps_and_original_token_denominator():
    calls,rows=fixture()
    assert validate_calls(calls,rows)==(399.,199.,600)


def test_dropped_first_wrong_order_and_overlap_rejected():
    calls,rows=fixture()
    with pytest.raises(ValueError):validate_calls(calls[1:],rows)
    changed=copy.deepcopy(calls);changed[0],changed[1]=changed[1],changed[0]
    with pytest.raises(ValueError):validate_calls(changed,rows)
    changed=copy.deepcopy(calls);changed[1]['input_start']=.1
    with pytest.raises(ValueError):validate_calls(changed,rows)


def test_wrong_token_or_image_shape_rejected():
    calls,rows=fixture();calls[0]['valid_input_tokens']=2048
    with pytest.raises(ValueError):validate_calls(calls,rows)
    calls,rows=fixture();calls[0]['input_shapes']['image_grid_thw']=[16,3]
    with pytest.raises(ValueError):validate_calls(calls,rows)


def test_original_git_helpers_and_feature_source_are_immutable():
    assert hashlib.sha256(original_source().encode()).hexdigest()==ORIGINAL_SHA
    assert 'scores.argsort(descending=True)' in original_source()


def test_do_not_substitute_saved_FP16_bank_or_wrong_title_identity():
    import numpy as np
    titles=[dict(product_id=str(i),title='title'+str(i)) for i in range(100)]
    vectors=np.zeros((100,2048),dtype=np.float32);vectors[:,0]=1
    receipt=dict(status='complete',purpose='offline_original_B1_FP32_title_bank',
        model_config_sha256=MODEL_CONFIG_SHA,model_weights_sha256=MODEL_WEIGHTS_SHA,
        original_source_sha256=ORIGINAL_SHA,instruction=DOCUMENT_INSTRUCTION,batch=1,
        product_ids=[r['product_id'] for r in titles],titles=[r['title'] for r in titles])
    assert validate_bank(dict(vectors=vectors),receipt,titles).shape==(100,2048)
    with pytest.raises(ValueError):validate_bank(dict(vectors=vectors.astype(np.float16)),receipt,titles)
    receipt['product_ids'].reverse()
    with pytest.raises(ValueError):validate_bank(dict(vectors=vectors),receipt,titles)
