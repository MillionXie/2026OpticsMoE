import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from t12_student_saved_shape_mac import module_mac,proof_reference,request_terms,split_cpu_forwards,expected_shapes


def test_original_two_proof_indices_not_double_counted():
    proof,template,samples,_=proof_reference()
    assert len(template)==56 and len(proof['actual_NN_shapes'])==224
    assert [s['test_index'] for s in samples]==[200,2303]
    for s in samples:
        assert s['actual_NN_MAC']==s['charged_serial_NN_MAC']+s['parallel_residual_NN_MAC']
        assert s['parallel_NN_terms']['text.layers.0.functional_QK_AV']==2*s['valid_tokens']**2*512
        assert s['other_executed_digital_dense_MAC']['two_router_dense_mask_contractions']==2*4*478**2


def test_duplicates_and_missing_modules_rejected():
    proof,template,_,_=proof_reference()
    rows=[r for r in proof['actual_NN_shapes'] if r['test_index']==200]
    bad=copy.deepcopy(rows);bad[-1]['weight_shape'][-1]=3
    with pytest.raises(ValueError):split_cpu_forwards(bad)
    with pytest.raises(ValueError):request_terms(template+template,template,38)
    with pytest.raises(ValueError):request_terms(template[:-1],template,38)


def test_original_linear_and_grouped_dense_convolution():
    assert module_mac(dict(kind='Linear',input_shape=[1,38,512],output_shape=[1,38,640],weight_shape=[640,512],groups=None))==38*640*512
    assert module_mac(dict(kind='Conv2d',input_shape=[1,224,14,14],output_shape=[1,224,14,14],weight_shape=[224,1,3,3],groups=224))==196*224*9
    with pytest.raises(ValueError):module_mac(dict(kind='Unknown',input_shape=[1],output_shape=[1],weight_shape=[1],groups=None))


def test_length_changes_only_original_language_token_axes():
    _,template,_,_=proof_reference()
    one=expected_shapes(template,1);last=expected_shapes(template,64)
    a=request_terms(one,template,1);b=request_terms(last,template,64)
    for r,s in zip(one,last):
        if r['module'].startswith('editor.'):assert r==s
    assert a['serial_NN_terms']['text.optical.core.input_adapter']==2*512*224
    assert b['serial_NN_terms']['text.optical.core.input_adapter']==2*64*512*224
    for length in (0,65):
        with pytest.raises(ValueError):expected_shapes(template,length)
