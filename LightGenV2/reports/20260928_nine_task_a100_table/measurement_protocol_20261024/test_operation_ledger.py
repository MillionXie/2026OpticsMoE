"""Pure-shape MAC rules, no Torch/CUDA/model/data dependency."""
from contextlib import ExitStack
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from operation_ledger import (matmul_mac,linear_mac,convolution_mac,attention_mac,
    OperationStageMeter,broadcast)


def test_mac_two_op_and_actual_dynamic_dimensions():
    assert matmul_mac((2,3),(3,4))==24
    assert 2*linear_mac((1,71,2048),(64,2048))==18612224
    assert linear_mac((1,77,2048),(64,2048))!=linear_mac((1,71,2048),(64,2048))


def test_vectors_and_broadcast_batches():
    assert matmul_mac((3,),(3,))==3
    assert matmul_mac((3,),(2,3,4))==24
    assert matmul_mac((2,4,3),(3,))==24
    assert matmul_mac((2,1,4,3),(5,3,6))==720
    assert broadcast((0,),(1,))==(0,)
    with pytest.raises(ValueError):matmul_mac((2,3),(4,2))
    with pytest.raises(ValueError):matmul_mac((2,3,4),(5,4,2))


def test_conv_group_and_transpose_are_explicit():
    assert convolution_mac((1,2,5,5),(3,2,3,3),(1,3,5,5))==1350
    assert convolution_mac((2,4,8,8),(4,1,3,3),(2,4,8,8),groups=4)==4608
    assert convolution_mac((1,4,7,7),(4,3,2,2),(1,3,14,14),transposed=True)==2352
    with pytest.raises(ValueError):convolution_mac((1,2,5,5),(3,2,3,3),(1,4,5,5))


def test_attention_counts_both_qk_and_av_once_no_causal_discount():
    assert attention_mac((1,2,3,4),(1,2,3,4),(1,2,3,4))==144
    assert attention_mac((2,8,7,16),(2,8,11,16),(2,8,11,32))==59136
    with pytest.raises(ValueError):attention_mac((1,8,7,16),(1,2,11,16),(1,2,11,16))


def test_parallel_residual_and_nested_stages_not_recharged():
    meter=OperationStageMeter();meter.sample_index=3
    with meter.scope('parallel_residual',family='parallel_residual',charged=False):
        assert meter.current==('parallel_residual','parallel_residual',False)
        with meter.scope('child_linear',charged=True):
            assert meter.current==('parallel_residual','parallel_residual',False)
    assert meter.current is None
    assert list(meter.stage_calls.items())==[((3,'parallel_residual'),1)]


def test_scope_restored_on_original_model_failure():
    meter=OperationStageMeter()
    with pytest.raises(RuntimeError):
        with meter.scope('head'):raise RuntimeError('original forward failed')
    assert meter.current is None
