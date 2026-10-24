from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compact_student_shape_receipt import subtotal


def group():
    return dict(calls=600,actual_NN_MAC_OP600=1200,charged_serial_NN_OP600=800,parallel_residual_NN_OP600=400,
        cost_references=[dict(summary_sha256=str(i),seconds200=1,joules200=2) for i in (1,2,3)],
        per_query_MAC=dict(actual_NN_MAC=1,charged_serial_NN_MAC=2/3,parallel_residual_NN_MAC=1/3,
            serial_NN_terms=dict(x=2/3),parallel_NN_terms=dict(x=1/3),other_executed_digital_dense_MAC=dict(router=1)),
        other_executed_digital_dense_MAC600=600,
        repeat_results=[dict(repeat=i,MAC200=200,records=[dict(MAC=1)]*200) for i in (1,2,3)])


def test_actual_NN_not_only_selected_and_router_is_separate():
    v=subtotal(group(),3,6,['1','2','3'])
    assert v['OP600']==1200 and v['charged_serial_NN_OP600']==800
    assert v['other_executed_digital_dense_OP600_not_in_NN_subtotal']==1200
    assert v['physical_optical_dense_equivalent_OP'] is None


def test_scope_cost_or_retained_raw_change_rejected():
    for seconds,energy,reports in [(3.1,6,['1','2','3']),(3,6.1,['1','2','3']),(3,6,['3','2','1'])]:
        with pytest.raises(ValueError):subtotal(group(),seconds,energy,reports)
    v=group();v['parallel_residual_NN_OP600']=0
    with pytest.raises(ValueError):subtotal(v,3,6,['1','2','3'])


def test_missing_call_or_unaccounted_matrix_rejected():
    v=group();v['repeat_results'][1]['records']=[]
    with pytest.raises(ValueError):subtotal(v,3,6,['1','2','3'])
    v=group();v['per_query_MAC']['serial_NN_terms']['unexpected']=1
    with pytest.raises(ValueError):subtotal(v,3,6,['1','2','3'])
