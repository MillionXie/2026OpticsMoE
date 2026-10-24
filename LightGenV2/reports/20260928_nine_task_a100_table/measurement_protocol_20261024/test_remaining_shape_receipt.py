import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from remaining_shape_receipt import independent_pair,qwen_subtotal,image_student_subtotal


def base():
    return dict(calls=600,cost_references=[dict(report_sha256=str(i),summary_sha256=str(i),seconds200=1,joules200=2) for i in (1,2,3)])


def test_only_checkout_provenance_can_differ(tmp_path):
    a=tmp_path/'a.json';b=tmp_path/'b.json'
    a.write_text(json.dumps(dict(commit='a',formula=dict(MAC=5))));b.write_text(json.dumps(dict(commit='b',formula=dict(MAC=5))))
    assert independent_pair(a,b,'commit')['commit']=='a'
    b.write_text(json.dumps(dict(commit='b',formula=dict(MAC=6))))
    with pytest.raises(ValueError):independent_pair(a,b,'commit')


def test_qwen600_and_exact_cost_binding():
    v=dict(**base(),GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,complete_scalar_OP_claim=False,
        repeat_results=[dict(repeat=i,MAC200=200,records=[dict(MAC=1,MAC_terms=dict(x=1))]*200) for i in (1,2,3)],MAC_OP600=1200)
    assert qwen_subtotal(v,3,6,['1','2','3'])['TOPS']==pytest.approx(400/1e12)
    for seconds,energy,reports in [(3.1,6,['1','2','3']),(3,6.1,['1','2','3']),(3,6,['2','1','3'])]:
        with pytest.raises(ValueError):qwen_subtotal(v,seconds,energy,reports)
    v['repeat_results'][1]['records']=v['repeat_results'][1]['records'][:-1]
    with pytest.raises(ValueError):qwen_subtotal(v,3,6,['1','2','3'])


def student():
    return dict(**base(),actual_NN_MAC600=6000,charged_serial_NN_MAC600=4200,parallel_residual_NN_MAC600=1200,
        native_position_MAC600_not_in_selected_timing_spans=600,no_GPU_or_neural_forwards=True,no_new_time_or_power=True,
        complete_scalar_OP_claim=False,native_mergers_executed_per_query=1,auxiliary_deepstack_mergers_executed=0,
        offline_gallery_inference_included=False,physical_optical_dense_equivalent_OP=None,
        per_call_first_repeat=[dict(actual_MAC=10,selected_serial_MAC=7,parallel_residual_MAC=2,native_position_MAC_not_in_selected_spans=1)]*200)


def test_one_actual_merger_and_unknown_optical_not_capacity():
    v=student();result=image_student_subtotal(v,3,6,['1','2','3'])
    assert result['OP600']==12000 and result['physical_optical_dense_equivalent_OP'] is None
    for key,bad in [('native_mergers_executed_per_query',4),('auxiliary_deepstack_mergers_executed',3),('physical_optical_dense_equivalent_OP',0)]:
        changed=copy.deepcopy(v);changed[key]=bad
        with pytest.raises(ValueError):image_student_subtotal(changed,3,6,['1','2','3'])


def test_all_actual_partition_and_all600_records_required():
    v=student();v['parallel_residual_NN_MAC600']+=1
    with pytest.raises(ValueError):image_student_subtotal(v,3,6,['1','2','3'])
    v=student();v['per_call_first_repeat']=v['per_call_first_repeat'][:-1]
    with pytest.raises(ValueError):image_student_subtotal(v,3,6,['1','2','3'])
