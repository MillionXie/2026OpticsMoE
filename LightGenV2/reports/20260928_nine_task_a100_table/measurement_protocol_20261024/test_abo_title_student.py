"""CPU-only title work, selected-stage and original-query boundary contracts."""
import ast
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import measure_abo_title_student as measured


def workload():
    work=[dict(index=i,unique_query_index=i%100,pass_index=i//100,row=dict(product_id=str(i%100),title='title'+str(i%100))) for i in range(200)]
    calls=[dict(index=str(i),query_index=str(i%100),pass_index=str(i//100),product_id=str(i%100),title='title'+str(i%100),valid_input_tokens=str(48 if i<176 else 47)) for i in range(200)]
    return calls,work


def test_shared_token_work_same_query_not_native_internal_length():
    calls,work=workload();assert sum(measured.common_work(calls,work))==9576
    calls[0]['product_id']='other'
    with pytest.raises(ValueError):measured.common_work(calls,work)


def test_rejects_noncanonical_shared_token_numerator():
    calls,work=workload();calls[0]['valid_input_tokens']='999'
    with pytest.raises(ValueError):measured.common_work(calls,work)


def test_true_input_no_cached_features_or_gallery_forwards_in_query_loop():
    text=(HERE/'measure_abo_title_student.py').read_text(encoding='utf8');tree=ast.parse(text)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='request')
    query=ast.unparse(loop)
    assert 'original._text_inputs(' in query and 'original.move_inputs(' in query
    assert 'original.student_embeddings(' in query and 'embedding[0].detach().float().cpu()' in query
    assert 'scores.argsort(descending=True, stable=True)' in query
    assert 'encode_student_samples' not in query
    assert 'original._encode_titles_student' not in query
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step') for n in ast.walk(tree))


def test_three_light_passes_and_all_true_gaps_charged():
    text=(HERE/'measure_abo_title_student.py').read_text(encoding='utf8')
    assert 'optical_ms=200*3*1.0447' in text
    assert "sum(c['internal_gap_ms'] for c in calls)" in text
    assert 'diagnostic_GPU_energy_not_used_in_phase_estimate=True' in text
    assert 'gallery=F.normalize(torch.from_numpy(vectors),dim=1)' in text


def test_controller_has_fresh_process_no_graph_or_warmup():
    text=(HERE/'run_abo_title_student.py').read_text(encoding='utf8')
    tree=ast.parse(text)
    assert 'for repeat in args.repeats' in text and 'subprocess.Popen(' in text
    assert 'warmup_forwards' not in text and 'torch' not in text
    assert 'validate_gallery(bank,titles,test)' in text
    assert 'fcntl.LOCK_EX|fcntl.LOCK_NB' in text


def test_auditor_never_loads_or_replays_a_model():
    text=(HERE/'audit_abo_title_student.py').read_text(encoding='utf8')
    assert 'import torch' not in text and 'torch.load' not in text
    assert "np.argsort(-scores,axis=1,kind='stable')" in text
    assert "ledger_key(e) for e in contract['unique_stage_ledger']" in text
    assert "observed_bridge_calls=0" in text
