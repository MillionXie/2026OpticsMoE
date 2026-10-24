"""No-model interval/identity contracts; optional exhaustive real CPU BF16 test."""
import ast
from collections import Counter
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from measure_openmoji_e45 import partition_live_input
from openmoji_grid_audit import metrics,native_bf16_edit


def test_input_partition_is_disjoint_and_keeps_the_real_embedding():
    original=dict(stage='frontend.plain_token_embedding',start_monotonic=2.,end_monotonic=3.)
    meter=SimpleNamespace(rows=[original],sample_index=7)
    partition_live_input(meter,0,1.,4.)
    assert meter.rows[1] is original
    assert [(r['start_monotonic'],r['end_monotonic']) for r in meter.rows]==[(1,2),(2,3),(3,4)]
    assert meter.rows[0]['family']=='input' and meter.rows[-1]['cuda_event_ms'] is None
    with pytest.raises(ValueError):partition_live_input(SimpleNamespace(rows=[original],sample_index=0),0,2.1,4)


def test_bfloat16_native_sigmoid_boundary_not_fp32_or_zero_logit_threshold():
    logits=np.array([-.0078125,-.00390625,0,.00390625],dtype=np.float32)
    assert native_bf16_edit(logits).tolist()==[False,True,True,True]
    with pytest.raises(ValueError):native_bf16_edit(np.array([.001],dtype=np.float32))


def test_original_changed_cell_metric_and_source_preservation():
    source=np.zeros((6,6),dtype=int);target=source.copy();target[0,0]=2
    row=dict(source_grid=source.tolist(),target_grid=target.tolist(),task_index=0)
    prediction=source.copy();edited=np.zeros((6,6),dtype=bool);edited[0,0]=True;prediction[0,0]=2
    result=metrics(prediction,edited,0,row)
    assert all(v==1 for v in result.values())


def test_true_original_model_and_no_target_network_input_or_gallery_warmup():
    text=(HERE/'measure_openmoji_e45.py').read_text(encoding='utf8');tree=ast.parse(text)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(index, row)' and 'live_input(' in ast.unparse(n))
    query=ast.unparse(loop)
    assert 'live_input(' in query and 'result = model(pixels, hidden)' in query
    assert 'compose_prediction(' in query and 'targets' not in query and 'target_grid' not in query
    assert 'range(200)' not in query and 'records[:200]' in query
    assert 'torch.autocast(' in query and 'torch.bfloat16' in query
    assert "common_Qwen_input_tokens200=None" in text
    assert 'time.sleep' not in query


def test_controller_is_fresh_process_and_auditor_has_no_model_or_torch():
    controller=(HERE/'run_openmoji_e45.py').read_text(encoding='utf8')
    audit=(HERE/'audit_openmoji_e45.py').read_text(encoding='utf8')
    assert 'for repeat in args.repeats' in controller and 'subprocess.Popen(' in controller
    assert 'engine.daemon(False)' in controller and 'child.wait(timeout=1)' in controller
    assert 'import torch' not in controller+audit and 'torch.load' not in audit
    assert "observed_energy(" in audit and "stage_contract_sha256" in audit


@pytest.mark.skipif(os.environ.get('OPENMOJI_REAL_CPU_TEST')!='1',reason='explicit server CPU only; no local Torch repair')
def test_every_finite_bfloat16_sigmoid_decision_matches_original_torch_CPU():
    import torch
    torch.set_num_threads(1)
    values=(np.arange(65536,dtype=np.uint32)<<16).view(np.float32)
    values=values[np.isfinite(values)]
    expected=torch.sigmoid(torch.from_numpy(values).to(torch.bfloat16)).ge(.5).numpy()
    assert np.array_equal(expected,native_bf16_edit(values))
