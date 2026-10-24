"""No Torch/CUDA/model import: static entry and partition/guard contracts."""
import ast
from pathlib import Path
import sys
import types
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import lgvq_live_frontend as live
import run_lgvq_live_contract as queue


def test_original_tail_padding_is_charged_but_not_an_extra_prediction():
    indices,valid=live.field_indices(200,16,12)
    assert indices==list(range(192,200))+list(range(8))
    assert valid==[True]*8+[False]*8
    assert sum(len(live.field_indices(200,16,i)[0]) for i in range(13))==208


def test_cpu_video_contract_uses_unmeasured_temporal_fields():
    for field in (13,14):
        indices,valid=live.field_indices(558,16,field)
        assert min(indices)>=200 and len(indices)==16 and all(valid)
    assert live.field_indices(558,1,557)==([557],[True])
    with pytest.raises(IndexError):live.field_indices(200,16,13)


def test_parallel_residuals_are_not_reclassified_as_serial():
    for name in ('vision_routes.0','language_routes.1','custom_conv_electronic_correction.net.0','electronic_quality_norm'):
        assert live.residual_name(name)
    for name in ('readout','frame_merger','parallel_optics.expert_readout','frontend.FrameStem192'):
        assert not live.residual_name(name)


def test_front_restores_only_its_own_scoped_wrappers():
    old=types.SimpleNamespace(forward='old');new=types.SimpleNamespace(forward='new')
    meter=types.SimpleNamespace(_restore=[(old,'forward','old_original'),(new,'forward','new_original')])
    live.restore_since(meter,1)
    assert old.forward=='old' and new.forward=='new_original' and len(meter._restore)==1


def test_static_entry_keeps_original_live_producer_and_two_decode_contracts():
    source=(HERE/'lgvq_live_frontend.py').read_text()
    tree=ast.parse(source)
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert 'front._extract_vision_rows' in calls
    assert 'decode_four_frames' in calls and 'FrameStem' in calls
    assert 'Qwen3VLVisionPatchEmbed' in calls
    assert 'Qwen3VLVisionModel' not in calls
    assert 'from_pretrained' not in calls
    assert 'crop_fraction=1.0' in source and 'FrameStem(192)' in source
    assert 'torch.load(BINDING' not in source


def test_waiting_is_not_a_successful_cpu_proof():
    assert queue.predecessor_ready(dict(status='waiting_foreign_process')) is False
    with pytest.raises(RuntimeError):queue.predecessor_ready(dict(status='stopped'))
    with pytest.raises(ValueError):queue.predecessor_ready(dict(status='complete'))


def test_predecessor_requires_actual_original_qwen_proof_and_digest():
    proof=dict(status='passed',no_gpu=True,baseline_sha256=queue.BASELINE_SHA,
        checks=[dict(test_index=i,bitwise_equal=True) for i in (200,2303)])
    assert queue.predecessor_ready(dict(status='complete',contract_sha256='original'),proof,'original')
    with pytest.raises(ValueError):queue.predecessor_ready(dict(status='complete',contract_sha256='wrong'),proof,'original')
    proof['checks'][0]['bitwise_equal']=False
    with pytest.raises(ValueError):queue.predecessor_ready(dict(status='complete',contract_sha256='original'),proof,'original')
