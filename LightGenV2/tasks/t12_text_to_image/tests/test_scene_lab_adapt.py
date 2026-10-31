import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_adapt import match_train


def fixture():
    requests=[dict(sample_id=str(i),reference_sha256=str(i)*64,prompt='Prompt '+str(i),split='train') for i in range(2)]
    records=[dict(reference_sha256=r['reference_sha256'],prompt=r['prompt'],input_key='key'+r['sample_id'],split='train') for r in reversed(requests)]
    return requests,records


def test_pairs_exact_input_identity_not_row_order():
    requests,records=fixture()
    assert match_train(requests,records)==[('0','key0'),('1','key1')]


def test_nominal_anchor_bounds():
    from LightGenV2.tasks.t12_text_to_image.scene_lab_adapt import validate_nominal_anchor
    assert validate_nominal_anchor(0)==0
    assert validate_nominal_anchor(1)==1
    for value in (-1,1.01,float('nan'),float('inf')):
        with pytest.raises(ValueError):validate_nominal_anchor(value)


def test_dual_domain_loss_preserves_default_and_both_gradients():
    import os
    if os.name=='nt':pytest.skip('Local Windows Torch DLL unavailable; real CPU check is mandatory on server')
    import torch
    from LightGenV2.tasks.t12_text_to_image.scene_lab_adapt import domain_alignment_loss
    from LightGenV2.tasks.t12_text_to_image.qwen3_scene_conditioning import alignment_loss
    target={k:torch.ones(2,3)for k in ('text','image','pooled')}
    actual={k:torch.full((2,3),.3,requires_grad=True)for k in target}
    nominal={k:torch.full((2,3),.7,requires_grad=True)for k in target}
    original=alignment_loss(actual,target)[0]/3
    assert torch.equal(domain_alignment_loss(actual,None,target,0),original)
    expected=(original+alignment_loss(nominal,target)[0]/3)/2
    loss=domain_alignment_loss(actual,nominal,target,1)
    assert torch.equal(loss,expected)
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all()and v.grad.abs().sum()>0 for v in [*actual.values(),*nominal.values()])
    with pytest.raises(ValueError):domain_alignment_loss(actual,None,target,1)


def test_no_test_changed_prompt_or_duplicate_teacher():
    requests,records=fixture()
    for changed in ([dict(requests[0],split='test'),requests[1]],[dict(requests[0],prompt='different'),requests[1]],requests+[requests[0]]):
        with pytest.raises(ValueError):match_train(changed,records)
    with pytest.raises(ValueError):match_train(requests,records+[records[0]])


def test_feature_export_is_optional_same_original_forward():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'scene_lab_native.py').read_text()
    assert 'export_features=False' in source and "if self.export_features:" in source
    assert source.count('self.qwen.model(**inputs')==1
    assert 'self.head(hidden.float(),native[0][None].float())' in source


def test_streaming_gate_requires_complete_actual_prefix_and_released_input_producer(tmp_path):
    import json
    from LightGenV2.tasks.t12_text_to_image.tests.test_scene_lab_protocol import fixture
    from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import STAGES
    from LightGenV2.tasks.t12_text_to_image.scene_lab_adapt import streaming_admission,wait_final_capture
    ledger,_,_=fixture(tmp_path/'ledger');base=ledger.root.parent
    with pytest.raises(FileNotFoundError):streaming_admission(ledger)
    for stage in STAGES[:-1]:(base/(stage+'_outputs_data_stream_complete.json')).write_text(json.dumps(dict(complete=True,count=2,ledger_SHA256=ledger.key)))
    job=base/('native_'+STAGES[-1]);job.mkdir();release=job/'release.json';release.write_text(json.dumps(dict(returncode=0,own_GPU_process_absent=False)))
    with pytest.raises(ValueError):streaming_admission(ledger)
    release.write_text(json.dumps(dict(returncode=0,own_GPU_process_absent=True)))
    with pytest.raises(ValueError):streaming_admission(ledger)
    for request in ledger.requests:
        folder=ledger.folder(STAGES[-1],request);folder.mkdir(parents=True);(folder/'input.ready').touch()
    streaming_admission(ledger)
    with pytest.raises(TimeoutError):wait_final_capture(ledger,ledger.requests[0],0)
    (ledger.folder(STAGES[-1],ledger.requests[0])/'capture.ready').touch();wait_final_capture(ledger,ledger.requests[0],0)
