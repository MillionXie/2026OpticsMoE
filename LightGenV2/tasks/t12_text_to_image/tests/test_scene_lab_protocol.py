import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import Ledger,STAGES,encode,array_sha,identity


def fixture(root):
    dark=np.full((1,478,478),2,dtype=np.float32);gain=np.full_like(dark,.25)
    session=json.loads((Path(__file__).parents[1]/'scene_lab_session_template.json').read_text())
    session.update(calibration_approved=True,session_id='SYNTHETIC_ONLY',scope='offline_synthetic',camera='fake',wavelength_nm=532,distance_m=.1,
        amplitude_pitch_um=8,phase_pitch_um=8,amplitude_native_shape=[1080,1920],phase_native_shape=[1200,1920],orientation='fixture_canonical',exposure_us=100,gain='fixture',wait_ms=240,intensity_units='fixture_sim_unit')
    for key in session:
        if key.endswith('_sha256'):session[key]='a'*64
    session.update(dark_sha256=array_sha(dark),gain_map_sha256=array_sha(gain))
    models=dict(F9='b'*64,F6='c'*64,source_commit='d'*40)
    requests=[dict(sample_id=str(i),split='train',reference='white.png',reference_sha256='e'*64,prompt='Natural scene. Two sentences.',seed=i,category='fixture')for i in range(2)]
    return Ledger(root,session,models,requests,create=True),dark,gain


def runner(request,provider):
    # A deterministic surrogate for the six callbacks, NOT a Qwen quality test.
    amplitude=np.full((1,478,478),.2+request['seed']*.01,dtype=np.float32)
    for stage in STAGES:
        intensity=provider(stage,amplitude,np.full_like(amplitude,1.7),None)
        amplitude=np.full_like(amplitude,.1+float(intensity.mean())*.01)
    return amplitude


def accept(ledger,stage,request,dark,gain,wrong=False):
    folder=ledger.folder(stage,request);ticket=json.loads((folder/'ticket.json').read_text());raw=np.full_like(dark,6)
    metadata=dict(ticket_sha256=identity(ticket),session_sha256=identity(ledger.session),stage=stage,
        request_sha256=identity(request),canonical_orientation=ledger.session['orientation'],exposure_us=100,gain='fixture',wait_ms=240,
        frame_provenance='synthetic_fixture',native_frame_sha256='f'*64,canonical_raw_sha256=array_sha(raw),capture_UTC='2026-10-30T12:00:00Z')
    if wrong:metadata['stage']=STAGES[(STAGES.index(stage)+1)%len(STAGES)]
    return ledger.accept(stage,request,raw,metadata,dark,gain)


def test_full_cohort_gate_and_six_layer_recompute(tmp_path):
    ledger,dark,gain=fixture(tmp_path/'run')
    with pytest.raises(FileNotFoundError):ledger.prepare(STAGES[1],runner)
    for stage in STAGES:
        assert len(ledger.prepare(stage,runner))==2
        with pytest.raises(ValueError):accept(ledger,stage,ledger.requests[0],dark,gain,wrong=True)
        accept(ledger,stage,ledger.requests[0],dark,gain)
        index=STAGES.index(stage)
        if index<5:
            with pytest.raises(FileNotFoundError):ledger.prepare(STAGES[index+1],runner)
        accept(ledger,stage,ledger.requests[1],dark,gain)
        assert ledger.prepare(stage,runner)==[]
    np.testing.assert_allclose(ledger.infer(ledger.requests[0],runner),.11,atol=1e-6)
    with pytest.raises(FileExistsError):accept(ledger,STAGES[-1],ledger.requests[0],dark,gain)


def test_wrong_or_changed_payload_rejected(tmp_path):
    ledger,dark,gain=fixture(tmp_path/'run');ledger.prepare(STAGES[0],runner)
    with pytest.raises(ValueError):accept(ledger,STAGES[0],ledger.requests[0],dark+1,gain)
    for request in ledger.requests:accept(ledger,STAGES[0],request,dark,gain)
    def changed(request,provider):return provider(STAGES[0],np.ones((1,478,478),np.float32)*.9,np.zeros((1,478,478),np.float32),None)
    with pytest.raises(ValueError):ledger.prepare(STAGES[1],changed)
    path=ledger.folder(STAGES[0],ledger.requests[0])/'calibrated.npy';np.save(path,np.zeros((1,478,478),np.float32))
    with pytest.raises(ValueError):ledger.receipt(STAGES[0],ledger.requests[0])


def test_missing_or_ideal_not_accepted(tmp_path):
    ledger,_,_=fixture(tmp_path/'run')
    def ideal(request,provider):return provider(STAGES[0],np.zeros((1,478,478)),np.zeros((1,478,478)),np.ones((1,478,478)))
    with pytest.raises(ValueError):ledger.prepare(STAGES[0],ideal)
    with pytest.raises(ValueError):ledger.prepare(STAGES[0],lambda *args:None)
    with pytest.raises(FileNotFoundError):ledger.infer(ledger.requests[0],runner)


def test_codes_zero_phase_wrap_and_no_second_tanh():
    a=np.full((1,478,478),.5,np.float32);a[0,0,0]=0;p=np.full_like(a,-np.pi/2)
    _,_,ag,pg=encode(a,p);expected=np.floor(np.remainder(p.astype(float),2*np.pi)*256/(2*np.pi))
    assert ag[0,0,0]==0 and ag[0,0,1]==128 and np.array_equal(pg,expected)
    with pytest.raises(ValueError):encode(a*3,p)
    assert np.max(abs(ag.astype(float)/255-a))<=.5/255+1e-7


def test_changed_contract_test_or_unreviewed_session(tmp_path):
    ledger,_,_=fixture(tmp_path/'run')
    session=dict(ledger.session,calibration_approved=False)
    with pytest.raises(ValueError):Ledger(tmp_path/'other',session,ledger.models,ledger.requests,create=True)
    with pytest.raises(ValueError):Ledger(ledger.root,dict(ledger.session,exposure_us=101),ledger.models,ledger.requests)
    with pytest.raises(ValueError):Ledger(tmp_path/'other',ledger.session,ledger.models,[dict(ledger.requests[0],split='test')],create=True)


def test_native_head_flag_restoration_is_not_defaulted():
    import ast
    source=(Path(__file__).parents[1]/'scene_lab_native.py').read_text()
    ast.parse(source)
    assert "expected_joint=expected_sha256==F9"in source
    assert "QwenSDXLConditions(joint_image_context=expected_joint)"in source and 'strict=True'in source
    assert 'use_cache=False' in source and 'simulation_diagnostic=False' in source
    assert "self.qwen.model(**inputs"in source and 'self.head(hidden,native[0][None])'in source
