import numpy as np
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_capture import drive_codes,await_payload
from LightGenV2.tasks.t12_text_to_image.scene_lab_hardware import native


def test_capture_codes_preserve_device_mapping_and_zero():
    a=np.zeros((1,478,478),np.uint8);a[0,100:140,200:250]=128;p=np.full_like(a,31)
    inverse=np.arange(256);inverse[128]=127
    amp,phase=drive_codes(a,p,inverse)
    assert amp.shape==(1080,1920)and phase.shape==(1200,1920)
    assert np.array_equal(amp,native(a[0],'amplitude',inverse))
    assert np.array_equal(phase,native(p[0],'phase'))
    assert amp.max()==127 and not amp[:32].any()and np.all(phase[:92]==255)


def test_capture_rejects_missing_batch_float_or_double_inverse():
    a=np.zeros((1,478,478),np.uint8)
    for bad in [a[0],a.astype(np.float32),np.zeros((2,478,478),np.uint8)]:
        with pytest.raises(ValueError):drive_codes(bad,a,np.arange(256))
    inverse=np.arange(256);inverse[0]=1
    with pytest.raises(ValueError):drive_codes(a,a,inverse)


def test_streaming_capture_wait_is_bounded_and_aborts(tmp_path):
    import json
    folder=tmp_path/'sample';folder.mkdir()
    with pytest.raises(TimeoutError):await_payload(folder,tmp_path,0,lambda:None)
    (tmp_path/'input_abort.json').write_text(json.dumps(dict(reason='fixture_failure')))
    with pytest.raises(RuntimeError):await_payload(folder,tmp_path,1,lambda:None)
    (folder/'input.ready').touch();await_payload(folder,tmp_path,0,lambda:None)


def test_explicit_TEST_frame_counts_never_become_TRAIN_or_DEV():
    from LightGenV2.tasks.t12_text_to_image.scene_lab_capture import cohort_summary
    import pytest
    for role in ('train','dev','test','calibration'):
        out=cohort_summary(role,7)
        assert out['model_CCD']==7 and out['TEST_gradients']is False
        assert out['model_TRAIN_CCD']==(7 if role=='train'else 0)
        assert out['model_TEST_CCD']==(7 if role=='test'else 0)
        assert out['TEST_used']==(role=='test')
    with pytest.raises(ValueError):cohort_summary('unknown',7)


def test_declared_model_signal_stop_does_not_change_or_replace_the_frame():
    from LightGenV2.tasks.t12_text_to_image.scene_lab_capture import guard_model_signal
    guard_model_signal(895,0)  # Existing callers retain the old opt-out contract.
    guard_model_signal(1500,1500)
    guard_model_signal(12492,1500)
    with pytest.raises(ValueError,match='before ledger acceptance'):
        guard_model_signal(881,1500)
    for invalid in (-1,65536,float('nan'),float('inf')):
        with pytest.raises(ValueError):guard_model_signal(12000,invalid)
        with pytest.raises(ValueError):guard_model_signal(invalid,1500)


def retained_fixture(tmp_path):
    import json
    from types import SimpleNamespace
    from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import sha,identity
    cohort=tmp_path/'TRAIN';source=cohort/'capture_original';frame=source/'sample0';frame.mkdir(parents=True)
    native=frame/'native.npy';np.save(native,np.full((16,16),20000,np.uint16))
    row=dict(tag='sample0',native_SHA256=sha(native),p99=20000,actual_camera=dict(exposure_us=1999.8,gain=0,auto=0),saturation_fraction=0,capture_UTC='2026-11-01T09:00:00Z')
    (frame/'receipt.json').write_text(json.dumps(row))
    receipt=dict(metadata=dict(native_frame_sha256=sha(native)))
    config=dict(exposure_us=1999.8,gain=0,wait_ms=240)
    (source/'protocol.json').write_text(json.dumps(dict(config=config,stage='language_expert',calibration_report_SHA256='b'*64)))
    requests=[dict(sample_id='sample0'),dict(sample_id='sample1')]
    ledger=SimpleNamespace(root=cohort/'ledger',key='a'*64,requests=requests,session=config,receipt=lambda stage,req:(receipt,None))
    manifest=cohort/'retained.json'
    value=dict(schema='scene_lab_retained_capture_prefix_v1',ledger_sha256=ledger.key,stage='language_expert',source_capture=str(source),retained=[dict(sample_id='sample0',ledger_receipt_identity=identity(receipt),physical_receipt_file_SHA256=sha(frame/'receipt.json'))])
    manifest.write_text(json.dumps(value))
    return ledger,config,source,manifest,value


def test_retained_prefix_is_physical_same_session_and_not_new_acquisition(tmp_path):
    from LightGenV2.tasks.t12_text_to_image.scene_lab_capture import load_retained_manifest,verify_retained_capture
    from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import sha
    ledger,config,source,manifest,value=retained_fixture(tmp_path)
    proof=load_retained_manifest(manifest,sha(manifest),ledger,'language_expert',config,'b'*64)
    r=verify_retained_capture(proof,ledger,'language_expert',ledger.requests[0],1500)
    assert r['sample_id']=='sample0'and r['physical_capture_root']==str(source.resolve())
    with pytest.raises(ValueError,match='outside the retained prefix'):
        verify_retained_capture(proof,ledger,'language_expert',ledger.requests[1],1500)
    assert load_retained_manifest(None,None,None,None,None,None)is None


def test_retained_prefix_rejects_changed_sha_settings_scope_order_and_raw(tmp_path):
    import json
    from LightGenV2.tasks.t12_text_to_image.scene_lab_capture import load_retained_manifest,verify_retained_capture
    from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import sha
    ledger,config,source,manifest,value=retained_fixture(tmp_path)
    with pytest.raises(ValueError,match='SHA mismatch'):load_retained_manifest(manifest,'f'*64,ledger,'language_expert',config,'b'*64)
    with pytest.raises(ValueError,match='settings/calibration'):load_retained_manifest(manifest,sha(manifest),ledger,'language_expert',dict(config,gain=1),'b'*64)
    with pytest.raises(ValueError,match='ledger/stage'):load_retained_manifest(manifest,sha(manifest),ledger,'language_global',config,'b'*64)
    value['retained'][0]['sample_id']='sample1';manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='original request prefix'):load_retained_manifest(manifest,sha(manifest),ledger,'language_expert',config,'b'*64)
    value['retained'][0]['sample_id']='sample0';manifest.write_text(json.dumps(value))
    proof=load_retained_manifest(manifest,sha(manifest),ledger,'language_expert',config,'b'*64)
    (source/'sample0/native.npy').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='native frame changed'):verify_retained_capture(proof,ledger,'language_expert',ledger.requests[0],1500)
