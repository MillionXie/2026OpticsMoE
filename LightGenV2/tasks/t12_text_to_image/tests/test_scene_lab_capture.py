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
