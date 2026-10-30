import json,subprocess,sys
import numpy as np
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import array_sha
from LightGenV2.tasks.t12_text_to_image.scene_lab_pilot import mark_input_ready


def test_ready_marker_requires_complete_matching_inputs(tmp_path):
    z=dict(amplitude=np.zeros((1,478,478),np.float32),total_phase=np.zeros((1,478,478),np.float32),amplitude_code=np.zeros((1,478,478),np.uint8),phase_code=np.zeros((1,478,478),np.uint8))
    keys=dict(amplitude='amplitude_sha256',total_phase='total_phase_sha256',amplitude_code='logical_amplitude_code_sha256',phase_code='logical_phase_code_sha256')
    (tmp_path/'ticket.json').write_text(json.dumps({v:array_sha(z[k])for k,v in keys.items()}))
    with pytest.raises(FileNotFoundError):mark_input_ready(tmp_path)
    assert not(tmp_path/'input.ready').exists()
    np.savez_compressed(tmp_path/'logical_fields.npz',**z);assert mark_input_ready(tmp_path).exists()
    z['amplitude'][0,0,0]=1;np.savez_compressed(tmp_path/'logical_fields.npz',**z)
    with pytest.raises(ValueError):mark_input_ready(tmp_path)


def test_full_cohort_help_never_loads_torch_or_devices():
    r=subprocess.run([sys.executable,'-m','LightGenV2.tasks.t12_text_to_image.scene_lab_pilot','--help'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert all(flag in r.stdout for flag in('--count','--ready-markers','--gpu-memory-fraction','--request-pause-ms'))
