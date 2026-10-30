import json,subprocess,sys
from pathlib import Path
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_camera_process import CameraProcess,checked_frame_path


def test_worker_capture_path_cannot_overwrite_or_escape(tmp_path):
    assert checked_frame_path(tmp_path/'a.npy',tmp_path)==tmp_path/'a.npy'
    (tmp_path/'a.npy').write_bytes(b'fixture')
    for path in(tmp_path/'a.npy',tmp_path.parent/'outside.npy',tmp_path/'a.txt'):
        with pytest.raises(ValueError):checked_frame_path(path,tmp_path)


def test_RPC_transport_with_explicit_non_camera_fixture(tmp_path,monkeypatch):
    # Fake transport only. It creates no valid CCD and is never a physical test.
    actual_popen=subprocess.Popen
    fake="""import sys,json,os
from pathlib import Path
for line in sys.stdin:
 r=json.loads(line);op=r['op']
 if op=='capture':Path(r['path']).write_bytes(b'RPC_TRANSPORT_FIXTURE_NOT_CCD')
 result=dict(device=dict(fixture=True),actual=dict(exposure_us=100,gain=0,auto=0),worker_pid=os.getpid())
 print('T12_CAMERA_RPC '+json.dumps(dict(id=r['id'],ok=True,result=result)),flush=True)
 if op=='close':break
"""
    def launch(command,**kwargs):return actual_popen([sys.executable,'-u','-c',fake],**kwargs)
    monkeypatch.setattr(subprocess,'Popen',launch)
    with CameraProcess('NOT_A_REAL_SDK',allowed_root=tmp_path)as proxy:
        proxy.capture(tmp_path/'fixture.npy');assert proxy.device_info()['fixture']
        assert proxy.live_settings()['exposure_us']==100
    assert proxy.process.returncode==0
    assert(tmp_path/'fixture.npy').read_bytes()==b'RPC_TRANSPORT_FIXTURE_NOT_CCD'
