from datetime import datetime,timezone,timedelta
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_camera_only_probe import validate_holder


def test_separate_camera_requires_live_explicit_SLM_only_owner():
    now=datetime.now(timezone.utc);r=dict(SLMs_held=True,camera_SDK_opened=False,deadline_UTC=(now+timedelta(seconds=120)).isoformat());lock=dict(scope='SLMs_only',camera_SDK_owned=False)
    validate_holder(r,lock,now)
    with pytest.raises(ValueError):validate_holder(dict(r,camera_SDK_opened=True),lock,now)
    with pytest.raises(ValueError):validate_holder(r,dict(lock,scope='all_devices'),now)
    with pytest.raises(ValueError):validate_holder(dict(r,deadline_UTC=(now+timedelta(seconds=5)).isoformat()),lock,now)


def test_vendor_save_audit_decodes_same_frame_without_capture(tmp_path):
    import ctypes
    import os
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from LightGenV2.tasks.t12_text_to_image.scene_lab_camera_only_probe import audit_vendor_save
    data=np.arange(16,dtype=np.uint16).reshape(4,4)
    class Frame(ctypes.Structure):_fields_=[('dummy',ctypes.c_int)]
    def save(handle,request):
        assert handle==123
        Image.fromarray(data).save(request.path.decode('mbcs'if os.name=='nt'else'utf-8')+'.tif')
        return 1
    module=SimpleNamespace(TUIMG_FORMATS=SimpleNamespace(TUFMT_TIF=SimpleNamespace(value=2)),
        TUCAM_FILE_SAVE=lambda fmt,path,frame:SimpleNamespace(path=path),TUCAM_File_SaveImage=save)
    camera=SimpleNamespace(_module=module,_frame=Frame(),_handle=lambda:123,_check=lambda status,op:None)
    result=audit_vendor_save(camera,tmp_path/'vendor',data)
    assert result['equals_decoded_array']and not result['new_camera_frame']and not result['settings_written']
    with pytest.raises(FileExistsError):audit_vendor_save(camera,tmp_path/'vendor',data)
