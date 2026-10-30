import ctypes
from types import SimpleNamespace
from LightGenV2.tasks.t12_text_to_image.scene_lab_raw_format_probe import set_buffer_format


def test_format_switch_is_buffer_only():
    calls=[]
    class Frame(ctypes.Structure):_fields_=[('pBuffer',ctypes.c_void_p),('ucFormatGet',ctypes.c_uint8),('uiRsdSize',ctypes.c_uint32)]
    def function(name):
        def call(*args):calls.append(name);return 1
        return call
    m=SimpleNamespace(TUCAM_FRAME=Frame,TUCAM_CAPTURE_MODES=SimpleNamespace(TUCCM_SEQUENCE=SimpleNamespace(value=0)))
    for name in ('TUCAM_Buf_AbortWait','TUCAM_Cap_Stop','TUCAM_Buf_Release','TUCAM_Buf_Alloc','TUCAM_Cap_Start'):setattr(m,name,function(name))
    camera=SimpleNamespace(_module=m,_handle=lambda:123,_capture_started=True,_buffer_allocated=True,_check=lambda result,operation:None)
    set_buffer_format(camera,16)
    assert calls==['TUCAM_Buf_AbortWait','TUCAM_Cap_Stop','TUCAM_Buf_Release','TUCAM_Buf_Alloc','TUCAM_Cap_Start']
    assert camera._frame.ucFormatGet==16 and camera._capture_started and camera._buffer_allocated
