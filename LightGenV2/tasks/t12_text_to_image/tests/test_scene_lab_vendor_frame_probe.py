import ctypes
import numpy as np
from PIL import Image
from LightGenV2.tasks.t12_text_to_image.scene_lab_vendor_frame_probe import header_fields,compare_file


def test_header_keeps_offset_stride_and_depth_code():
    class Frame(ctypes.Structure):
        _fields_=[('signature',ctypes.c_char*8),('usHeader',ctypes.c_uint16),('usOffset',ctypes.c_uint16),('uiWidthStep',ctypes.c_uint32),('ucDepth',ctypes.c_uint8)]
    f=Frame(b'FRAME',80,0,4096,2);r=header_fields(f)
    assert r['usHeader']==80 and r['usOffset']==0 and r['uiWidthStep']==4096 and r['ucDepth']==2


def test_tiff_comparison_does_not_normalize_dark_values(tmp_path):
    a=np.arange(100,dtype=np.uint16).reshape(10,10)+300;p=tmp_path/'native.tif';Image.fromarray(a).save(p)
    r=compare_file(p,a);assert r['same_pixel_values']and r['maximum']==399
    assert not compare_file(p,(a*100).astype(np.uint16))['same_pixel_values']
