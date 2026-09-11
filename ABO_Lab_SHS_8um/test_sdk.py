import ctypes
import unittest
from sdk import NodeInfo, IntegerInfo, FloatInfo, BufferInfo, decode_mono, SDKError
import numpy as np


class ABITests(unittest.TestCase):
    def test_packed_node_abi(self):
        self.assertEqual(ctypes.sizeof(IntegerInfo),32)
        self.assertEqual(ctypes.sizeof(FloatInfo),32)
        self.assertEqual(ctypes.sizeof(NodeInfo),104)
        self.assertEqual(NodeInfo.visibility.offset,40)

    def test_buffer_abi(self):
        self.assertEqual(ctypes.sizeof(BufferInfo),112)
        self.assertEqual(BufferInfo.timestamp.offset,64)

    def test_padding_and_copy(self):
        a,bits=decode_mono(bytes([99,1,2,0,3,4,0]),2,2,0x01080001,xpadding=1,image_offset=1)
        np.testing.assert_array_equal(a,[[1,2],[3,4]])
        self.assertEqual(bits,8)

    def test_reject_short_and_packed(self):
        with self.assertRaises(SDKError):decode_mono(b'xx',3,2,0x01080001)
        with self.assertRaises(SDKError):decode_mono(b'x'*6,2,2,0x010c0047)

    def test_safe_exposure_restoration_order(self):
        from capture import restore_settings
        class Fake:
            def __init__(self):self.values={'ExposureTime':'100','AcquisitionFrameRate':'2250'};self.calls=[]
            def get(self,name):return self.values[name]
            def stop(self):pass
            def set(self,name,value):self.calls.append((name,str(value)));self.values[name]=str(value)
        camera=Fake();before={'ExposureTime':{'value':'444.2'},'AcquisitionFrameRate':{'value':'2250'}}
        self.assertEqual(restore_settings(camera,before,list(before)),[])
        self.assertEqual(camera.calls,[('AcquisitionFrameRate','100'),('ExposureTime','444.2'),('AcquisitionFrameRate','2250')])


if __name__=='__main__':unittest.main()
