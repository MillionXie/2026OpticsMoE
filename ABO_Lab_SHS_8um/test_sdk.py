import ctypes
import unittest
from sdk import NodeInfo, IntegerInfo, FloatInfo, BufferInfo, decode_mono, SDKError
import numpy as np


class ABITests(unittest.TestCase):
    def test_startup_warmup_once_even_for_dark_scene(self):
        from sdk import Camera
        from unittest.mock import patch,Mock
        camera=Camera({'startup_warmup_s':0.1})
        camera.grab=Mock(return_value=(np.zeros((2,2)),{}))
        with patch('sdk.time.perf_counter',side_effect=[0,0,0.05,0.1,0.1]):
            camera._warmup_first_stream()
        self.assertEqual(camera.grab.call_count,2)
        self.assertEqual(camera.startup_warmup['discarded_frames'],2)
        camera._warmup_first_stream()
        self.assertEqual(camera.grab.call_count,2)

    def test_startup_warmup_bounds(self):
        from sdk import Camera
        for value in (-1,11,float('nan')):
            with self.assertRaises(ValueError):Camera({'startup_warmup_s':value})

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
