import unittest
import numpy as np
from .hardware_bridge import native_image,crop_native


class TestHardware(unittest.TestCase):
    def test_native_panels(self):
        x=np.zeros((1016,1016),np.uint8);x[17,29]=87
        amplitude=native_image(x);phase=native_image(x,True)
        self.assertEqual(amplitude.shape,(1080,1920));self.assertEqual(phase.shape,(1200,1920))
        np.testing.assert_array_equal(amplitude[32:1048,452:1468],x)
        np.testing.assert_array_equal(255-phase[92:1108,452:1468],np.flipud(np.fliplr(x)))
        self.assertEqual(int(amplitude[:32].max()),0)
        with self.assertRaises(ValueError):native_image(np.zeros((478,478),np.uint8))

    def test_full_precision_crop(self):
        raw=np.full((2048,2048),12345,np.uint16)
        result=crop_native(raw,np.eye(3,dtype=np.float64))
        self.assertEqual(result.dtype,np.float32);self.assertEqual(result.shape,(1016,1016))
        np.testing.assert_allclose(result,12345/65535,atol=1e-7,rtol=0)
        self.assertNotAlmostEqual(float(result[0,0])*255,round(float(result[0,0])*255),places=3)


if __name__=='__main__':unittest.main()
