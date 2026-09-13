import unittest
import numpy as np
from gray_response_scan import aperture,uniform,validate_settings

class GrayResponseTests(unittest.TestCase):
    def test_requested_settings(self):
        for e,w in [(400,250),(350,200),(350,250),(300,220)]:validate_settings(e,w)
        for e,w in [(float('nan'),200),(400,float('nan')),(400,10),(10000,200)]:
            with self.assertRaises(ValueError):validate_settings(e,w)
    def test_physical_aperture_and_fixed_background(self):
        c=dict(model_active_pixels=478,model_pitch_um=17,amplitude_slm=dict(pixel_pitch_um=8,center_xy=[960,540]))
        self.assertEqual(aperture(c),(452,32,1016))
        a=uniform(c,128);self.assertEqual(a.shape,(1080,1920));self.assertEqual(a.dtype,np.uint8)
        self.assertEqual(int((a==128).sum()),1016**2);self.assertEqual(int(a[0,0]),0)
    def test_invalid_position(self):
        c=dict(model_active_pixels=478,model_pitch_um=17,amplitude_slm=dict(pixel_pitch_um=8,center_xy=[0,0]))
        with self.assertRaises(ValueError):aperture(c)

if __name__=='__main__':unittest.main()
