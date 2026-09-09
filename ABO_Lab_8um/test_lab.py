import unittest
import numpy as np
from common import config
from patterns import raster,amplitude
from hardware import geometry,canonical

class PhysicalContractTests(unittest.TestCase):
    def setUp(self): self.c,_=config('lab.json')
    def test_width(self):
        a=raster(np.full((478,478),255),self.c['amplitude_slm'],self.c)
        y,x=np.nonzero(a); self.assertEqual((x.max()-x.min()+1,y.max()-y.min()+1),(1016,1016))
        self.assertEqual(a.shape,(1080,1920)); self.assertEqual(float(x.mean()),959.5)
    def test_phase_no_wrap_interpolation(self):
        p=np.zeros((478,478),np.float32); p[:,239:]=2*np.pi-.01
        x=raster(p,self.c['phase_slm'],self.c,phase=True)
        self.assertEqual(set(np.unique(x)),{0,255})
    def test_zero_amplitude(self):
        x,meta=amplitude(np.zeros((478,478)),self.c); self.assertFalse(x.any())
    def test_outside_panel(self):
        self.c['amplitude_slm']['center_xy']=[100,100]
        with self.assertRaises(ValueError): raster(np.zeros((478,478)),self.c['amplitude_slm'],self.c)
    def set_corners(self,points):
        self.c['logical_corners_full_sensor_xy']=dict(zip(('top_left','top_right','bottom_right','bottom_left'),points))
        self.c['geometry_confirmed']=True; self.c['capture_input_range']=[0,255]
    def test_horizontal_mirror(self):
        self.set_corners([[478,0],[0,0],[0,478],[478,478]])
        raw=np.tile(np.arange(479,dtype=np.float32),(479,1))/2
        rect=canonical(raw,self.c); self.assertGreater(float(rect[:,0].mean()),float(rect[:,-1].mean()))
    def test_crossed_points_fail(self):
        self.set_corners([[478,0],[0,0],[478,478],[0,478]])
        with self.assertRaises(ValueError): geometry(self.c,(500,500))
    def test_fixed_scale(self):
        self.set_corners([[0,0],[478,0],[478,478],[0,478]])
        a=canonical(np.full((479,479),100,np.uint8),self.c)
        b=canonical(np.full((479,479),200,np.uint8),self.c)
        self.assertAlmostEqual(float(a[10:-10,10:-10].mean()),100)
        self.assertAlmostEqual(float(b[10:-10,10:-10].mean()),200)

if __name__=='__main__': unittest.main()
