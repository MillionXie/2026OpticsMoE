import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import config,ROOT
from patterns import raster,amplitude
from hardware import geometry,canonical
import patterns
from common import read

class PhysicalContractTests(unittest.TestCase):
    def setUp(self): self.c,_=config(ROOT/'lab.json')
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
    def test_fresnel_roi_vertices_and_single_corners(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(patterns,'ROOT',Path(tmp)):
            patterns.calibration(self.c)
            d=Path(tmp)/'generated/cal'; m=read(d/'geometry.json')
            np.testing.assert_allclose(m['arrays']['4']['centers_xy'],
                [[451.625,91.625],[1467.375,91.625],[451.625,1107.375],[1467.375,1107.375]])
            from PIL import Image
            for name in ('TL','TR','BL','BR'):
                with Image.open(d/f'P_F_{name}.bmp') as im: self.assertEqual(im.size,(1920,1200))
            from fresnel import four_array
            expected,owner,support,_=four_array(self.c)
            with Image.open(d/'P_F4.bmp') as im: np.testing.assert_array_equal(np.asarray(im),expected)
            for i,name in enumerate(('TL','TR','BL','BR')):
                with Image.open(d/f'P_F_{name}.bmp') as im:
                    np.testing.assert_array_equal(np.asarray(im),np.where((owner==i)&support,expected,0))
            with Image.open(d/'A_ACTIVE.bmp') as im: self.assertEqual(im.size,(1920,1080))
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
