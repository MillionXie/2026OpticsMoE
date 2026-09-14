import unittest
import numpy as np
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import raster
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import PINS,STAGES

class TestLabBench(unittest.TestCase):
 def test_pins_distinct(self):
  self.assertNotEqual(PINS['spatial']['sha256'],PINS['temporal']['sha256'])
  self.assertEqual(PINS['temporal']['videos_per_field'],16);self.assertEqual(len(STAGES),6)
 def test_phase_inversion(self):
  c=dict(pixel_pitch_um=8,size_wh=[1920,1200],center_xy=[960,600],flip_vertical=True,flip_horizontal=True,gray_encoding='inverted_255_minus_g')
  a=raster(np.zeros((478,478)),c,'phase');self.assertEqual(a.shape,(1200,1920));self.assertTrue((a==255).all())
  b=raster(np.full((478,478),np.pi),c,'phase');self.assertEqual(int(b[600,960]),127)
 def test_amplitude_extent(self):
  c=dict(pixel_pitch_um=8,expected_resolution_wh=[1920,1080],center_xy=[960,540])
  a=raster(np.ones((478,478)),c,'amplitude');self.assertEqual(int((a==255).sum()),1016**2)
 def test_reject_blank(self):
  with self.assertRaises(ValueError):raster(np.zeros((478,478)),{},'amplitude')

if __name__=='__main__':unittest.main()
