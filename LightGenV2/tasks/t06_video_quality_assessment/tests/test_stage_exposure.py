import copy,unittest
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import stage_config,effective_stage_identity

class StageExposureTests(unittest.TestCase):
 def test_resolution_and_inheritance(self):
  a={'camera':{'exposure_us':400},'detector_intensity_scale':{'vision_router':1/255,'language_router':1/255}}
  b=copy.deepcopy(a);b['camera_exposure_us_by_stage']={'language_router':1600};b['detector_intensity_scale']['language_router']/=4
  self.assertEqual(effective_stage_identity(a,'vision_router'),effective_stage_identity(b,'vision_router'))
  self.assertNotEqual(effective_stage_identity(a,'language_router'),effective_stage_identity(b,'language_router'))
  self.assertEqual(stage_config(b,'language_router')['camera']['exposure_us'],1600)
  self.assertEqual(b['camera']['exposure_us'],400)
  b['camera']['gain']='changed';self.assertNotEqual(effective_stage_identity(a,'vision_router'),effective_stage_identity(b,'vision_router'))
 def test_invalid_exposure(self):
  for v in [0,-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):stage_config({'camera':{'exposure_us':400},'camera_exposure_us_by_stage':{'language_router':v}},'language_router')
  with self.assertRaises(ValueError):stage_config({'camera':{'exposure_us':400},'camera_exposure_us_by_stage':{'bad':1600}},'language_router')
