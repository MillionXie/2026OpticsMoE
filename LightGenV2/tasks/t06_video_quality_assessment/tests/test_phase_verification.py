import unittest
import numpy as np
from LightGenV2.tasks.t06_video_quality_assessment.lab_phase_verification import assess,compare
from LightGenV2.tasks.t06_video_quality_assessment.lab_exposure_session import retained_stages
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import STAGES

class OpticalVerificationTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(12);self.a=rng.uniform(20,180,(40,40));self.b=rng.uniform(20,180,(40,40))
 def test_switch_and_repeat(self):
  self.assertTrue(assess([self.a,self.b,self.a+1,self.b+1])['passed'])
 def test_stuck_phase_fails(self):
  self.assertFalse(assess([self.a,self.a,self.a,self.a])['passed'])
 def test_nonrepeatable_fails(self):
  self.assertFalse(assess([self.a,self.b,self.b,self.a])['passed'])
 def test_dark_and_clipped_fail(self):
  for v in (np.zeros_like(self.a),np.full_like(self.a,255)):
   with self.assertRaises(ValueError):compare(v,self.a)
 def test_prefix_excludes_language(self):
  self.assertEqual(retained_stages(list(STAGES),3),list(STAGES[:3]))
  self.assertEqual(retained_stages(list(STAGES[:5])),list(STAGES[:5]))
  for n in (0,6,7):
   with self.assertRaises(ValueError):retained_stages(list(STAGES[:5]),n)
  with self.assertRaises(ValueError):retained_stages([STAGES[1]],1)
