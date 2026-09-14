import unittest
from pathlib import Path
import torch
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.lab_runtime import CachedStudent, STAGES, replay, phase_planes, OpticalBoundary
from LightGenV2.tasks.t06_video_quality_assessment.lab_manual_stage import desktop_code
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import stage_config


class LabRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        settings=load_settings(Path(__file__).resolve().parents[1]/'configs/moe_alpha40_sam_batch8_crosssample_20260913.yaml')
        settings.vision_hidden_size=1024
        torch.manual_seed(123)
        cls.model=CachedStudent(settings).eval()
        cls.model.core.set_phase_dropout_active(False)
        cls.batch={'tokens':torch.randn(256,1024),'grid':torch.tensor([[1,16,16]])}

    def test_three_detectors_and_replay(self):
        with torch.inference_mode(): expected=self.model(self.batch)
        actual,tap=replay(self.model,self.batch)
        restored,_=replay(self.model,self.batch,tap.detectors)
        self.assertEqual(tuple(tap.amplitudes),STAGES)
        torch.testing.assert_close(expected,actual,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(expected,restored,atol=1e-4,rtol=1e-4)
        for plane in phase_planes(self.model).values():self.assertEqual(plane.shape,(478,478))

    def test_prefix_validation_and_restore(self):
        with self.assertRaises(ValueError):OpticalBoundary(self.model,{'vision_global':torch.zeros(1,478,478)})
        _,tap=replay(self.model,self.batch,stop_before='vision_expert')
        self.assertEqual(tuple(tap.amplitudes),STAGES[:2])
        _,again=replay(self.model,self.batch)
        self.assertEqual(tuple(again.amplitudes),STAGES)

    def test_generic_coordinator_final_stage(self):
        code=desktop_code('p','b','s','c','vision_global','out',STAGES)
        self.assertNotIn('language_router',code)
        compile(code,'coordinator','exec')
        with self.assertRaises(ValueError):stage_config({'camera':{'exposure_us':400},'camera_exposure_us_by_stage':{'language_router':500}},'vision_router',STAGES)


if __name__=='__main__':unittest.main()
