import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.scene_inference_sampler import descriptor,install
from LightGenV2.tasks.t12_text_to_image.scene_paired_render import frozen_test_protocol
from LightGenV2.tasks.t12_text_to_image.tests.test_scene_frozen_test import FrozenSceneTEST


class InferenceSampler(unittest.TestCase):
    def test_default_keeps_scheduler_and_high_order_keeps_trained_betas(self):
        from diffusers import EulerDiscreteScheduler,DPMSolverMultistepScheduler
        old=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',prediction_type='epsilon')
        pipe=SimpleNamespace(scheduler=old);self.assertIsNone(install(pipe,'original'));self.assertIs(pipe.scheduler,old)
        self.assertEqual(install(pipe,'dpmpp2m-karras'),descriptor('dpmpp2m-karras'))
        self.assertIsInstance(pipe.scheduler,DPMSolverMultistepScheduler);torch.testing.assert_close(pipe.scheduler.betas,old.betas)
        pipe.scheduler.set_timesteps(12);x=torch.randn(1,4,8,8);evaluations=0
        for step in pipe.scheduler.timesteps:
            inp=pipe.scheduler.scale_model_input(x,step);prediction=torch.zeros_like(inp);evaluations+=1
            x=pipe.scheduler.step(prediction,step,x).prev_sample
        self.assertEqual(evaluations,12);self.assertTrue(torch.isfinite(x).all());self.assertFalse(pipe.scheduler.config.thresholding)

    def test_TEST_solver_choice_must_be_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            a,state,ready,frozen=FrozenSceneTEST().fixture(Path(folder));a.sampler='dpmpp2m-karras'
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)
            frozen['models']['qwen']['inference_sampler']=descriptor(a.sampler);a.frozen_evaluation.write_text(json.dumps(frozen))
            self.assertEqual(frozen_test_protocol(a,state,ready)['descriptor']['inference_sampler'],descriptor(a.sampler))
            a.sampler='original'
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)


if __name__=='__main__':unittest.main()
