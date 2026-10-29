import unittest
import torch
from diffusers import EulerDiscreteScheduler
from LightGenV2.tasks.t12_text_to_image.scene_tail_distill import collapsed_epsilon


class EulerTailContract(unittest.TestCase):
    def test_collapsed_velocity_reaches_same_endpoint(self):
        start=torch.tensor([.7,-1.1]);middle=start+torch.tensor([.2,.5])*(2.-4.)
        end=middle+torch.tensor([-.1,.8])*(0.-2.)
        effective=collapsed_epsilon(start,end,4.,0.)
        torch.testing.assert_close(start+effective*(0.-4.),end)

    def test_descending_noise_intervals_required(self):
        with self.assertRaises(ValueError):collapsed_epsilon(torch.ones(1),torch.zeros(1),1.,1.)
        with self.assertRaises(ValueError):collapsed_epsilon(torch.ones(1),torch.zeros(1),0.,1.)

    def test_six_step_grid_exact_subset_of_leading_teacher(self):
        teacher=EulerDiscreteScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',
            timestep_spacing='leading',steps_offset=1,prediction_type='epsilon')
        teacher.set_timesteps(12)
        student=EulerDiscreteScheduler.from_config(teacher.config)
        student.set_timesteps(timesteps=teacher.timesteps[::2].tolist())
        torch.testing.assert_close(student.timesteps,teacher.timesteps[::2])
        torch.testing.assert_close(student.sigmas,teacher.sigmas[::2])


if __name__=='__main__':unittest.main()
