import unittest
import torch
from diffusers import EulerDiscreteScheduler
from LightGenV2.tasks.t12_text_to_image.scene_onpolicy_tail import teacher_edges,local_teacher_endpoint


class ConstantUNet(torch.nn.Module):
    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
    def forward(self,x,t,**kwargs):
        class Result:pass
        out=Result();out.sample=torch.ones_like(x)*self.p;return out


class OnpolicyTeacher(unittest.TestCase):
    def test_teacher_interval_starts_at_actual_middle_sigma(self):
        scheduler=EulerDiscreteScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',timestep_spacing='leading',steps_offset=1,prediction_type='epsilon');scheduler.set_timesteps(30)
        model=ConstantUNet();start=torch.randn(1,4,8,8,dtype=torch.bfloat16,requires_grad=True)
        state={'negative_text':torch.zeros(1,2,8),'negative_pooled':torch.zeros(1,4),'negative_image':torch.zeros(1,3,4),'SDXL_conditioning_size':512}
        conditions={'text':torch.ones(1,2,8),'pooled':torch.ones(1,4),'image':torch.ones(1,3,4)}
        for begin,end in [(7,10),(20,23),(7,10)]:
            actual=local_teacher_endpoint(scheduler,model,start,conditions,state,begin,end);expected=start.detach().clone()
            for j in range(begin,end):expected=(expected.float()+float(scheduler.sigmas[j+1]-scheduler.sigmas[j])).bfloat16()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertFalse(actual.requires_grad);self.assertIsNone(start.grad);self.assertIsNone(model.p.grad)
            self.assertEqual(scheduler.step_index,end)
    def test_nonuniform_native_endpoint_mapping_and_rejection(self):
        full=[90.,80.,70.,60.,50.,40.];self.assertEqual(teacher_edges(full,[90.,70.,60.]),[0,2,3,6])
        for invalid in [[80.,60.],[90.,75.],[90.,90.],[90.,60.,70.],[]]:
            with self.assertRaises(ValueError):teacher_edges(full,invalid)


if __name__=='__main__':unittest.main()
