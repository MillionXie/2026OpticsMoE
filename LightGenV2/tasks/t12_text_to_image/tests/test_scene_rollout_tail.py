import unittest
import torch
from diffusers import EulerDiscreteScheduler
from LightGenV2.tasks.t12_text_to_image.scene_rollout_tail import rollout,objective


class ToyUNet(torch.nn.Module):
    def __init__(self):super().__init__();self.weights=torch.nn.Parameter(torch.tensor([.02,.03,.04]))
    def forward(self,x,t,**kwargs):
        class Result:pass
        index={901.:0,501.:1,101.:2}[float(t)];out=Result();out.sample=x*self.weights[index]+.2;return out


class FullRollout(unittest.TestCase):
    def test_checkpoint_matches_plain_forward_and_all_steps_receive_gradients(self):
        grid=[901.,501.,101.];model=ToyUNet();z=torch.randn(1,4,8,8,dtype=torch.bfloat16)
        cond={'text':torch.ones(1,2,8),'pooled':torch.ones(1,4),'image':torch.ones(1,3,4)}
        state={'negative_text':torch.zeros(1,2,8),'negative_pooled':torch.zeros(1,4),'negative_image':torch.zeros(1,3,4)}
        scheduler=EulerDiscreteScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',prediction_type='epsilon')
        ordinary,_=rollout(scheduler,model,z,cond,state,grid);ordinary.float().square().mean().backward();before=model.weights.grad.clone();model.zero_grad()
        checked,points=rollout(scheduler,model,z,cond,state,grid,gradient=True);checked.float().square().mean().backward()
        torch.testing.assert_close(ordinary,checked,rtol=0,atol=0);torch.testing.assert_close(before,model.weights.grad,rtol=0,atol=0)
        self.assertTrue((model.weights.grad.abs()>0).all());self.assertEqual(set(points),{1,2,3})
        with torch.no_grad():teacher,_=rollout(scheduler,model,z,cond,state,grid)
        self.assertFalse(teacher.requires_grad)

    def test_endpoint_mapping_and_terminal_loss(self):
        x=torch.ones(1,4,2,2,requires_grad=True);y=torch.zeros_like(x)
        loss,terminal,anchor=objective(x,y,{1:x,2:x},{2:y,5:y},[0,2,5])
        self.assertAlmostEqual(float(loss),1.1,places=5);loss.backward();self.assertGreater(x.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
