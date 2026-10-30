import unittest
import torch
from diffusers import EulerDiscreteScheduler
from ..scene_cached_rollout_recovery import rollout_from_scaled
from ..scene_rollout_tail import rollout


class ToyUNet(torch.nn.Module):
    def __init__(self):super().__init__();self.weights=torch.nn.Parameter(torch.tensor([.02,.03,.04]))
    def forward(self,x,t,**kwargs):
        class Output:pass
        out=Output();out.sample=x*self.weights[{901.:0,501.:1,101.:2}[float(t)]]+.2;return out


class ExactCachedStart(unittest.TestCase):
    def test_exact_scaled_initial_state_matches_native_rollout_and_all_step_gradients(self):
        torch.manual_seed(1042);model=ToyUNet();grid=[901.,501.,101.];z=torch.randn(1,4,8,8,dtype=torch.bfloat16)
        cond=dict(text=torch.ones(1,2,8),pooled=torch.ones(1,4),image=torch.ones(1,3,4))
        state=dict(negative_text=torch.zeros(1,2,8),negative_pooled=torch.zeros(1,4),negative_image=torch.zeros(1,3,4))
        scheduler=EulerDiscreteScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',prediction_type='epsilon')
        expected,_=rollout(scheduler,model,z,cond,state,grid);expected.float().square().mean().backward();gradient=model.weights.grad.clone();model.zero_grad()
        scheduler.set_timesteps(timesteps=grid);scaled=z*scheduler.init_noise_sigma
        actual,points=rollout_from_scaled(scheduler,model,scaled,cond,state,grid,gradient=True);actual.float().square().mean().backward()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(model.weights.grad,gradient,rtol=0,atol=0)
        self.assertTrue((model.weights.grad.abs()>0).all());self.assertEqual(set(points),{1,2,3})


if __name__=='__main__':unittest.main()
