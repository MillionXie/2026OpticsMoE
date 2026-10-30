import unittest
import torch
from diffusers import EulerDiscreteScheduler
from ..scene_joint_cached_rollout import joint_rollout,condition_anchor


class TinyHead(torch.nn.Module):
    def __init__(self):super().__init__();self.linear=torch.nn.Linear(2,1)
    def forward(self,hidden,visual):
        x=self.linear(hidden.mean(1)+visual.mean(1))
        return dict(text=x[:,None].expand(-1,2,8),pooled=x.expand(-1,4),image=x[:,None].expand(-1,3,4))


class TinyUNet(torch.nn.Module):
    def __init__(self):super().__init__();self.weights=torch.nn.Parameter(torch.tensor([.02,.03,.04]))
    def forward(self,x,t,encoder_hidden_states,**kwargs):
        class Output:pass
        result=Output();index={901.:0,501.:1,101.:2}[float(t)]
        result.sample=x*self.weights[index]+encoder_hidden_states.mean((1,2))[:,None,None,None]*.02
        return result


class JointGradients(unittest.TestCase):
    def test_readout_and_all_internal_steps_receive_actual_gradients(self):
        torch.manual_seed(1042);head=TinyHead();unet=TinyUNet()
        feature=dict(hidden=torch.randn(1,3,2),visual=torch.randn(1,2,2));initial=torch.randn(1,4,8,8,dtype=torch.bfloat16)
        state=dict(negative_text=torch.zeros(1,2,8),negative_pooled=torch.zeros(1,4),negative_image=torch.zeros(1,3,4))
        scheduler=EulerDiscreteScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',prediction_type='epsilon')
        end,points,conditions=joint_rollout(scheduler,unet,head,feature,initial,state,[901.,501.,101.],gradient=True)
        end.float().square().mean().backward()
        self.assertTrue((unet.weights.grad.abs()>0).all());self.assertGreater(float(head.linear.weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(head.linear.weight.grad).all());self.assertEqual(set(points),{1,2,3})
        self.assertIsNone(feature['hidden'].grad);self.assertIsNone(feature['visual'].grad)
        reference={n:v.detach().clone()for n,v in conditions.items()}
        self.assertEqual(float(condition_anchor(conditions,reference).detach()),0)


if __name__=='__main__':unittest.main()
