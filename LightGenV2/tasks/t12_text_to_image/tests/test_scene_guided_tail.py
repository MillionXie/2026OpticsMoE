import unittest
from types import SimpleNamespace
import torch
from torch import nn
from ..scene_tail_distill import predict,student_sampling_contract


class ConditionalUNet(nn.Module):
    def forward(self,sample,timestep,encoder_hidden_states,added_cond_kwargs):
        b=len(sample);t=torch.as_tensor(timestep).reshape(-1)
        if len(t)==1:t=t.expand(b)
        assert len(t)==b and len(encoder_hidden_states)==b
        assert all(len(added_cond_kwargs[k])==b for k in ['text_embeds','time_ids'])
        image=added_cond_kwargs['image_embeds'][0];assert len(image)==b
        scalar=encoder_hidden_states.float().mean((1,2))+added_cond_kwargs['text_embeds'].float().mean(1)+image.float().mean((1,2,3))+t.float()/1000
        return SimpleNamespace(sample=(sample.float()*.1+scalar[:,None,None,None]).to(sample.dtype))


class GuidedTail(unittest.TestCase):
    def test_real_guided_operator_preserves_unconditional_branch_and_batched_gradients(self):
        torch.manual_seed(1042);unet=ConditionalUNet();x=torch.randn(2,4,3,3)
        cond={'text':torch.randn(2,3,2,requires_grad=True),'pooled':torch.randn(2,2,requires_grad=True),
            'image':torch.randn(2,5,2,requires_grad=True)}
        state={'negative_text':torch.randn(1,3,2),'negative_pooled':torch.randn(1,2),'negative_image':torch.randn(1,5,2)}
        t=torch.tensor([700.,300.]);sigma=[4.,1.]
        guided=predict(unet,x,t,cond,state,sigma,True)
        individual=torch.cat([predict(unet,x[i:i+1],t[i],{n:v[i:i+1] for n,v in cond.items()},state,sigma[i],True) for i in range(2)])
        torch.testing.assert_close(guided,individual,rtol=0,atol=0)
        negative={'text':state['negative_text'].expand(2,-1,-1),'pooled':state['negative_pooled'].expand(2,-1),
            'image':state['negative_image'].expand(2,-1,-1)}
        u=predict(unet,x,t,negative,state,sigma,False);c=predict(unet,x,t,cond,state,sigma,False)
        torch.testing.assert_close(guided,u+7.5*(c-u),rtol=0,atol=0)
        guided.float().square().mean().backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in cond.values()))
        with self.assertRaises(ValueError):predict(unet,x,t,cond,state,[1.],True)

    def test_guidance_and_effective_interval_targets_have_distinct_sampling_contracts(self):
        targets={'timesteps':[958,859,760,661,562,463,364,265,199,133,67,1],'student_steps':12,'teacher_steps':30}
        old=student_sampling_contract(targets,1.,.25);new=student_sampling_contract(targets,7.5,.25)
        self.assertNotEqual(old['kind'],new['kind']);self.assertEqual(new['CFG'],7.5)
        self.assertEqual(old['timesteps'],new['timesteps']);self.assertEqual(targets['student_steps'],12)
        for args in [(targets,2.,.25),(targets,7.5,1.1),(dict(targets,timesteps=[1,2]),7.5,.25)]:
            with self.assertRaises(ValueError):student_sampling_contract(*args)


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
