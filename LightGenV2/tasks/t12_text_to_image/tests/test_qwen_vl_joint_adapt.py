import unittest
import torch
from torch import nn
from LightGenV2.tasks.t12_text_to_image.qwen_vl_joint_adapt import install_joint_lora,joint_adapter_state,load_joint_adapters,RawQwenHead
from LightGenV2.tasks.t12_text_to_image.tests.test_qwen_vl_language_adapt import Block


class VisionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.attn=nn.Module();self.mlp=nn.Module()
        self.attn.qkv=nn.Linear(8,8);self.attn.proj=nn.Linear(8,8)
        self.mlp.linear_fc1=nn.Linear(8,8);self.mlp.linear_fc2=nn.Linear(8,8)
    def forward(self,x):return x+.01*self.attn.proj(self.attn.qkv(x))+.01*self.mlp.linear_fc2(self.mlp.linear_fc1(x))


class JointAdapterTests(unittest.TestCase):
    def test_raw_features_whitelist_inputs_and_preserve_gradients_instead_of_cached_states(self):
        from unittest.mock import patch
        q=nn.Module();q.gain=nn.Parameter(torch.tensor(1.))
        model=RawQwenHead(nn.Identity(),q,None,{})
        payload={'reference':torch.zeros(2,3,16,16,dtype=torch.uint8),'target':torch.ones(2,3,16,16),
                 'rows':[{'prompt':'first','teacher_caption':'SECRET'},{'prompt':'second','target':'SECRET'}],
                 'hidden':[torch.full((3,8),float('nan'))]*2}
        def actual(qwen,processor,inputs,i,layout,device):
            self.assertEqual(set(inputs),{'reference','rows'})
            self.assertTrue(all(set(row)=={'prompt'} for row in inputs['rows']))
            return qwen.gain*torch.ones(3+i,8)
        with patch('LightGenV2.tasks.t12_text_to_image.qwen_vl_hybrid_run.raw_hidden',side_effect=actual):
            h,valid=model.raw_features(payload,[1,0],torch.device('cpu'))
        self.assertTrue(torch.isfinite(h).all());h.sum().backward();self.assertGreater(float(q.gain.grad),0)
    def fake(self):
        q=nn.Module();q.model=nn.Module();q.model.visual=nn.Module();q.model.language_model=nn.Module()
        q.model.visual.blocks=nn.ModuleList([VisionBlock() for _ in range(24)])
        q.model.language_model.layers=nn.ModuleList([Block() for _ in range(28)])
        return q
    def forward(self,q,x):
        for block in q.model.visual.blocks:x=block(x)
        for block in q.model.language_model.layers:x=block(x)
        return x
    def test_zero_identity_both_modalities_real_gradients_and_reload(self):
        q=self.fake();x=torch.randn(2,4,8);before=self.forward(q,x).detach()
        cfg={'qwen_vision_lora_rank':2,'qwen_language_lora_rank':2};install_joint_lora(q,cfg,train=True)
        after=self.forward(q,x);torch.testing.assert_close(after,before,rtol=0,atol=0);after.square().mean().backward()
        for branch in ('visual','language_model'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in q.named_parameters() if branch in n and p.requires_grad and p.grad is not None),0)
        state=joint_adapter_state(q)
        with torch.no_grad():
            for n,p in q.named_parameters():
                if '.lora_B.' in n:p.add_(.1)
        self.assertFalse(torch.equal(self.forward(q,x),before));load_joint_adapters(q,state)
        torch.testing.assert_close(self.forward(q,x),before,rtol=0,atol=0)
        fresh=self.fake();install_joint_lora(fresh,cfg,state,train=False)
        self.assertFalse(any(p.requires_grad for p in fresh.parameters()))
