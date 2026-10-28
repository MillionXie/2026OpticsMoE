import unittest
import torch
from torch import nn

from LightGenV2.tasks.t12_text_to_image.qwen_vl_language_adapt import NativeLanguageTail,inject_language_lora,install_in_full_qwen,frontend_adapter_digest,verify_frontend_feature_report,TARGETS


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        for name in TARGETS:setattr(self,name,nn.Linear(8,8))
    def forward(self,x,**kwargs):
        for name in TARGETS:x=x+.01*getattr(self,name)(x)
        return x


class LanguageAdaptTests(unittest.TestCase):
    def test_adapted_terminal_cache_must_match_selected_frontend_weights(self):
        state={'schema':'qwen_spatial_language_lora_v1','qwen_language_lora':{'a':torch.ones(2,2)}}
        digest=frontend_adapter_digest(state)
        verify_frontend_feature_report({'qwen_language_lora_state_sha256':digest},state)
        state['qwen_language_lora']['a'][0,0]+=1
        with self.assertRaises(ValueError):verify_frontend_feature_report({'qwen_language_lora_state_sha256':digest},state)
        with self.assertRaises(ValueError):verify_frontend_feature_report({},state)
        verify_frontend_feature_report({}, {'schema':'qwen_spatial_v1'})
    def test_zero_adapter_identity_gradients_and_exact_reload(self):
        context={n:{'position_embeddings':(torch.ones(1,n,4),torch.zeros(1,n,4)),
                    'attention_mask':None,'position_ids':torch.arange(n)[None],
                    'cache_position':torch.arange(n)} for n in (3,5)}
        tail=NativeLanguageTail([Block() for _ in range(4)],nn.LayerNorm(8),context)
        h=torch.randn(2,5,8);valid=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
        before=tail(h,valid).detach();inject_language_lora(tail,2)
        after=tail(h,valid);torch.testing.assert_close(after,before,rtol=0,atol=0)
        after.square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in tail.named_parameters() if '.lora_' in n and p.grad is not None),0)
        state=tail.adapter_state()
        with torch.no_grad():
            for n,p in tail.named_parameters():
                if '.lora_B.' in n:p.add_(.02)
        self.assertFalse(torch.equal(tail(h,valid),before))
        tail.load_adapters(state);torch.testing.assert_close(tail(h,valid),before,rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for n,p in tail.named_parameters() if '.lora_' not in n))

    def test_deployment_installs_into_the_actual_original_language_layers(self):
        lm=nn.Module();lm.layers=nn.ModuleList([nn.Identity() for _ in range(24)]+[Block() for _ in range(4)]);lm.norm=nn.LayerNorm(8)
        full=nn.Module();full.model=nn.Module();full.model.language_model=lm
        tail=NativeLanguageTail(list(lm.layers[24:28]),lm.norm,{})
        inject_language_lora(tail,2);state=tail.adapter_state()
        with torch.no_grad():
            for n,p in tail.named_parameters():
                if '.lora_B.' in n:p.fill_(.02)
        state=tail.adapter_state()
        # A fresh full graph: installing a temporary tail wrapper must mutate
        # the original layer objects used by actual deployment.
        lm2=nn.Module();lm2.layers=nn.ModuleList([nn.Identity() for _ in range(24)]+[Block() for _ in range(4)]);lm2.norm=nn.LayerNorm(8)
        fresh=nn.Module();fresh.model=nn.Module();fresh.model.language_model=lm2
        original=lm2.layers[24]
        install_in_full_qwen(fresh,{'schema':'qwen_spatial_language_lora_v1','config':{'qwen_lora_rank':2},'qwen_language_lora':state})
        self.assertIs(original,lm2.layers[24]);self.assertEqual(len(lm2.layers),28)
        self.assertTrue(torch.equal(lm2.layers[24].q_proj.lora_B['default'].weight,state['layers.0.q_proj.lora_B.default.weight']))
        self.assertFalse(any(p.requires_grad for p in fresh.parameters()))
