import unittest
from types import SimpleNamespace
import torch
from torch import nn
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_joint_optical import forward_with_visual
from LightGenV2.tasks.t12_text_to_image.scene_frontend_rebind import binding_metadata


class Visual(nn.Module):
    def __init__(self):
        super().__init__();self.frozen=nn.Linear(4,4).requires_grad_(False);self.macro=nn.Linear(4,4)
    def forward(self,x):return (self.macro(self.frozen(x)),)


class CoupledModel(nn.Module):
    def __init__(self):
        super().__init__();self.visual=Visual();self.language=nn.Linear(4,4).requires_grad_(False)
    def forward(self,pixels,**unused):return SimpleNamespace(last_hidden_state=self.language(self.visual(pixels)[0]))


class RawJointOptical(unittest.TestCase):
    def test_binding_rejects_unrelated_or_TEST_trained_frontends(self):
        state=dict(optical_frontend=dict(checkpoint_sha256='old'))
        report=dict(complete=True,only_existing_optical_macros_trainable=True,parent_frontend_sha256='old',
            head_UNet_VAE_retained_Qwen_frozen=True,TEST_opened=False,DEV_or_TEST_forward_performed=False)
        self.assertEqual(binding_metadata(state,report,'new')['checkpoint_sha256'],'new')
        self.assertEqual(state['optical_frontend']['checkpoint_sha256'],'old')
        for changed in [dict(parent_frontend_sha256='another'),dict(TEST_opened=True),dict(DEV_or_TEST_forward_performed=True)]:
            with self.assertRaises(ValueError):binding_metadata(state,dict(report,**changed),'new')

    def test_language_condition_backpropagates_into_actual_visual_macro(self):
        torch.manual_seed(1042);model=CoupledModel();inputs=dict(pixels=torch.randn(1,3,4))
        h,v=forward_with_visual(model,inputs);h.square().mean().backward()
        self.assertGreater(model.visual.macro.weight.grad.abs().sum().item(),0)
        self.assertIsNone(model.visual.frozen.weight.grad);self.assertIsNone(model.language.weight.grad)
        with torch.no_grad():
            old=model(**inputs).last_hidden_state;actual,unused=forward_with_visual(model,inputs)
        torch.testing.assert_close(old,actual,rtol=0,atol=0)
        self.assertEqual(len(model.visual._forward_hooks),0)


if __name__=='__main__':unittest.main()
