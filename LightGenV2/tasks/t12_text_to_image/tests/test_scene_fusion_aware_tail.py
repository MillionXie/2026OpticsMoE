import copy
import unittest
import torch
from peft.tuners.lora.layer import Linear
from peft import LoraConfig,get_peft_model
from LightGenV2.tasks.t12_text_to_image.scene_fusion_aware_tail import install,trainer_arguments


def layer():
    torch.manual_seed(1042)
    base=torch.nn.Linear(16,12,bias=True).bfloat16().requires_grad_(False)
    wrapped=get_peft_model(torch.nn.Sequential(base),LoraConfig(r=4,lora_alpha=4,
        target_modules=['0'],lora_dropout=0.,init_lora_weights=False),adapter_name='scene_pair')
    module=next(m for m in wrapped.modules() if isinstance(m,Linear))
    for n,p in module.named_parameters():
        if 'lora_' in n:p.data=p.data.float();p.requires_grad_(True)
    return module


class FusionAware(unittest.TestCase):
    def test_trainer_selection_preserves_original_mode_and_stage_arguments(self):
        args=['--student','a','--steps','600']
        self.assertEqual(trainer_arguments(args),('onpolicy',args))
        stage=['--stage','smoke','--student','a','--student-guidance','7.5']
        self.assertEqual(trainer_arguments(['--trainer','trajectory',*stage]),('trajectory',stage))

    def test_forward_matches_actual_official_safe_merge_and_preserves_parameters(self):
        module=layer();reference=copy.deepcopy(module);reference.merge(safe_merge=True,adapter_names=['scene_pair'])
        saved={n:p.detach().clone() for n,p in module.named_parameters()};ids={n:id(p) for n,p in module.named_parameters()}
        contract=install(torch.nn.Sequential(module));self.assertEqual(contract['added_parameters'],0)
        x=torch.randn(2,5,16,dtype=torch.bfloat16)
        torch.testing.assert_close(module(x),reference(x),rtol=0,atol=0)
        for n,p in module.named_parameters():
            self.assertEqual(id(p),ids[n]);torch.testing.assert_close(p,saved[n],rtol=0,atol=0)
        module(x).float().square().mean().backward()
        self.assertIsNone(module.base_layer.weight.grad);self.assertIsNone(module.base_layer.bias.grad)
        for p in [module.lora_A['scene_pair'].weight,module.lora_B['scene_pair'].weight]:
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):install(torch.nn.Sequential(module))

    def test_teacher_and_base_dtypes_are_guarded(self):
        module=layer().requires_grad_(False)
        with self.assertRaises(ValueError):install(torch.nn.Sequential(module))
        module=layer();module.base_layer.float()
        with self.assertRaises(ValueError):install(torch.nn.Sequential(module))
        with self.assertRaises(ValueError):install(torch.nn.Linear(2,2))


if __name__=='__main__':unittest.main()
