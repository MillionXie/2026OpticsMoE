import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from LightGenV2.tasks.t12_text_to_image.scene_fast_tail import checked_tail,REPO,REVISION,fuse_task_adapter,tail_configuration
from LightGenV2.tasks.t12_text_to_image.scene_paired_render import frozen_test_protocol
from LightGenV2.tasks.t12_text_to_image.tests.test_scene_frozen_test import FrozenSceneTEST


class FastTailContract(unittest.TestCase):
    def test_LCM_guidance_is_explicit_and_does_not_mutate_asset_identity(self):
        original=dict(kind='verified-tail',CFG=1.,weight_sha256='pinned')
        with patch('LightGenV2.tasks.t12_text_to_image.scene_fast_tail.checked_tail',return_value=original):
            self.assertEqual(tail_configuration(Path('asset')),original)
            self.assertEqual(tail_configuration(Path('asset'),2.)['CFG'],2.)
            self.assertEqual(original['CFG'],1.)
            with self.assertRaises(ValueError):tail_configuration(Path('asset'),7.5)

    def test_task_merge_retains_output_and_rejects_double_or_unknown_adapter(self):
        from peft import LoraConfig,inject_adapter_in_model
        from diffusers.utils.peft_utils import recurse_remove_peft_layers
        class Pipe:
            def __init__(self):
                self.unet=nn.Sequential();self.unet.add_module('proj',nn.Linear(4,4))
            def fuse_lora(self,adapter_names,safe_fusing):self.unet.proj.merge(adapter_names=adapter_names,safe_merge=safe_fusing)
            def unload_lora_weights(self):recurse_remove_peft_layers(self.unet)
        torch.manual_seed(1042);pipe=Pipe();original={n:tuple(p.shape) for n,p in pipe.unet.named_parameters()}
        inject_adapter_in_model(LoraConfig(r=2,lora_alpha=2,target_modules=['proj']),pipe.unet,adapter_name='scene_pair')
        with torch.no_grad():pipe.unet.proj.lora_A['scene_pair'].weight.fill_(.1);pipe.unet.proj.lora_B['scene_pair'].weight.fill_(.2)
        x=torch.ones(1,4);before=pipe.unet(x).detach();descriptor=fuse_task_adapter(pipe,original)
        torch.testing.assert_close(pipe.unet(x),before);self.assertEqual(descriptor['task_adapter_raw_parameters'],16)
        self.assertTrue(descriptor['task_LoRA_fused']);self.assertFalse(descriptor['added_deployed_structure'])
        with self.assertRaises(ValueError):fuse_task_adapter(pipe,original)
        other=Pipe();inject_adapter_in_model(LoraConfig(r=2,target_modules=['proj']),other.unet,adapter_name='unverified')
        with self.assertRaises(ValueError):fuse_task_adapter(other,original)

    def test_TEST_task_merge_graph_must_be_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            a,state,ready,frozen=FrozenSceneTEST().fixture(Path(folder));a.fuse_task_lora=True
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)
            frozen['models']['qwen']['task_LoRA_fused']=True;a.frozen_evaluation.write_text(json.dumps(frozen))
            self.assertTrue(frozen_test_protocol(a,state,ready)['descriptor']['task_LoRA_fused'])
            a.fuse_task_lora=False
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)

    def test_pretrained_merge_unload_then_task_adapter_remains_active(self):
        from peft import LoraConfig,inject_adapter_in_model
        from diffusers.utils.peft_utils import recurse_remove_peft_layers
        torch.manual_seed(1042);model=nn.Sequential();model.add_module('proj',nn.Linear(4,4))
        original={n:tuple(p.shape) for n,p in model.named_parameters()}
        inject_adapter_in_model(LoraConfig(r=2,lora_alpha=2,target_modules=['proj']),model,adapter_name='lcm')
        with torch.no_grad():model.proj.lora_A['lcm'].weight.fill_(.1);model.proj.lora_B['lcm'].weight.fill_(.2)
        x=torch.ones(1,4);pretrained=model(x).detach();model.proj.merge(adapter_names=['lcm'],safe_merge=True)
        recurse_remove_peft_layers(model);torch.testing.assert_close(model(x),pretrained)
        self.assertEqual(original,{n:tuple(p.shape) for n,p in model.named_parameters()})
        inject_adapter_in_model(LoraConfig(r=1,lora_alpha=1,target_modules=['proj']),model,adapter_name='task')
        with torch.no_grad():model.proj.lora_A['task'].weight.fill_(.05);model.proj.lora_B['task'].weight.fill_(.3)
        torch.testing.assert_close(model(x),pretrained+torch.full((1,4),.06),atol=1e-6,rtol=1e-6)

    def test_bad_external_model_identity_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);report=dict(complete=True,repo_id=REPO,revision=REVISION,license='openrail++',weight_sha256='wrong')
            (root/'asset_report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):checked_tail(root)

    def test_TEST_cannot_silently_change_to_an_unfrozen_fast_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            a,state,ready,frozen=FrozenSceneTEST().fixture(Path(folder));a.lcm_adapter=Path('verified-asset')
            descriptor=dict(kind='verified-fast-tail',sha256='external-weights')
            with patch('LightGenV2.tasks.t12_text_to_image.scene_fast_tail.checked_tail',return_value=descriptor):
                with self.assertRaises(KeyError):frozen_test_protocol(a,state,ready)
                frozen['models']['qwen']['fast_tail']=descriptor;a.frozen_evaluation.write_text(json.dumps(frozen))
                self.assertEqual(frozen_test_protocol(a,state,ready)['descriptor']['fast_tail'],descriptor)
                a.lcm_adapter=None
                with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)


if __name__=='__main__':unittest.main()
