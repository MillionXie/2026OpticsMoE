"""Numerical-contract checks; no model assets, generation or GPU required."""
from types import SimpleNamespace
import unittest
import torch
from .. import scene_precision as p


class PrecisionContract(unittest.TestCase):
    def test_default_condition_has_identical_values_and_storage_when_already_BF16(self):
        x=torch.randn(2,3,7).bfloat16();out=p.condition(x,SimpleNamespace())
        self.assertTrue(torch.equal(x,out));self.assertEqual(x.data_ptr(),out.data_ptr())
        x=torch.randn(2,3,7);self.assertTrue(torch.equal(x.bfloat16(),p.condition(x,SimpleNamespace())))

    def test_optional_condition_preserves_legacy_BF16_values(self):
        x=torch.randn(2,3,7);out=p.condition(x,SimpleNamespace(unet_precision='fp16'))
        self.assertEqual(out.dtype,torch.float16)
        self.assertTrue(torch.equal(out.float(),x.bfloat16().float()))

    def test_vae_bridge_only_changes_last_latent(self):
        pipe=SimpleNamespace(num_timesteps=12,vae=SimpleNamespace(dtype=torch.bfloat16))
        x=torch.randn(1,4,8,8).half();values=dict(latents=x,other=object())
        first=p.final_vae_cast(pipe,0,None,values);last=p.final_vae_cast(pipe,11,None,values)
        self.assertIs(first['latents'],x);self.assertIs(last['other'],values['other'])
        self.assertEqual(last['latents'].dtype,torch.bfloat16)
        self.assertTrue(torch.equal(last['latents'],x.bfloat16()));self.assertIs(values['latents'],x)

    def test_default_has_no_extra_callback_or_graph(self):
        self.assertEqual(p.generation_kwargs(SimpleNamespace(),None),{})
        self.assertFalse(p.descriptor(SimpleNamespace())['last_latent_to_VAE_cast'])
        self.assertEqual(p.dtype(SimpleNamespace()),torch.bfloat16)

    def test_only_explicit_unet_change_is_supported(self):
        args=SimpleNamespace(unet_precision='fp16')
        good=SimpleNamespace(unet=SimpleNamespace(dtype=torch.float16),vae=SimpleNamespace(dtype=torch.bfloat16))
        self.assertEqual(p.generation_kwargs(args,good)['callback_on_step_end_tensor_inputs'],['latents'])
        self.assertEqual(p.descriptor(args)['VAE'],'bf16')
        good.vae.dtype=torch.float16
        with self.assertRaises(ValueError):p.generation_kwargs(args,good)
        with self.assertRaises(ValueError):p.name(SimpleNamespace(unet_precision='fp32'))
