import unittest
import torch
from ..qwen3_scene_conditioning import QwenSDXLConditions,alignment_loss,qwen_asset_manifest


class NativeConditionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4);torch.set_num_threads(2)
        self.model=QwenSDXLConditions(input_width=32,width=16,text_width=12,image_width=8,pooled_width=8)

    def test_both_modalities_have_real_gradients(self):
        h=torch.randn(2,7,32,requires_grad=True);v=torch.randn(2,4,32,requires_grad=True)
        out=self.model(h,v);self.assertEqual(out['text'].shape,(2,77,12));self.assertEqual(out['image'].shape,(2,257,8))
        target={k:torch.randn_like(x) for k,x in out.items()};loss,_=alignment_loss(out,target);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(h.grad.abs().sum().item(),0);self.assertGreater(v.grad.abs().sum().item(),0)
        for module in [self.model.text,self.model.image,self.model.pooled]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)

    def test_padded_hidden_cannot_change_native_conditions(self):
        self.model.eval();h=torch.randn(1,7,32);v=torch.randn(1,4,32);mask=torch.tensor([[False]*5+[True]*2])
        a=self.model(h,v,mask);h[:,5:]=1000;b=self.model(h,v,mask)
        for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)

    def test_perfect_interface_has_zero_loss(self):
        target={'text':torch.randn(2,77,12),'image':torch.randn(2,257,8),'pooled':torch.randn(2,8)}
        loss,_=alignment_loss(target,target);self.assertLess(abs(loss.item()),1e-6)

    def test_hf_cache_parent_does_not_hide_actual_model_assets(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'.cache'/'hub'/'snapshot';root.mkdir(parents=True)
            (root/'model.safetensors').write_bytes(b'asset identity fixture')
            (root/'.cache').mkdir();(root/'.cache'/'metadata.json').write_text('{}')
            self.assertEqual(set(qwen_asset_manifest(root)),{'model.safetensors'})


if __name__=='__main__':unittest.main()
