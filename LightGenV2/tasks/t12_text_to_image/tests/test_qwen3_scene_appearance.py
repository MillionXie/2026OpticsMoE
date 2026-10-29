import unittest
import torch
from ..qwen3_scene_conditioning import QwenSDXLConditions
from ..qwen3_scene_appearance_probe import freeze_appearance,text_loss


class AppearanceReadoutTests(unittest.TestCase):
    def test_real_text_gradient_cannot_modify_appearance_reader(self):
        torch.manual_seed(7);torch.set_num_threads(2)
        head=freeze_appearance(QwenSDXLConditions(input_width=32,width=16,text_width=12,image_width=8,pooled_width=8))
        frozen={n:p.detach().clone() for n,p in head.named_parameters() if not p.requires_grad}
        pred=head(torch.randn(2,9,32),torch.randn(2,4,32))
        target={'text':torch.randn(2,77,12),'pooled':torch.randn(2,8)}
        opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-3)
        loss=text_loss(pred,target,[6,12]);loss.backward();self.assertTrue(torch.isfinite(loss));opt.step()
        for n,p in head.named_parameters():
            if n in frozen:self.assertIsNone(p.grad);torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
        for module in [head.text,head.pooled]:self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)


if __name__=='__main__':unittest.main()
