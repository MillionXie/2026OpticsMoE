import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_conditioning import QwenSDXLConditions
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_paired_train import TRAINReferenceContrast


class JointImageContext(unittest.TestCase):
    def head(self,flag=False):
        torch.manual_seed(42)
        return QwenSDXLConditions(input_width=8,width=8,text_width=12,image_width=16,pooled_width=10,joint_image_context=flag).eval()

    def test_default_is_exact_old_arithmetic_and_same_parameters(self):
        head=self.head();h=torch.randn(1,5,8);v=torch.randn(1,4,8)
        text,q=head.text(head.project(h));image,_=head.image(head.project(v))
        actual=head(h,v)
        for name,wanted in [('text',text),('image',image),('pooled',head.pooled(q.mean(1)))]:
            torch.testing.assert_close(actual[name],wanted,rtol=0,atol=0)
        other=self.head(True)
        self.assertEqual(set(head.state_dict()),set(other.state_dict()))
        self.assertEqual(sum(p.numel() for p in head.parameters()),sum(p.numel() for p in other.parameters()))

    def test_only_new_image_readout_receives_language_gradient(self):
        for flag in [False,True]:
            h=torch.randn(1,5,8,requires_grad=True);v=torch.randn(1,4,8,requires_grad=True)
            self.head(flag)(h,v)['image'].square().mean().backward()
            self.assertGreater(v.grad.abs().sum().item(),0)
            if flag:self.assertGreater(h.grad.abs().sum().item(),0)
            else:self.assertIsNone(h.grad)

    def test_language_padding_is_not_visible_to_image_queries(self):
        head=self.head(True);h=torch.randn(1,6,8);v=torch.randn(1,4,8);padding=torch.tensor([[False]*4+[True]*2])
        changed=h.clone();changed[:,4:]=999
        torch.testing.assert_close(head(h,v,padding)['image'],head(changed,v,padding)['image'])

    def test_reference_contrast_has_TRAIN_only_gradient(self):
        rows=[dict(split='train',input_key=str(i),reference_sha256=str(i),source_product_id=str(i)) for i in range(3)]
        target={str(i):{'image':torch.randn(3,4)} for i in range(3)}
        bank=TRAINReferenceContrast({'records':rows,'TRAIN_electronic_conditions':target},device='cpu')
        image=torch.randn(1,3,4,requires_grad=True);loss=bank.loss({'image':image},rows[0]);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(image.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):bank.loss({'image':image},dict(rows[0],split='dev'))
        with self.assertRaises(ValueError):TRAINReferenceContrast({'records':rows,'TRAIN_electronic_conditions':{**target,'dev':{'image':torch.randn(3,4)}}},device='cpu')


if __name__=='__main__':unittest.main()
