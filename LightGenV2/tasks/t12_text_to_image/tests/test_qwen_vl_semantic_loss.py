import unittest
import torch
from ..qwen_vl_semantic_loss import CaptionImageLoss,product_caption


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.randn(3,8))
    def encode_image(self,image):
        return image.mean((2,3))@self.weight


class CaptionLossTests(unittest.TestCase):
    def test_generic_edit_semantics_do_not_reintroduce_unrequested_target_attributes(self):
        row={'render_protocol':'controlled_room_edit_v1','mode':'object','category':'pillow',
             'requested_family':'striped','target':{'sample_id':'SKU-specific'}}
        descriptions={'SKU-specific':'a red silk pillow with thin golden vertical stripes'}
        self.assertEqual(product_caption(row,descriptions),'a striped cushion')
        row['mode']='background'
        self.assertEqual(product_caption(row,descriptions),descriptions['SKU-specific'])
        row['mode']='object';row['render_protocol']='rich_product_scene_v1'
        self.assertEqual(product_caption(row,descriptions),descriptions['SKU-specific'])

    def test_frozen_teacher_keeps_image_gradient_without_any_gt_image(self):
        model=CaptionImageLoss(Encoder());image=torch.zeros(2,3,256,256,requires_grad=True)
        loss=model(image,torch.randn(2,8));loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all());self.assertGreater(float(image.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))


if __name__=='__main__':
    unittest.main()
