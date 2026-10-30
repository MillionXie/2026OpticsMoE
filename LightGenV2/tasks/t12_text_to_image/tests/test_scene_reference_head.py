import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_conditioning import QwenSDXLConditions
from LightGenV2.tasks.t12_text_to_image.scene_reference_head_train import configure,prefix_loss,reference_rows


class ReferenceHead(unittest.TestCase):
    def test_actual_backward_only_updates_existing_image_queries(self):
        torch.manual_seed(4)
        h=QwenSDXLConditions(input_width=8,width=8,text_width=12,image_width=16,pooled_width=10,joint_image_context=True)
        before={n:p.detach().clone() for n,p in h.named_parameters()};params=configure(h)
        optimizer=torch.optim.SGD(params,lr=.1);hidden=torch.randn(2,5,8,requires_grad=True);visual=torch.randn(2,4,8,requires_grad=True)
        loss=prefix_loss(h(hidden,visual)['image'],torch.randn(2,257,16));loss.backward();optimizer.step()
        self.assertIsNone(hidden.grad);self.assertGreater(float(visual.grad.abs().sum()),0)
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in h.named_parameters() if n.startswith('image.')))
        self.assertTrue(all(torch.equal(p,before[n]) and p.grad is None for n,p in h.named_parameters() if not n.startswith('image.')))
        self.assertFalse(h.joint_image_context)

    def test_loss_normalizes_each_reference_not_batch_energy(self):
        target=torch.randn(2,4,8);pred=torch.randn(2,4,8)
        torch.testing.assert_close(prefix_loss(pred,target),sum(prefix_loss(pred[i:i+1],target[i:i+1]) for i in range(2))/2)
        torch.testing.assert_close(prefix_loss(target,target),torch.tensor(0.),atol=1e-7,rtol=0)

    def test_TRAIN_unique_reference_and_conflicting_teacher_guard(self):
        x=torch.randn(4,8);y=torch.randn(5,8)
        rows=[dict(split='train',input_key=k,reference_sha256='ref') for k in ['a','b']]+[dict(split='dev',input_key='c',reference_sha256='other')]
        cache=dict(records=rows,features={k:dict(visual=x.clone()) for k in ['a','b']},TRAIN_electronic_conditions={k:dict(image=y.clone()) for k in ['a','b']})
        self.assertEqual(len(reference_rows(cache)),1)
        cache['TRAIN_electronic_conditions']['c']=dict(image=y)
        with self.assertRaises(ValueError):reference_rows(cache)
        del cache['TRAIN_electronic_conditions']['c'];cache['TRAIN_electronic_conditions']['b']['image']+=1
        with self.assertRaises(ValueError):reference_rows(cache)


if __name__=='__main__':unittest.main()
