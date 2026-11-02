import unittest
import torch
from ..scene_resnet_frontend import ResNetConditionReadout, condition_loss, tokenize_text, validate_training_rows


class ResNetFrontendTests(unittest.TestCase):
    def test_full_prompt_and_no_silent_truncation(self):
        tokens=tokenize_text(['完整两句文字。 Keep the table lamp on the desk.'])
        self.assertEqual(tokens[0,0].item(),1)
        self.assertEqual(tokens[0,-1].item(),2)
        with self.assertRaises(ValueError):tokenize_text(['x'*513])

    def test_condition_shapes_gradients_and_reference_only_image(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        model=ResNetConditionReadout()
        visual=torch.randn(2,64,512)
        tokens=tokenize_text(['Put the lamp on a table.','Put the lamp on a bed.'])
        out=model(visual,tokens)
        self.assertEqual(out['text'].shape,(2,77,2048))
        self.assertEqual(out['image'].shape,(2,257,1280))
        self.assertEqual(out['pooled'].shape,(2,1280))
        changed=model(visual,tokens.flip(0))
        self.assertTrue(torch.equal(out['image'],changed['image']))
        self.assertFalse(torch.equal(out['text'],changed['text']))
        target={k:torch.randn_like(v,requires_grad=True)for k,v in out.items()}
        condition_loss(out,target).backward()
        self.assertTrue(all(t.grad is None for t in target.values()))
        for prefix in ['text_encoder','project','text_projection','text.','image.','pooled.']:
            grads=[p.grad for n,p in model.named_parameters()if n.startswith(prefix)]
            self.assertTrue(any(g is not None and torch.isfinite(g).all()and g.abs().sum()>0 for g in grads),prefix)

    def test_TEST_gradient_context_is_rejected(self):
        rows=[dict(split='train',input_key=str(i),reference_id=str(i%76),prompt='Two sentences. Complete instruction.')for i in range(624)]
        validate_training_rows(rows)
        rows[0]['split']='test'
        with self.assertRaises(ValueError):validate_training_rows(rows)

    def test_three_metric_aggregate_excludes_reward(self):
        from ..scene_resnet_frontend_metrics import aggregate,FIELDS
        rows=[dict(run_id='rn',category=c,CLIP_T=.3,DINOv2_I=.4,CLIP_I=.7)for c in ['lamp','lamp','pillow','pillow']]
        out=aggregate(rows)
        self.assertEqual(out['rn/all']['n'],4)
        self.assertEqual(set(out['rn/lamp']['scores']),set(FIELDS))
        self.assertAlmostEqual(out['rn/all']['scores']['CLIP_T']['mean'],.3)


if __name__=='__main__':unittest.main()
