import unittest,torch
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_paired_train import electronic_condition_loss


class ConditionKDTest(unittest.TestCase):
    def test_all_heads_get_gradients_and_zero_match(self):
        wanted={n:torch.ones(2,3) for n in ['image','text','pooled']}
        payload={'TRAIN_electronic_conditions':{'one':wanted}}
        pred={n:torch.zeros(1,2,3,requires_grad=True) for n in wanted}
        row={'split':'train','input_key':'one'};loss=electronic_condition_loss(pred,payload,row);loss.backward()
        self.assertTrue(all(v.grad.abs().sum()>0 for v in pred.values()))
        self.assertEqual(electronic_condition_loss({n:v.unsqueeze(0) for n,v in wanted.items()},payload,row).item(),0)
        with self.assertRaises(ValueError):electronic_condition_loss(pred,payload,dict(row,split='dev'))


if __name__=='__main__':unittest.main()
