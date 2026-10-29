import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.scene_tail_batches import pack_features,interval_loss
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_conditioning import QwenSDXLConditions


class TailMinibatches(unittest.TestCase):
    def test_variable_length_condition_batch_matches_individual_requests(self):
        torch.manual_seed(1042);head=QwenSDXLConditions(input_width=8,width=16,text_width=12,image_width=10,pooled_width=10,joint_image_context=True).eval()
        rows=[dict(hidden=torch.randn(n,8),visual=torch.randn(4,8)) for n in [3,7]]
        h,v,padding=pack_features(rows,'cpu');actual=head(h,v,padding)
        for i,row in enumerate(rows):
            single=head(row['hidden'][None],row['visual'][None])
            for name in actual:torch.testing.assert_close(actual[name][i:i+1],single[name],atol=3e-6,rtol=3e-6)
        self.assertEqual(padding[0].tolist(),[False,False,False,True,True,True,True])

    def test_each_interval_preserves_its_own_target_normalizer(self):
        wanted=torch.tensor([.5,10.]).reshape(2,1,1,1);predicted=wanted+1
        actual=interval_loss(predicted,wanted)
        expected=(interval_loss(predicted[:1],wanted[:1])+interval_loss(predicted[1:],wanted[1:]))/2
        torch.testing.assert_close(actual,expected);self.assertAlmostEqual(actual.item(),.505,places=6)


if __name__=='__main__':unittest.main()
