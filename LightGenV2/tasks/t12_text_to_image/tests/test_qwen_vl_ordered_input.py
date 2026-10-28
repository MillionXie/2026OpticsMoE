import unittest
import torch

from LightGenV2.tasks.t12_text_to_image.qwen_vl_ordered_input import canonical_permutation


class OrderedInputTests(unittest.TestCase):
    def test_permutation_preserves_each_visual_token_and_text_state(self):
        # Prefix, text, image-start, four image tokens, image-end, suffix.
        native=torch.tensor([1,2,7,8,3,9,9,9,9,4,5,6])
        canonical=torch.tensor([1,2,3,9,9,9,9,4,7,8,5,6])
        perm=canonical_permutation(native,canonical)
        self.assertEqual(perm.tolist(),[0,1,4,5,6,7,8,9,2,3,10,11])
        value=torch.randn(12,8,requires_grad=True)
        value[perm].square().sum().backward()
        torch.testing.assert_close(value.grad,2*value)

    def test_token_mismatch_or_length_is_rejected(self):
        for other in (torch.tensor([1,2]),torch.tensor([1,3,3])):
            with self.assertRaises(ValueError):canonical_permutation(torch.tensor([1,2,3]),other)
