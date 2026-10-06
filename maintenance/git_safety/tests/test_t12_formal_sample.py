import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.infer_formal_sample import predict


class Lookup:
    def batch(self, prompts, device):
        return torch.zeros(1, 1, 2), torch.ones(1, 1), None


class FormalSampleTests(unittest.TestCase):
    def test_original_seed_and_floor_quantization_no_grad(self):
        noises = []
        def model(reference, embeddings, mask, noise):
            self.assertFalse(torch.is_grad_enabled())
            noises.append(noise.clone())
            return torch.zeros_like(reference)
        reference = torch.zeros(1, 3, 2, 2)
        a = predict(model, reference, Lookup(), 'prompt', 4)
        b = predict(model, reference, Lookup(), 'prompt', 4)
        self.assertTrue((a == 127).all())
        self.assertTrue((a == b).all())
        expected = torch.randn(reference.shape, generator=torch.Generator().manual_seed(1046))
        self.assertTrue(torch.equal(noises[0], expected))
        self.assertTrue(torch.equal(noises[0], noises[1]))
