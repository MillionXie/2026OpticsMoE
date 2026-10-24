import unittest
import torch
from .refine_measured_readout import ridge_weights


class MeasuredReadoutTest(unittest.TestCase):
    def test_conditioning_folds_into_original_biasless_linear(self):
        torch.manual_seed(7)
        x = 1 + .02 * torch.randn(40, 8, dtype=torch.float64)
        x = x / x.mean(1, keepdim=True)
        y = torch.arange(40) % 2
        folded, stats = ridge_weights(x, y, 100., classes=2)
        direct = ((x - stats['TRAIN_mean']) / stats['TRAIN_scale']) @ stats['standardized_weights'] + stats['TRAIN_prior']
        torch.testing.assert_close(x.float() @ folded.T, direct.float(), atol=2e-5, rtol=2e-5)
        self.assertEqual(folded.shape, (2, 8))


if __name__ == '__main__':
    unittest.main()
