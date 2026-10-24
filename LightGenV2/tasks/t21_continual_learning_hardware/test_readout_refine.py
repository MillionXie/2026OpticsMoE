import unittest
import torch

from .refine_native_readout import fit_head, feature_scores, upstream_state


class ReadoutRefinementTest(unittest.TestCase):
    def test_TRAIN_updates_original_linear_without_VAL_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        x = torch.randn(80, 4)
        y = (x[:, 0] > 0).long()
        head = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            head.weight.zero_()
        val = torch.randn(40, 4, requires_grad=True)
        target = (val[:, 0] > 0).long()
        rows = []
        cfg = dict(learning_rate=.03, minimum_learning_rate=.01,
                   weight_decay=.001, epochs=12, batch_size=16,
                   TRAIN_feature_dropout=.02, TRAIN_feature_noise_std=.01,
                   label_smoothing=.02, LBFGS_steps=3, LBFGS_anchor_strength=.001)
        fit_head(head, x, y, val, target, cfg,
                 lambda *args: rows.append(feature_scores(head, val, target)))
        self.assertIsNone(val.grad)
        self.assertGreater(rows[-1]['accuracy'], .85)
        self.assertEqual(head.weight.numel(), 8)
        self.assertEqual(len(rows), 16)

    def test_frozen_state_includes_buffers(self):
        model = torch.nn.Module()
        model.register_buffer('display_gain', torch.tensor([1.]))
        model.shared_head = torch.nn.Linear(4, 2, bias=False)
        before = upstream_state(model)
        with torch.no_grad():
            model.shared_head.weight.add_(1.)
        self.assertEqual(before, upstream_state(model))
        model.display_gain.add_(1.)
        self.assertNotEqual(before, upstream_state(model))


if __name__ == '__main__':
    unittest.main()
