import unittest
import torch

from LightGenV2.tasks.t12_cross_modal_lifelong.model import CrossModalOptics, normalize_power
from LightGenV2.tasks.t12_cross_modal_lifelong.data import encode_clevr


class ContractTest(unittest.TestCase):
    def test_fixed_geometry_and_heads(self):
        moe = CrossModalOptics("moe", phase_dropout=0)
        d2nn = CrossModalOptics("d2nn", phase_dropout=0)
        self.assertEqual((moe.height, moe.width), (772, 1026))
        self.assertEqual((d2nn.active_height, d2nn.active_width), (732, 986))
        self.assertEqual(len(moe.first_phase), 12)
        self.assertIsNone(d2nn.router_phase)
        self.assertEqual(set(moe.heads), {"sen12ms", "clevr", "sonyc"})
        self.assertEqual(moe.heads["sen12ms"][-1].out_features, 10)
        self.assertEqual(moe.heads["clevr"][-1].out_features, 2)

    def test_freeze_contract(self):
        model = CrossModalOptics("moe", phase_dropout=0)
        model.configure_task(1, warmup=False)
        self.assertEqual(int(model.active_count), 8)
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[:4]))
        self.assertTrue(all(p.requires_grad for p in model.first_phase[4:8]))
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[8:]))
        self.assertTrue(all(not p.requires_grad for p in model.heads["sen12ms"].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.heads["clevr"].parameters()))

    def test_power_normalization(self):
        x = normalize_power(torch.rand(3, 224, 224))
        self.assertTrue(torch.allclose(x.square().sum((-2, -1)), torch.ones(3), atol=1e-5))

    def test_clevr_rgb_text_packing(self):
        images = torch.zeros(2, 20, 30, 3, dtype=torch.uint8)
        images[..., 0], images[..., 1], images[..., 2] = 20, 80, 160
        words = torch.zeros(2, 32, 64); words[:, 0, 2] = 1
        field = encode_clevr(images, words)
        self.assertEqual(tuple(field.shape), (2, 224, 224))
        self.assertTrue(torch.allclose(field[:, :, :112].square().sum((-2, -1)), torch.full((2,), .5), atol=1e-5))
        self.assertTrue(torch.allclose(field[:, :, 112:].square().sum((-2, -1)), torch.full((2,), .5), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
