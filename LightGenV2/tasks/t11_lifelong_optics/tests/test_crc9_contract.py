import unittest

import torch

from tasks.t11_lifelong_optics.crc9_model import CRC9Optics, encode_rgb


class CRC9ContractTests(unittest.TestCase):
    def test_rgb_encoding_is_224_and_unit_power(self):
        image = torch.randint(0, 256, (2, 48, 52, 3), dtype=torch.uint8)
        value = encode_rgb(image)
        self.assertEqual(tuple(value.shape), (2, 224, 224))
        torch.testing.assert_close(value.square().sum((-2, -1)), torch.ones(2))

    def test_d2nn_head_adaptation_freezes_both_phase_planes(self):
        model = CRC9Optics("d2nn", phase_dropout=0.)
        model.configure_d2nn_head_adaptation()
        self.assertFalse(model.first_phase.requires_grad)
        self.assertFalse(model.global_phase.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in model.readout.parameters()))

    def test_later_moe_stage_freezes_old_experts_and_shared_phase(self):
        model = CRC9Optics("moe", phase_dropout=0.)
        model.configure_moe_stage(2)
        self.assertEqual(int(model.active_count), 12)
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[:8]))
        self.assertTrue(all(p.requires_grad for p in model.first_phase[8:12]))
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[12:]))
        self.assertFalse(model.global_phase.requires_grad)
        self.assertTrue(model.router_phase.requires_grad)

    def test_both_models_have_same_feature_and_class_shapes(self):
        image = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
        for architecture in ("moe", "d2nn"):
            model = CRC9Optics(architecture, phase_dropout=0.)
            output = model(image)
            self.assertEqual(tuple(output["ccd_features"].shape), (1, 256))
            self.assertEqual(tuple(output["logits"].shape), (1, 9))
            torch.testing.assert_close(output["probabilities"].sum(1), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
