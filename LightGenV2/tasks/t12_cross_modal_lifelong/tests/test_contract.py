import unittest
import torch

from LightGenV2.tasks.t12_cross_modal_lifelong.model import CrossModalOptics, normalize_power
from LightGenV2.tasks.t12_cross_modal_lifelong.data import encode_clevr
from LightGenV2.tasks.t12_cross_modal_lifelong.prepare_physical_concepts import rank
from LightGenV2.tasks.t12_cross_modal_lifelong.run import combine_current_replay


class ContractTest(unittest.TestCase):
    def test_fixed_geometry_and_heads(self):
        moe = CrossModalOptics("moe", phase_dropout=0)
        d2nn = CrossModalOptics("d2nn", phase_dropout=0)
        self.assertEqual((moe.height, moe.width), (1026, 1026))
        self.assertEqual((d2nn.active_height, d2nn.active_width), (986, 986))
        self.assertEqual(len(moe.first_phase), 16)
        self.assertIsNone(d2nn.router_phase)
        self.assertEqual(set(moe.heads), {"sen12ms", "clevr", "sonyc", "video"})
        self.assertEqual(moe.heads["sen12ms"][-1].out_features, 10)
        self.assertEqual(moe.heads["clevr"][-1].out_features, 2)
        self.assertEqual(moe.heads["video"][-1].out_features, 2)

    def test_freeze_contract(self):
        model = CrossModalOptics("moe", phase_dropout=0)
        model.configure_task(1, warmup=False)
        self.assertEqual(int(model.active_count), 8)
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[:4]))
        self.assertTrue(all(p.requires_grad for p in model.first_phase[4:8]))
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[8:]))
        self.assertTrue(all(not p.requires_grad for p in model.heads["sen12ms"].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.heads["clevr"].parameters()))

    def test_sequential_d2nn_freezes_old_heads_not_shared_phases(self):
        model = CrossModalOptics("d2nn", phase_dropout=0)
        model.configure_task(2, warmup=False)
        self.assertTrue(model.first_phase.requires_grad)
        self.assertTrue(model.global_phase.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.heads["sen12ms"].parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.heads["clevr"].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.heads["sonyc"].parameters()))

    def test_fourth_task_uses_last_four_slots(self):
        model = CrossModalOptics("moe", phase_dropout=0)
        model.configure_task(3, warmup=False)
        self.assertEqual(int(model.active_count), 16)
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[:12]))
        self.assertTrue(all(p.requires_grad for p in model.first_phase[12:]))
        self.assertTrue(all(p.requires_grad for p in model.heads["video"].parameters()))

    def test_old_task_capacity_mask_survives_expansion(self):
        model = CrossModalOptics("moe", phase_dropout=0)
        model.configure_task(3, warmup=False)
        captured = {}
        original = model.route

        def route(amplitude, **kwargs):
            captured["mask"] = kwargs["expert_mask"].detach().cpu()
            return original(amplitude, **kwargs)

        model.route = route
        model(torch.rand(1, 224, 224), "sen12ms")
        self.assertEqual(captured["mask"].tolist(), [True] * 4 + [False] * 12)

    def test_power_normalization(self):
        x = normalize_power(torch.rand(3, 224, 224))
        self.assertTrue(torch.allclose(x.square().sum((-2, -1)), torch.ones(3), atol=1e-5))

    def test_video_subset_and_split_use_independent_hashes(self):
        identities = [f"continuity:{i // 100}:{i % 100}" for i in range(2000)]
        selected = sorted(identities, key=rank)[:1024]
        buckets = [int(rank(x, "split")[:8], 16) / 0xffffffff for x in selected]
        counts = [sum(x < .70 for x in buckets),
                  sum(.70 <= x < .85 for x in buckets),
                  sum(x >= .85 for x in buckets)]
        self.assertTrue(all(count > 100 for count in counts), counts)

    def test_current_task_weight_does_not_shrink_with_task_count(self):
        current = torch.tensor(2.0)
        replay = [torch.tensor(1.0), torch.tensor(3.0), torch.tensor(5.0)]
        self.assertEqual(float(combine_current_replay(current, replay, 1.0)), 2.5)
        self.assertEqual(float(combine_current_replay(current, [], 1.0)), 2.0)

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
