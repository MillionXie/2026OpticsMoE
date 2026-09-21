import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch

from LightGenV2.tasks.t12_cross_modal_lifelong.model import CrossModalOptics, normalize_power
from LightGenV2.tasks.t12_cross_modal_lifelong.data import encode_clevr, load_common
from LightGenV2.tasks.t12_cross_modal_lifelong.prepare_physical_concepts import rank
from LightGenV2.tasks.t12_cross_modal_lifelong.prepare_kather2016 import encode_rgb
from LightGenV2.tasks.t12_cross_modal_lifelong.run import (combine_current_replay,
    save_continual_matrix, validate_full_protocol, augment_kather_fields)


class ContractTest(unittest.TestCase):
    def test_fixed_geometry_and_heads(self):
        moe = CrossModalOptics("moe", phase_dropout=0)
        d2nn = CrossModalOptics("d2nn", phase_dropout=0)
        self.assertEqual((moe.height, moe.width), (1026, 1026))
        self.assertEqual((d2nn.active_height, d2nn.active_width), (986, 986))
        self.assertEqual(len(moe.first_phase), 16)
        self.assertIsNone(d2nn.router_phase)
        self.assertEqual(set(moe.heads), {"kather2016", "clevr", "sonyc", "video"})
        self.assertEqual(moe.heads["kather2016"][-1].out_features, 8)
        self.assertEqual(moe.heads["clevr"][-1].out_features, 2)
        self.assertEqual(moe.heads["video"][-1].out_features, 2)

    def test_freeze_contract(self):
        model = CrossModalOptics("moe", phase_dropout=0)
        model.configure_task(1, warmup=False)
        self.assertEqual(int(model.active_count), 8)
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[:4]))
        self.assertTrue(all(p.requires_grad for p in model.first_phase[4:8]))
        self.assertTrue(all(not p.requires_grad for p in model.first_phase[8:]))
        self.assertTrue(all(not p.requires_grad for p in model.heads["kather2016"].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.heads["clevr"].parameters()))

    def test_sequential_d2nn_freezes_old_heads_not_shared_phases(self):
        model = CrossModalOptics("d2nn", phase_dropout=0)
        model.configure_task(2, warmup=False)
        self.assertTrue(model.first_phase.requires_grad)
        self.assertTrue(model.global_phase.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.heads["kather2016"].parameters()))
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
        model(torch.rand(1, 224, 224), "kather2016")
        self.assertEqual(captured["mask"].tolist(), [True] * 4 + [False] * 12)

    def test_power_normalization(self):
        x = normalize_power(torch.rand(3, 224, 224))
        self.assertTrue(torch.allclose(x.square().sum((-2, -1)), torch.ones(3), atol=1e-5))

    def test_full_video_split_uses_quadruplet_identity(self):
        identities = [f"continuity:{i // 250}:{i % 250}" for i in range(5000)]
        buckets = [int(rank(x, "split")[:8], 16) / 0xffffffff for x in identities]
        counts = [sum(x < .70 for x in buckets),
                  sum(.70 <= x < .85 for x in buckets),
                  sum(x >= .85 for x in buckets)]
        self.assertEqual(sum(counts), 5000)
        self.assertTrue(all(count > 650 for count in counts[1:]), counts)

    def test_current_task_weight_does_not_shrink_with_task_count(self):
        current = torch.tensor(2.0)
        replay = [torch.tensor(1.0), torch.tensor(3.0), torch.tensor(5.0)]
        self.assertEqual(float(combine_current_replay(current, replay, 1.0)), 2.5)
        self.assertEqual(float(combine_current_replay(current, [], 1.0)), 2.0)

    def test_formal_training_rejects_subsets(self):
        with self.assertRaisesRegex(ValueError, "all_original_samples"):
            validate_full_protocol("video", {"source_quadruplets": 5000})
        with self.assertRaisesRegex(ValueError, "expected 5000"):
            validate_full_protocol("video", {"all_original_samples": True,
                                                "source_quadruplets": 1024})
        validate_full_protocol("video", {"all_original_samples": True,
                                           "source_quadruplets": 5000})

    def test_clevr_rgb_text_packing(self):
        images = torch.zeros(2, 20, 30, 3, dtype=torch.uint8)
        images[..., 0], images[..., 1], images[..., 2] = 20, 80, 160
        words = torch.zeros(2, 32, 64); words[:, 0, 2] = 1
        field = encode_clevr(images, words)
        self.assertEqual(tuple(field.shape), (2, 224, 224))
        self.assertTrue(torch.allclose(field[:, :, :112].square().sum((-2, -1)), torch.full((2,), .5), atol=1e-5))
        self.assertTrue(torch.allclose(field[:, :, 112:].square().sum((-2, -1)), torch.full((2,), .5), atol=1e-5))

    def test_kather_uses_all_rgb_channels_and_unit_power(self):
        images = np.zeros((2, 150, 150, 3), dtype=np.uint8)
        images[..., 0], images[..., 1], images[..., 2] = 20, 80, 160
        field = torch.from_numpy(encode_rgb(images))
        self.assertEqual(tuple(field.shape), (2, 224, 224))
        self.assertTrue(torch.allclose(field.float().square().sum((-2, -1)),
                                       torch.ones(2), atol=2e-3))
        self.assertGreater(float(field[:, 112:, :112].mean()),
                           float(field[:, :112, :112].mean()))

    def test_kather_augmentation_keeps_tiles_aligned_and_power(self):
        tile = torch.arange(112 * 112, dtype=torch.float32).reshape(112, 112)
        fields = torch.cat((torch.cat((tile, tile), -1),
                            torch.cat((tile, tile), -1)), -2)[None]
        torch.manual_seed(17)
        augmented = augment_kather_fields(fields)
        quadrants = [augmented[:, :112, :112], augmented[:, :112, 112:],
                     augmented[:, 112:, :112], augmented[:, 112:, 112:]]
        self.assertTrue(all(torch.equal(quadrants[0], q) for q in quadrants[1:]))
        self.assertTrue(torch.allclose(fields.square().sum(), augmented.square().sum()))

    def test_lazy_clevr_and_sonyc_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"storage": "clevr_lazy_v1"}))
            np.save(root / "train_images.npy", np.full((1, 64, 64, 3), 64, np.uint8))
            np.save(root / "train_image_index.npy", np.array([0, 0], np.int32))
            ids = np.zeros((2, 32), np.uint8); ids[:, 0] = 2
            np.save(root / "train_token_ids.npy", ids)
            np.save(root / "train_labels.npy", np.array([0, 1]))
            fields, labels, rows = load_common(root, "train")
            self.assertEqual(tuple(fields[:2].shape), (2, 224, 224))
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(rows, [{}, {}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"storage": "sonyc_lazy_v1"}))
            np.save(root / "train_audio_fields.npy", np.ones((1, 224, 112), np.float16))
            np.save(root / "train_audio_index.npy", np.array([0], np.int32))
            np.save(root / "train_event_index.npy", np.array([3], np.uint8))
            np.save(root / "train_labels.npy", np.array([1]))
            np.save(root / "event_text_fields.npy", np.ones((8, 224, 112), np.float16))
            fields, labels, rows = load_common(root, "train")
            self.assertEqual(tuple(fields[:1].shape), (1, 224, 224))
            self.assertEqual(rows[0]["event"], "4_powered-saw_presence")

    def test_continual_matrix_is_lower_triangular(self):
        metric = lambda value: {"balanced_accuracy": value, "macro_f1": value, "macro_ap": value}
        rows = [
            {"validation":{"kather2016":metric(.8)},"test":{"kather2016":metric(.7)}},
            {"validation":{"kather2016":metric(.6),"clevr":metric(.9)},
             "test":{"kather2016":metric(.5),"clevr":metric(.8)}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            payload = save_continual_matrix(Path(directory), rows)
            saved = json.loads((Path(directory) / "continual_matrix.json").read_text())
        self.assertEqual(payload, saved)
        self.assertEqual(list(saved["test"][0]), ["kather2016"])
        self.assertEqual(list(saved["test"][1]), ["kather2016", "clevr"])
        self.assertAlmostEqual(saved["continual"]["backward_transfer"]["kather2016"], -.2)
        self.assertAlmostEqual(saved["continual"]["forgetting"]["kather2016"], .2)


if __name__ == "__main__":
    unittest.main()
