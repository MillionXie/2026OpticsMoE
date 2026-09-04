from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ..data import LGVQFrameDataset, attach_training_soft_targets, load_frame_cache
from ..hardware_contract import (
    OPTICAL_PASSES,
    forward_hardware,
    parallel_router_from_ccd,
    phase_canvases,
)
from ..modeling import ElectronicGridRoute, ElectronicSequenceRoute, QualityReadout, build_model, deterministic_bridge
from ..preflight import FORBIDDEN_CLASS_FRAGMENTS, _architecture_audit, run_preflight
from ..settings import OpticalGeometry, load_settings
from ..training import (
    _checkpoint,
    _load_state,
    _optimizer,
    apply_training_initialization,
    inspect_training_initialization,
    pairwise_ranking_loss,
    batch_correlation_loss,
    normalized_soft_target_loss,
    weighted_smooth_l1_loss,
)


CONFIG = Path(__file__).parents[1] / "configs" / "release" / "formal_alpha50.yaml"
KD_CONFIG = Path(__file__).parents[1] / "configs" / "release" / "formal_alpha50_kd.yaml"
WARM_CONFIG = Path(__file__).parents[1] / "configs" / "release" / "formal_alpha50_kd075_warm_v1.yaml"
STATS_WARM_CONFIG = Path(__file__).parents[1] / "configs" / "release" / "formal_alpha50_kd075_center100_stats_warm_v1.yaml"
SPATIAL2_CONFIG = Path(__file__).parents[1] / "configs" / "release" / "formal_alpha50_kd075_center100_stats_warm_v1_spatial2.yaml"


def small_settings():
    value = load_settings(CONFIG, synthetic=True)
    return replace(
        value,
        geometry=OpticalGeometry(96, 88, 40, 16, 24),
        frame_size=64,
        token_grid=4,
        width=192,
        bridge_pool=1,
        parallel_detector_intervals=((6, 12), (28, 34)),
        serial_detector_intervals=((20, 30), (58, 68)),
        input_shift_pixels=0,
        phase_shift_pixels=0,
        ccd_shift_pixels=0,
        detector_projection_size=16,
        head_width=32,
        num_workers=0,
        k_space_enabled=False,
        synthetic=True,
    )


class CoreTests(unittest.TestCase):
    def test_hardware_phase_and_amplitude_contract(self) -> None:
        settings = small_settings()
        model = build_model(settings).eval()
        phases = phase_canvases(model)
        self.assertEqual(tuple(phases), OPTICAL_PASSES)
        self.assertTrue(
            all(
                tuple(value.shape)
                == (settings.geometry.active_size, settings.geometry.active_size)
                for value in phases.values()
            )
        )
        frames = torch.randint(0, 256, (1, 4, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            result = forward_hardware(
                model, frames, ["sample"], stop_before="stage1_expert"
            )
        self.assertEqual(
            tuple(result.amplitudes["stage1_router"].shape),
            (1, settings.geometry.active_size, settings.geometry.active_size),
        )
        self.assertEqual(
            tuple(result.amplitudes["stage1_expert"].shape),
            (1, settings.geometry.active_size, settings.geometry.active_size),
        )
        self.assertEqual(
            tuple(result.routing["stage1"]["selected_mask"].shape), (1, 4, 4)
        )

    def test_hardware_simulated_forward_matches_formal_forward(self) -> None:
        model = build_model(small_settings()).eval()
        frames = torch.randint(0, 256, (1, 4, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            expected = model(frames, optical_enabled=True)["prediction"]
            actual = forward_hardware(model, frames, ["sample"]).prediction
        self.assertIsNotNone(actual)
        self.assertTrue(torch.allclose(expected, actual, atol=1.0e-6, rtol=1.0e-5))

    def test_parallel_measured_router_returns_two_experts_per_frame(self) -> None:
        model = build_model(small_settings()).eval()
        active = torch.rand(2, 88, 88)
        result = parallel_router_from_ccd(active, model)
        self.assertEqual(tuple(result["probabilities"].shape), (2, 4, 4))
        self.assertTrue(bool((result["selected_mask"].sum(-1) == 2).all()))

    def test_geometry_and_alpha_are_locked(self) -> None:
        settings = load_settings(CONFIG, synthetic=True)
        self.assertEqual((settings.geometry.canvas_size, settings.geometry.active_size, settings.geometry.quadrant_size, settings.geometry.expert_size, settings.geometry.expert_pitch), (518, 478, 232, 109, 123))
        self.assertEqual(settings.top_k, 2)
        self.assertGreaterEqual(settings.alpha_min, 0.50)
        self.assertEqual((settings.wavelength_nm, settings.pixel_pitch_um, settings.distance_m), (532.0, 17.0, 0.10))
        self.assertTrue(settings.k_space_enabled)
        self.assertEqual(settings.theta_max_deg, 1.0)
        self.assertEqual(settings.parallel_detector_intervals, ((79, 108), (124, 153)))
        self.assertEqual(settings.serial_detector_intervals, ((164, 223), (255, 314)))
        self.assertEqual(settings.detector_projection_size, 196)

    def test_model_has_no_forbidden_module_type(self) -> None:
        model = build_model(small_settings())
        names = [module.__class__.__name__.lower() for module in model.modules()]
        self.assertFalse([name for name in names if any(fragment in name for fragment in FORBIDDEN_CLASS_FRAGMENTS)])

    def test_forward_on_and_same_model_off(self) -> None:
        model = build_model(small_settings()).eval()
        frames = torch.randint(0, 256, (1, 4, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            on = model(frames, optical_enabled=True)
            off = model(frames, optical_enabled=False)
        self.assertEqual(tuple(on["prediction"].shape), (1, 2))
        self.assertEqual(tuple(off["prediction"].shape), (1, 2))
        self.assertEqual(set(on["routing"]), {"stage1", "stage3"})
        self.assertEqual(off["routing"], {})
        self.assertTrue(all(bool((item["selected_mask"].sum(-1) == 2).all()) for item in on["routing"].values()))

    def test_bridge_shape(self) -> None:
        value = torch.randn(2, 4, 196, 192)
        self.assertEqual(tuple(deterministic_bridge(value, 4).shape), (2, 68, 192))

    def test_bridge_preserves_v1_order(self) -> None:
        value = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
        bridged = deterministic_bridge(value, 2).flatten()
        expected = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
             8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
             1.5, 5.5, 9.5, 13.5]
        )
        self.assertTrue(torch.equal(bridged, expected))

    def test_readout_preserves_v1_direct_token_contract(self) -> None:
        settings = replace(small_settings(), bridge_pool=4)
        readout = QualityReadout(settings).eval()
        value = torch.randn(2, settings.serial_token_count, settings.width)
        changed_means = value.clone()
        changed_means[:, -settings.frame_count :] += 100.0
        with torch.no_grad():
            self.assertTrue(torch.equal(readout(value), readout(changed_means)))

    def test_v4stats_spatial_readout_uses_within_frame_statistics(self) -> None:
        torch.manual_seed(7)
        settings = replace(small_settings(), bridge_pool=4, spatial_statistics_pooling=True)
        readout = QualityReadout(settings).eval()
        baseline = torch.zeros(2, settings.serial_token_count, settings.width)
        changed = baseline.clone()
        count = settings.frame_count * settings.bridge_pool * settings.bridge_pool
        changed[:, :count:2] = -1.0
        changed[:, 1:count:2] = 1.0
        with torch.no_grad():
            baseline_prediction = readout(baseline)
            changed_prediction = readout(changed)
        self.assertFalse(torch.equal(baseline_prediction[:, :1], changed_prediction[:, :1]))
        self.assertTrue(torch.equal(baseline_prediction[:, 1:], changed_prediction[:, 1:]))

    def test_electronic_routes_preserve_v1_direct_output(self) -> None:
        grid = ElectronicGridRoute(192, 4)
        sequence = ElectronicSequenceRoute(192)
        grid_value = torch.randn(2, 4, 16, 192)
        sequence_value = torch.randn(2, 8, 192)
        with torch.no_grad():
            grid.pointwise.weight.zero_()
            grid.pointwise.bias.zero_()
            sequence.pointwise.weight.zero_()
            sequence.pointwise.bias.zero_()
            self.assertEqual(float(grid(grid_value).abs().max()), 0.0)
            self.assertEqual(float(sequence(sequence_value).abs().max()), 0.0)

    def test_quality_front_channels_and_output(self) -> None:
        settings = small_settings()
        front = build_model(settings).frame_stem.eval()
        frames = torch.randint(0, 256, (2, 4, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            channels = front.quality_channels(frames)
            output = front(frames)
        self.assertEqual(tuple(channels.shape), (2, 4, 14, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 4, 16, 192))
        self.assertEqual(float(channels[:, 0, 10].abs().max()), 0.0)
        self.assertTrue(bool((channels[:, :, [6, 7, 8, 9, 10]] >= 0.0).all()))
        self.assertEqual(sum(isinstance(layer, torch.nn.Conv2d) for layer in front.modules()), 5)
        self.assertFalse([name for name, parameter in front.named_parameters() if not parameter.requires_grad])

    def test_v3_architecture_contract_is_reported(self) -> None:
        settings = small_settings()
        audit = _architecture_audit(settings)
        self.assertEqual(settings.architecture_label, "lgvq_quality14_conv5_fourstage_oeo518_o2e109_alpha50_v3")
        self.assertEqual(
            replace(settings, alpha_min=0.65, alpha_initial=0.70).architecture_label,
            "lgvq_quality14_conv5_fourstage_oeo518_o2e109_alpha65_v3",
        )
        self.assertEqual(audit["front_quality_channel_count"], 14)
        self.assertEqual(audit["front_trainable_conv2d_count"], 5)
        self.assertFalse(audit["front_pretrained_network"])
        self.assertEqual(audit["front_frozen_parameter_count"], 0)
        self.assertEqual(audit["bridge_and_readout_contract"], "v1_pooled_tokens_then_frame_means")
        self.assertFalse(audit["spatial_statistics_pooling"])
        self.assertEqual(audit["temporal_readout_contract"], "v1_frame_mean_depthwise_k3")
        self.assertEqual(audit["electronic_route_contract"], "v1_direct_convolution_output")

    def test_v4stats_architecture_contract_is_distinct(self) -> None:
        settings = replace(small_settings(), spatial_statistics_pooling=True)
        audit = _architecture_audit(settings)
        self.assertEqual(
            settings.architecture_label,
            "lgvq_quality14_conv5_fourstage_oeo518_o2e109_alpha50_v4stats",
        )
        self.assertTrue(audit["spatial_statistics_pooling"])
        self.assertEqual(audit["bridge_and_readout_contract"], "v4stats_mean_std_max")

    def test_v3_state_contract_does_not_gain_v4stats_parameters(self) -> None:
        settings = small_settings()
        state_keys = set(build_model(settings).state_dict())
        self.assertFalse([key for key in state_keys if "spatial_statistics" in key])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v3.pt"
            model = build_model(settings)
            _checkpoint(checkpoint, model, _optimizer(model, settings), settings, epoch=1, metrics=None)
            _load_state(build_model(settings), checkpoint, settings)
            with self.assertRaisesRegex(RuntimeError, "architecture"):
                _load_state(
                    build_model(replace(settings, spatial_statistics_pooling=True)),
                    checkpoint,
                    replace(settings, spatial_statistics_pooling=True),
                )

    def test_older_checkpoint_is_rejected_before_state_loading(self) -> None:
        settings = small_settings()
        model = build_model(settings)
        optimizer = _optimizer(model, settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _checkpoint(path, model, optimizer, settings, epoch=1, metrics=None)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["architecture"], settings.architecture_label)
            for old_label in (
                "lgvq_rawframes_fourstage_oeo518_o2e109_alpha50_v1",
                "lgvq_rawframes_fourstage_oeo518_o2e109_alpha50_v2",
            ):
                saved["architecture"] = old_label
                torch.save(saved, path)
                with self.assertRaisesRegex(RuntimeError, "architecture"):
                    _load_state(build_model(settings), path, settings)

    def test_cache_contract(self) -> None:
        payload = {
            "schema_version": 1,
            "frames": torch.zeros(3, 4, 3, 16, 16, dtype=torch.uint8),
            "sample_ids": ["a", "b", "c"],
            "video_paths": ["a.mp4", "b.mp4", "c.mp4"],
            "splits": ["train", "train", "test"],
            "targets": torch.zeros(3, 2),
            "target_names": ["spatial", "temporal"],
            "alignment_target_present": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(payload, path)
            loaded = load_frame_cache(path)
        self.assertEqual(loaded["frames"].dtype, torch.uint8)

    def test_soft_targets_are_aligned_by_id_and_train_only(self) -> None:
        payload = {
            "frames": torch.zeros(3, 4, 3, 16, 16, dtype=torch.uint8),
            "sample_ids": ["train-a", "test-a", "train-b"],
            "video_paths": ["a.mp4", "t.mp4", "b.mp4"],
            "splits": ["train", "test", "train"],
            "targets": torch.zeros(3, 2),
        }
        teacher = {
            "sample_ids": ["train-b", "train-a"],
            "predictions": torch.tensor([[30.0, 40.0], [10.0, 20.0]]),
            "target_names": ["spatial", "temporal"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(teacher, path)
            aligned = attach_training_soft_targets(payload, path)
        train = LGVQFrameDataset(aligned, "train")
        test = LGVQFrameDataset(aligned, "test")
        self.assertTrue(torch.equal(train[0]["soft_target"], torch.tensor([10.0, 20.0])))
        self.assertTrue(torch.equal(train[1]["soft_target"], torch.tensor([30.0, 40.0])))
        self.assertNotIn("soft_target", test[0])
        self.assertEqual(aligned["training_soft_target_provenance"]["sample_count"], 2)

    def test_soft_target_alignment_rejects_missing_train_id(self) -> None:
        payload = {
            "frames": torch.zeros(2, 4, 3, 16, 16, dtype=torch.uint8),
            "sample_ids": ["a", "b"],
            "video_paths": ["a.mp4", "b.mp4"],
            "splits": ["train", "train"],
            "targets": torch.zeros(2, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save({"sample_ids": ["a"], "predictions": torch.zeros(1, 2), "target_names": ["spatial", "temporal"]}, path)
            with self.assertRaisesRegex(ValueError, "exactly match"):
                attach_training_soft_targets(payload, path)

    def test_soft_target_loss_uses_training_target_normalization(self) -> None:
        mean = torch.tensor([50.0, 40.0])
        std = torch.tensor([10.0, 20.0])
        teacher = torch.tensor([[60.0, 20.0], [40.0, 60.0]])
        prediction = (teacher - mean) / std
        self.assertEqual(float(normalized_soft_target_loss(prediction, teacher, mean, std)), 0.0)

    def test_target_loss_weights_prioritize_spatial_without_changing_mean_scale(self) -> None:
        prediction = torch.tensor([[2.0, 0.5], [0.0, 0.0]])
        target = torch.zeros_like(prediction)
        unweighted = weighted_smooth_l1_loss(prediction, target)
        spatial2 = weighted_smooth_l1_loss(prediction, target, (2.0, 1.0))
        temporal2 = weighted_smooth_l1_loss(prediction, target, (1.0, 2.0))
        self.assertGreater(float(spatial2), float(unweighted))
        self.assertLess(float(temporal2), float(unweighted))
        self.assertAlmostEqual(
            float(weighted_smooth_l1_loss(prediction, target, (2.0, 2.0))),
            float(unweighted),
        )

    def test_ranking_and_correlation_accept_target_loss_weights(self) -> None:
        target = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        prediction = torch.stack((-target[:, 0], target[:, 1]), dim=1)
        self.assertGreater(
            float(pairwise_ranking_loss(prediction, target, target_weights=(2.0, 1.0))),
            float(pairwise_ranking_loss(prediction, target, target_weights=(1.0, 2.0))),
        )
        self.assertGreater(
            float(batch_correlation_loss(prediction, target, target_weights=(2.0, 1.0))),
            float(batch_correlation_loss(prediction, target, target_weights=(1.0, 2.0))),
        )

    def test_spatial2_config_keeps_v4stats_and_weights_only_training_targets(self) -> None:
        settings = load_settings(SPATIAL2_CONFIG, synthetic=True)
        self.assertTrue(settings.spatial_statistics_pooling)
        self.assertEqual((settings.spatial_target_weight, settings.temporal_target_weight), (2.0, 1.0))
        self.assertEqual(settings.output_dir.name, "lgvq_oeo109_alpha50_kd075_center100_v4stats_warmv1_spatial2")
        self.assertEqual(settings.architecture_label, "lgvq_quality14_conv5_fourstage_oeo518_o2e109_alpha50_v4stats")

    def test_target_loss_weights_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "target weights must be positive"):
            replace(small_settings(), spatial_target_weight=0.0).validate()

    def test_kd_config_is_optional_and_training_only(self) -> None:
        plain = load_settings(CONFIG, synthetic=True)
        kd = load_settings(KD_CONFIG, synthetic=True)
        self.assertEqual(plain.soft_target_weight, 0.0)
        self.assertIsNone(plain.training_soft_targets_path)
        self.assertEqual(kd.soft_target_weight, 0.35)
        self.assertTrue(str(kd.training_soft_targets_path).endswith("training_only_teacher_predictions.pt"))

    def test_warm_v1_config_has_unique_v3_output(self) -> None:
        settings = load_settings(WARM_CONFIG, synthetic=True)
        self.assertEqual(settings.initialization_checkpoint.name, "best_observed_test_checkpoint.pt")
        self.assertEqual(settings.initialization_checkpoint.parent.name, "lgvq_oeo109_alpha50_kd075")
        self.assertEqual(settings.output_dir.name, "lgvq_oeo109_alpha50_kd075_v3_warmv1")
        self.assertEqual(settings.architecture_label, "lgvq_quality14_conv5_fourstage_oeo518_o2e109_alpha50_v3")

    def test_stats_warm_v1_config_has_unique_v4_output(self) -> None:
        settings = load_settings(STATS_WARM_CONFIG, synthetic=True)
        self.assertTrue(settings.spatial_statistics_pooling)
        self.assertEqual(settings.crop_fraction, 1.0)
        self.assertEqual(settings.soft_target_weight, 0.75)
        self.assertEqual(settings.initialization_checkpoint.parent.name, "lgvq_oeo109_alpha50_kd075")
        self.assertEqual(settings.output_dir.name, "lgvq_oeo109_alpha50_kd075_center100_v4stats_warmv1")
        self.assertEqual(settings.architecture_label, "lgvq_quality14_conv5_fourstage_oeo518_o2e109_alpha50_v4stats")

    def test_partial_v1_warm_start_loads_only_same_name_and_shape(self) -> None:
        settings = small_settings()
        model = build_model(settings)
        matching_name = next(name for name, _ in model.named_parameters() if not name.startswith("frame_stem."))
        matching_source = torch.full_like(model.state_dict()[matching_name], 0.125)
        front_before = model.frame_stem.conv1.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.pt"
            torch.save({
                "schema_version": 1,
                "architecture": "lgvq_rawframes_fourstage_oeo518_o2e109_alpha50_v1",
                "epoch": 37,
                "state_dict": {
                    matching_name: matching_source,
                    "frame_stem.conv1.weight": torch.zeros(1),
                },
                "optimizer": {"must_not_be_restored": True},
            }, path)
            warm = replace(settings, initialization_checkpoint=path)
            inspected = inspect_training_initialization(model, warm)
            self.assertFalse(inspected["applied_to_model"])
            preflight = run_preflight(warm, require_cache=False)
            self.assertEqual(preflight["training_initialization"]["source_sha256"], inspected["source_sha256"])
            report = apply_training_initialization(model, warm)
        self.assertTrue(torch.equal(model.state_dict()[matching_name], matching_source))
        self.assertTrue(torch.equal(model.frame_stem.conv1.weight, front_before))
        self.assertEqual(report["source_epoch"], 37)
        self.assertEqual(len(report["source_sha256"]), 64)
        self.assertEqual(report["matched_parameter_count"], matching_source.numel())
        self.assertEqual(report["matched_tensor_count"], 1)
        self.assertIn("frame_stem.conv1.weight", {
            item["key"] for item in report["skipped_keys"]["source"]
        })
        self.assertIn("frame_stem.conv4.weight", report["skipped_keys"]["target_fresh"])
        self.assertIn("frame_stem.conv5.weight", report["skipped_keys"]["target_fresh"])
        self.assertFalse(report["optimizer_restored"])
        self.assertEqual(_optimizer(model, warm).state, {})

    def test_formal_warm_start_rejects_non_v1_and_missing_checkpoint(self) -> None:
        settings = small_settings()
        model = build_model(settings)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            missing = replace(settings, initialization_checkpoint=directory / "missing.pt")
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                inspect_training_initialization(model, missing)
            wrong = directory / "v2.pt"
            torch.save({
                "architecture": "lgvq_rawframes_fourstage_oeo518_o2e109_alpha50_v2",
                "epoch": 1,
                "state_dict": {"frame_stem.conv1.weight": torch.zeros(1)},
            }, wrong)
            formal = replace(settings, initialization_checkpoint=wrong, synthetic=False)
            with self.assertRaisesRegex(RuntimeError, "ends with _v1"):
                apply_training_initialization(model, formal)

    def test_checkpoint_records_only_soft_target_provenance(self) -> None:
        settings = replace(small_settings(), soft_target_weight=0.35)
        model = build_model(settings)
        optimizer = _optimizer(model, settings)
        provenance = {
            "usage": "training_only_scalar_soft_targets",
            "path": "teacher.pt",
            "sha256": "abc123",
            "sample_count": 2,
            "target_names": ["spatial", "temporal"],
            "full_teacher_loaded_during_inference": False,
        }
        initialization = {
            "source_path": "v1.pt",
            "source_sha256": "def456",
            "source_architecture": "lgvq_rawframes_fourstage_oeo518_o2e109_alpha50_v1",
            "source_epoch": 25,
            "matched_parameter_count": 123,
            "skipped_keys": {"source": [], "target_fresh": ["frame_stem.conv1.weight"]},
            "optimizer_restored": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _checkpoint(
                path,
                model,
                optimizer,
                settings,
                epoch=1,
                metrics=None,
                soft_target_provenance=provenance,
                initialization_provenance=initialization,
            )
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(saved["training_soft_targets"]["sha256"], "abc123")
        self.assertFalse(saved["deployed_inference_uses_teacher"])
        self.assertNotIn("teacher_state_dict", saved)
        self.assertEqual(saved["training_initialization"]["source_sha256"], "def456")
        self.assertFalse(saved["training_initialization"]["optimizer_restored"])

    def test_phase_optimizer_groups_have_no_weight_decay(self) -> None:
        settings = small_settings()
        optimizer = _optimizer(build_model(settings), settings)
        groups = {str(group["name"]): group for group in optimizer.param_groups}
        self.assertEqual(float(groups["electronic"]["weight_decay"]), settings.weight_decay)
        self.assertEqual(float(groups["feature_phase"]["weight_decay"]), 0.0)
        self.assertEqual(float(groups["router_phase"]["weight_decay"]), 0.0)


if __name__ == "__main__":
    unittest.main()
