import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.data import (
    PairedEuroSatFields, balanced_negative_words, paired_rgb_sar_field,
)
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_eurosat import accuracy_metrics


def test_all_phase_parameters_begin_at_raw_zero_and_one_shared_linear_head():
    moe_head = DirectCCDOptics("moe").shared_head
    d2nn_head = DirectCCDOptics("d2nn").shared_head
    assert torch.equal(moe_head.weight, d2nn_head.weight)
    assert moe_head.weight is not d2nn_head.weight
    for architecture in ("moe", "d2nn"):
        model = DirectCCDOptics(architecture)
        phases = [model.global_phase, *model.additional_phases]
        if architecture == "moe":
            phases.extend([model.router_phase, *model.first_phase])
        else:
            phases.append(model.first_phase)
        assert all(torch.count_nonzero(phase) == 0 for phase in phases)
        assert all(torch.allclose(model.transmission(phase).real,
                                  torch.full_like(phase, -1)) for phase in phases)
        linear = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        assert linear == [model.shared_head]
        assert tuple(model.shared_head.weight.shape) == (10, 784)
        assert model.shared_head.bias is None
        assert model.shared_head.weight.requires_grad
        assert len(model.output_centers) == 10
        assert len(set(model.output_centers)) == 10


def test_router_and_expert_slots_share_quadrant_numbering_and_gaps():
    model = DirectCCDOptics("moe", activation_order="quadrant")
    assert len(model.slots) == len(model.router_centers) == 16
    assert model.router_pitch - model.router_side == 16
    assert len(set(model.slots)) == len(set(model.router_centers)) == 16
    for index, (row, col) in enumerate(model.quadrant_order):
        y, x = model.slots[index]
        assert y == model.border + row * (model.expert_size + model.gap)
        assert x == model.border + col * (model.expert_size + model.gap)
        ry, rx = model.router_centers[index]
        assert ry == model.height // 2 + (row - 1.5) * model.router_pitch
        assert rx == model.width // 2 + (col - 1.5) * model.router_pitch
    for stage in range(4):
        model.configure_stage(stage)
        assert int(model.active_count) == 4 * (stage + 1)
        assert [phase.requires_grad for phase in model.first_phase] == [
            4 * stage <= index < 4 * (stage + 1) for index in range(16)
        ]
    assert model.output_side == 96
    assert (model.x_pitch, model.y_pitch) == (128, 160)


def test_center_out_activation_uses_one_slot_per_quadrant_and_freezes_old():
    model = DirectCCDOptics("moe")
    assert model.activation_order == "center_out"
    assert (model.active_indices[:4] + 1).tolist() == [4, 7, 10, 13]
    for stage in range(4):
        model.configure_stage(stage)
        new = set(model.active_indices[4*stage:4*(stage+1)].tolist())
        assert sum(phase.requires_grad for phase in model.first_phase) == 4
        assert all(phase.requires_grad == (index in new)
                   for index, phase in enumerate(model.first_phase))
    model.configure_stage(0)
    with torch.no_grad():
        result = model(torch.ones(1, 224, 224))
    assert result["logits"].shape == (1, 10)
    assert result["ccd_features"].shape == (1, 784)
    assert torch.isfinite(result["logits"]).all()
    assert torch.all(result["route_power"][0, model.active_indices[4:]] == 0)


def test_class_windows_are_diagnostics_not_classification_logits():
    model = DirectCCDOptics("d2nn").eval()
    field = torch.rand(1, 224, 224)
    with torch.no_grad():
        original = model(field)
        model.set_output_geometry(80, 112, 144)
        changed = model(field)
    assert torch.allclose(original["logits"], changed["logits"])
    assert not torch.allclose(original["window_power"], changed["window_power"])


def test_linear_loss_backpropagates_into_optics_in_both_architectures():
    for architecture in ("moe", "d2nn"):
        model = DirectCCDOptics(architecture, activation_order="center_out")
        logits = model(torch.rand(1, 224, 224))["logits"]
        F.cross_entropy(logits, torch.tensor([3])).backward()
        assert torch.isfinite(model.shared_head.weight.grad).all()
        assert model.shared_head.weight.grad.abs().sum() > 0
        assert model.global_phase.grad is not None
        assert torch.isfinite(model.global_phase.grad).all()
        assert model.global_phase.grad.abs().sum() > 0
        if architecture == "moe":
            assert model.router_phase.grad is not None
            assert model.router_phase.grad.abs().sum() > 0
            assert all(model.first_phase[index].grad is not None
                       for index in model.active_indices[:4])


def test_full_split_metric_counts_every_class_without_task_head():
    labels = np.repeat(np.arange(10), 2)
    guesses = labels.copy()
    guesses[::2] = (guesses[::2] + 1) % 10
    metrics = accuracy_metrics(labels, guesses)
    assert metrics["n"] == 20
    assert metrics["accuracy"] == metrics["balanced_accuracy"] == 0.5
    assert metrics["per_class_recall"] == [0.5] * 10


def test_paired_rgb_sar_uses_each_modality_and_equal_power():
    rgb = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 255, 128, 64
    sar = np.zeros_like(rgb)
    sar[..., 0], sar[..., 1] = 80, 160
    field = paired_rgb_sar_field(rgb, sar)
    assert field.shape == (1, 224, 224)
    assert field[:, 112:, 112:].min() > 0
    rgb_power = (field[:, :112, :].square().sum((-2, -1))
                 + field[:, 112:, :112].square().sum((-2, -1)))
    assert torch.allclose(rgb_power,
                          torch.full((1,), 0.5), atol=1e-5)
    assert torch.allclose(field[:, 112:, 112:].square().sum((-2, -1)),
                          torch.full((1,), 0.5), atol=1e-5)
    assert torch.allclose(field.square().sum((-2, -1)), torch.ones(1), atol=1e-5)


def test_pair_adapter_checks_ids_labels_and_order(tmp_path):
    images = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    images[0, ..., 0], images[2, ..., 0] = 255, 128
    images[1, ..., 0], images[3, ..., 0] = 80, 160
    row = {"train_images": images, "train_labels": np.array([2, 2, 3, 3]),
           "train_domains": np.array([0, 1, 0, 1]),
           "train_ids": np.array(["a:0", "a:1", "b:0", "b:1"])}
    path = tmp_path / "pairs.npz"
    np.savez(path, **row)
    dataset = PairedEuroSatFields(path, path, "train")
    assert len(dataset) == 2
    assert dataset.labels.tolist() == [2, 3]
    assert dataset[[0, 1]].shape == (2, 224, 224)
    assert dataset[0].shape == (224, 224)
    row["train_ids"][3] = "wrong:1"
    np.savez(path, **row)
    try:
        PairedEuroSatFields(path, path, "train")
    except ValueError as error:
        assert "IDs" in str(error)
    else:
        raise AssertionError("mismatched RGB/SAR pair was accepted")


def test_audio_word_negatives_preserve_histogram_without_matching():
    positive = np.tile(np.arange(8), 3)
    negative = balanced_negative_words(positive)
    assert np.array_equal(np.bincount(positive), np.bincount(negative))
    assert not np.any(positive == negative)


def test_corner_detector_candidates_have_ten_valid_shared_windows():
    model = DirectCCDOptics("moe")
    centers = [(76, 76), (950, 950), (76, 950), (894, 188),
               (76, 188), (950, 838), (188, 838), (950, 76),
               (188, 132), (838, 894)]
    model.set_output_windows(centers, 64)
    assert len(set(model.output_centers)) == 10
