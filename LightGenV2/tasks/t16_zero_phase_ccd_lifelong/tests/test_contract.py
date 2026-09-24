import numpy as np
import torch

from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.data import (
    PairedEuroSatFields, balanced_negative_words, paired_rgb_sar_field,
)
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics


def test_all_phase_parameters_begin_at_raw_zero_and_no_linear_head():
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
        assert not any(isinstance(layer, torch.nn.Linear) for layer in model.modules())
        assert len(model.output_centers) == 10
        assert len(set(model.output_centers)) == 10


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
