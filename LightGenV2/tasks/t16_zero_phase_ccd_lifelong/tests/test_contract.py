import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.data import (
    ClevrAttributeQueryPairs, ClevrCompactQueryPairs, PairedEuroSatFields, PhysicalBinaryPairs,
    balanced_negative_words,
    paired_rgb_sar_field,
)
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong import data as t16_data
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.audit_phase_dependence import (
    _IndexedFieldAdapter,
)
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_eurosat import (
    accuracy_metrics, augment_paired_dihedral, load_initial_checkpoint,
    routing_balance_penalty,
)
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks import (
    clevr_pairwise_loss,
    configure_single_task_capacity,
    metrics as other_task_metrics,
    RESUME_CONTRACT,
    validate_resume_contract,
)
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_lifelong_moe import (
    stage_epoch_batches,
)


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


def test_binary_tasks_use_two_class_macro_recall_with_ten_output_head():
    result = other_task_metrics(np.array([0, 0, 1, 1]),
                                np.array([0, 2, 1, 2]), 2)
    assert result["n"] == 4
    assert result["accuracy"] == result["balanced_accuracy"] == 0.5
    assert result["per_class_recall"] == [0.5, 0.5]


def test_clevr_pairwise_loss_rewards_same_image_query_separation():
    logits = torch.zeros(2, 10, requires_grad=True)
    labels = torch.tensor([1, 0])
    initial = clevr_pairwise_loss(logits, labels)
    assert torch.allclose(initial, torch.tensor(np.log(2), dtype=initial.dtype))
    initial.backward()
    assert logits.grad[0, 1] < 0 and logits.grad[1, 1] > 0
    separated = logits.detach().clone()
    separated[0, 1] = 2
    separated[1, 1] = -2
    assert clevr_pairwise_loss(separated, labels) < initial


def test_lifelong_replay_covers_all_old_records_and_preserves_clevr_pairs():
    class Records:
        def __init__(self, count):
            self.count = count

        def __len__(self):
            return self.count

    datasets = {"eurosat": Records(8), "clevr": Records(14)}
    batches = list(stage_epoch_batches(datasets, tuple(datasets), 4, seed=17))
    assert len(batches) == 4
    for name, count in (("eurosat", 8), ("clevr", 14)):
        seen = np.concatenate([row[name] for row in batches])
        assert set(seen.tolist()) == set(range(count))
    for row in batches:
        clevr = row["clevr"]
        assert len(clevr) % 2 == 0
        assert np.array_equal(clevr[1::2], clevr[::2] + 1)
        assert np.all(clevr[::2] % 2 == 0)


def test_compact_clevr_text_spreads_actual_words_without_changing_image(tmp_path):
    source = tmp_path
    image = np.full((1, 16, 16, 3), 128, dtype=np.uint8)
    np.save(source / "train_images.npy", image)
    np.save(source / "train_image_index.npy", np.zeros(6, dtype=np.int64))
    np.save(source / "train_labels.npy", np.array([1, 0, 1, 0, 1, 0]))
    tokens = np.zeros((6, 32), dtype=np.uint8)
    tokens[0, :6] = [2, 3, 4, 11, 6, 7]
    tokens[1, :6] = [2, 3, 4, 10, 6, 7]
    np.save(source / "train_token_ids.npy", tokens)
    data = ClevrCompactQueryPairs(source, "train")
    fields = data.get_batch(np.array([0, 1]), "cpu")
    assert fields.shape == (2, 224, 224)
    assert torch.equal(fields[0, :112], fields[1, :112])
    assert not torch.equal(fields[0, 112:, 112:], fields[1, 112:, 112:])
    assert fields[0, 112 + 50:112 + 75, 112:].square().sum() > 0
    assert torch.allclose(fields.square().sum((-2, -1)), torch.ones(2), atol=1e-5)
    data.token_ids[0, 9] = 5
    try:
        data.get_batch(np.array([0]), "cpu")
    except ValueError as error:
        assert "nine-row" in str(error)
    else:
        raise AssertionError("non-padding query token was silently discarded")


def test_attribute_query_encoding_preserves_pairs_and_power(tmp_path):
    image = np.full((1, 16, 16, 3), 128, dtype=np.uint8)
    np.save(tmp_path / "train_images.npy", image)
    np.save(tmp_path / "train_image_index.npy", np.zeros(6, dtype=np.int64))
    np.save(tmp_path / "train_labels.npy", np.array([1, 0, 1, 0, 1, 0]))
    tokens = np.zeros((6, 32), dtype=np.uint8)
    tokens[0, :7] = [2, 3, 4, 11, 6, 7, 1]  # blue cube
    tokens[1, :7] = [2, 3, 4, 10, 6, 7, 1]  # red cube
    tokens[2:6] = tokens[:4]
    np.save(tmp_path / "train_token_ids.npy", tokens)
    pairs = ClevrAttributeQueryPairs(tmp_path, "train")
    fields = pairs.get_batch(np.array([0, 1]), "cpu")
    assert pairs.labels[:2].tolist() == [1, 0]
    assert torch.equal(fields[0, :112], fields[1, :112])
    assert torch.equal(fields[0, 112:, :112], fields[1, 112:, :112])
    assert not torch.equal(fields[0, 112:, 112:], fields[1, 112:, 112:])
    assert torch.all(fields >= 0)
    assert torch.allclose(fields.square().sum((-2, -1)), torch.ones(2), atol=1e-5)
    assert torch.allclose(fields[:, 112:, 112:].square().sum((-2, -1)),
                          torch.full((2,), 0.5), atol=1e-5)
    assert torch.allclose(fields[:, 116:164, 112:].square().sum((-2, -1)),
                          torch.full((2,), 0.25), atol=1e-5)
    assert torch.allclose(fields[:, 172:220, 112:].square().sum((-2, -1)),
                          torch.full((2,), 0.25), atol=1e-5)
    assert fields[0, 116:164, 112 + 2 * 14 + 2:112 + 2 * 14 + 12].count_nonzero() == 480
    assert fields[1, 116:164, 112 + 1 * 14 + 2:112 + 1 * 14 + 12].count_nonzero() == 480
    pairs.token_ids[0, 4] = 0
    try:
        pairs.get_batch(np.array([0]), "cpu")
    except ValueError as error:
        assert "exactly one color and shape" in str(error)
    else:
        raise AssertionError("missing shape was accepted")


def test_resume_keeps_data_model_optimizer_and_test_policy_fixed():
    prior = {key: f"same-{key}" for key in RESUME_CONTRACT}
    continued = {**prior, "epochs": 12, "model_git_commit": "new"}
    validate_resume_contract(prior, continued)
    for key in ("source_protocol_sha256", "source_manifest_sha256", "lr",
                "test_policy", "clevr_pairwise_weight"):
        changed = {**continued, key: "different"}
        try:
            validate_resume_contract(prior, changed)
        except ValueError as error:
            assert key in str(error)
        else:
            raise AssertionError(f"resume accepted a changed {key}")
    validate_resume_contract(prior, {**continued, "moe_active_experts": 4})
    try:
        validate_resume_contract(prior, {**continued, "moe_active_experts": 8})
    except ValueError as error:
        assert "moe_active_experts" in str(error)
    else:
        raise AssertionError("resume accepted a changed expert capacity")
    try:
        validate_resume_contract(prior, {**continued, "route_balance_weight": 1.0})
    except ValueError as error:
        assert "route_balance_weight" in str(error)
    else:
        raise AssertionError("resume accepted a changed routing objective")


def test_independent_eight_expert_check_does_not_freeze_untrained_slots():
    moe = DirectCCDOptics("moe")
    configure_single_task_capacity(moe, 8)
    assert int(moe.active_count) == 8
    active = set(moe.active_indices[:8].tolist())
    assert sum(phase.requires_grad for phase in moe.first_phase) == 8
    assert all(phase.requires_grad == (index in active)
               for index, phase in enumerate(moe.first_phase))
    d2nn = DirectCCDOptics("d2nn")
    try:
        configure_single_task_capacity(d2nn, 8)
    except ValueError as error:
        assert "D2NN" in str(error)
    else:
        raise AssertionError("D2NN accepted an MoE expert setting")


def test_batch_router_balance_penalty_is_slot_based_and_differentiable():
    indices = torch.tensor([3, 6, 9, 12])
    uniform = torch.zeros(2, 16)
    uniform[:, indices] = 0.25
    assert routing_balance_penalty(uniform, indices) == 0
    peaked = torch.zeros(2, 16, requires_grad=True)
    with torch.no_grad():
        peaked[:, 12] = 1
    loss = routing_balance_penalty(peaked, indices)
    assert loss > 0
    loss.backward()
    assert peaked.grad is not None and peaked.grad[:, indices].abs().sum() > 0


def test_dihedral_augmentation_preserves_tile_identity_and_power():
    ramp = torch.arange(112 * 112, dtype=torch.float32).reshape(112, 112) / 10000
    fields = torch.cat((torch.cat((ramp, ramp + 10), -1),
                        torch.cat((ramp + 20, ramp + 30), -1)), -2)[None].repeat(8, 1, 1)
    transformed = augment_paired_dihedral(fields, np.arange(8))
    assert torch.equal(transformed[0], fields[0])
    for row in transformed:
        assert torch.allclose(row[:112, 112:] - row[:112, :112],
                              torch.full((112, 112), 10.0))
        assert torch.allclose(row[112:, :112] - row[:112, :112],
                              torch.full((112, 112), 20.0))
        assert torch.allclose(row.square().sum(), fields[0].square().sum(), atol=0.1)


def test_finetune_checkpoint_rejects_changed_source(tmp_path):
    original = DirectCCDOptics("moe")
    config = {"task": "eurosat_paired_rgb_sar", "architecture": "moe",
              "activation_order": "center_out",
              "source_sha256": {"trainval": "train", "holdout": "holdout"}}
    with torch.no_grad():
        original.global_phase.fill_(0.25)
    path = tmp_path / "best_checkpoint.pt"
    torch.save({"model": original.state_dict(), "config": config}, path)
    restored = DirectCCDOptics("moe")
    load_initial_checkpoint(restored, path, config, torch.device("cpu"))
    assert torch.equal(restored.global_phase, original.global_phase)
    changed = {**config, "source_sha256": {**config["source_sha256"], "holdout": "other"}}
    try:
        load_initial_checkpoint(restored, path, changed, torch.device("cpu"))
    except ValueError as error:
        assert "different" in str(error)
    else:
        raise AssertionError("different data source was accepted")


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


def test_physical_binary_pairs_keep_video_fixed_and_balance_description(monkeypatch):
    class FakePhysical:
        def __init__(self, roots, split):
            self.original = np.tile(np.arange(10), 2)

        def labels(self):
            return self.original

        def __getitem__(self, indices):
            video = torch.ones(len(indices), 224, 224)
            video[:, :, :112] *= torch.as_tensor(indices).float()[:, None, None] + 1
            return video

    monkeypatch.setattr(t16_data, "PhysicalRank10Fields", FakePhysical)
    pairs = PhysicalBinaryPairs({}, "train")
    assert len(pairs) == 40
    assert np.array_equal(np.bincount(pairs.candidate_descriptions[::2], minlength=10),
                          np.bincount(pairs.candidate_descriptions[1::2], minlength=10))
    assert np.all(pairs.candidate_descriptions[::2] !=
                  pairs.candidate_descriptions[1::2])
    field = pairs.get_batch(np.array([0, 1, 2, 3]), "cpu")
    assert torch.allclose(field[0, :, :112], field[1, :, :112])
    assert torch.allclose(field[2, :, :112], field[3, :, :112])
    assert not torch.allclose(field[0, :, 112:], field[1, :, 112:])
    assert pairs.labels[:4].tolist() == [1, 0, 1, 0]


def test_raw_physical_video_omits_signed_frame_subtraction(monkeypatch):
    class FakePhysical:
        def __init__(self, roots, split):
            self.fields = [np.ones((2, 224, 224), dtype=np.float32)]
            self.offsets = [0, 2]

        def labels(self):
            return np.array([0, 1])

    monkeypatch.setattr(t16_data, "PhysicalRank10Fields", FakePhysical)
    pairs = PhysicalBinaryPairs({}, "train", video_mode="raw")
    field = pairs.get_batch(np.array([0, 1]), "cpu")
    assert torch.equal(field[0, :, :112], field[1, :, :112])
    assert torch.all(field[:, :, :112] >= 0)
    assert torch.allclose(field.square().sum((-2, -1)), torch.ones(2), atol=1e-5)


def test_eurosat_indexed_field_supports_validation_audit_batching():
    class Indexed:
        labels = np.array([0, 1])

        def __len__(self):
            return 2

        def __getitem__(self, indices):
            return torch.as_tensor(indices)[:, None, None].float().expand(-1, 224, 224)

    data = _IndexedFieldAdapter(Indexed())
    field = data.get_batch(np.array([1, 0]), "cpu")
    assert field.shape == (2, 224, 224)
    assert field[0, 0, 0] == 1 and field[1, 0, 0] == 0


def test_corner_detector_candidates_have_ten_valid_shared_windows():
    model = DirectCCDOptics("moe")
    centers = [(76, 76), (950, 950), (76, 950), (894, 188),
               (76, 188), (950, 838), (188, 838), (950, 76),
               (188, 132), (838, 894)]
    model.set_output_windows(centers, 64)
    assert len(set(model.output_centers)) == 10
