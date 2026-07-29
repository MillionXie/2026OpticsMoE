from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.datasets import (
    _duts_pairs,
    paired_saliency_transform,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.io_utils import (
    atomic_torch_save,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.modeling import (
    CCDResidualRecombiner,
    LoadedVisionBackbone,
    OpticalVisionBackbone,
    optical_parameter_breakdown,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.objectives import (
    feature_distillation_loss,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.pca import (
    FixedPCAProjection,
    fit_pca_matrix,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.teacher_cache import (
    TeacherTargetStore,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_formal_config_encodes_requested_three_stage_architecture() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "coco_duts_pretrain.yaml")
    assert settings.image_size == 224
    assert settings.expert_layers == 3
    assert settings.num_experts == 16
    assert settings.top_k == 4
    assert settings.detector_output_size == 224
    assert settings.pca_rank == 224
    assert settings.recombiner_alpha_init == pytest.approx(0.1)
    assert settings.duts_head_warmup_epochs == 5
    assert settings.duts_optical_learning_rate == pytest.approx(1e-4)
    assert settings.duts_recombiner_learning_rate == pytest.approx(2e-4)
    assert settings.duts_head_learning_rate == pytest.approx(1e-3)


def test_recombiner_matches_exact_residual_formula() -> None:
    module = CCDResidualRecombiner(
        4,
        alpha_init=0.1,
        alpha_trainable=True,
        layernorm_affine=True,
    )
    with torch.no_grad():
        module.linear.weight.copy_(torch.eye(4))
        module.linear.bias.zero_()
        module.norm.weight.fill_(1.0)
        module.norm.bias.zero_()
    value = torch.tensor([[1.0, 2.0, 4.0, 8.0]], requires_grad=True)
    expected = value + 0.1 * module.norm(value)
    output = module(value)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert module.alpha.grad is not None
    assert module.linear.weight.grad is not None


def test_fixed_pca_has_no_parameters_and_correct_shapes() -> None:
    mean = torch.randn(8)
    components, _ = torch.linalg.qr(torch.randn(8, 4), mode="reduced")
    projection = FixedPCAProjection(mean, components)
    hidden = torch.randn(3, 8, requires_grad=True)
    latent = projection.encode(hidden)
    reconstructed = projection.decode(latent)
    assert latent.shape == (3, 4)
    assert reconstructed.shape == (3, 8)
    assert list(projection.parameters()) == []
    assert not projection.mean.requires_grad
    assert not projection.components.requires_grad
    latent.sum().backward()
    assert hidden.grad is not None
    assert projection.mean.grad is None
    assert projection.components.grad is None


def test_pca_fit_reports_reconstruction_and_variance() -> None:
    torch.manual_seed(7)
    latent = torch.randn(128, 4)
    mixing = torch.randn(4, 8)
    samples = latent @ mixing + 0.01 * torch.randn(128, 8)
    projection, metadata = fit_pca_matrix(
        samples,
        rank=4,
        oversample=2,
        niter=2,
        seed=42,
    )
    assert projection.components.shape == (8, 4)
    assert metadata["explained_variance_ratio_total"] > 0.99
    assert metadata["relative_reconstruction_error"] < 0.05


def test_feature_loss_is_cosine_plus_smooth_l1_and_router_balance() -> None:
    teacher = torch.randn(7, 224)
    student = teacher.clone().requires_grad_(True)
    balance = torch.tensor(1.25, requires_grad=True)
    total, parts = feature_distillation_loss(
        student,
        teacher,
        cosine_weight=1.0,
        smooth_l1_weight=0.5,
        smooth_l1_beta=0.1,
        router_balance=balance,
        router_balance_weight=0.03,
    )
    assert float(parts["cosine_loss"].detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(parts["smooth_l1_loss"].detach()) == pytest.approx(0.0, abs=1e-7)
    assert float(total.detach()) == pytest.approx(0.03 * 1.25)
    total.backward()
    assert student.grad is not None
    assert float(balance.grad) == pytest.approx(0.03)


def test_teacher_target_store_reads_only_valid_rows(tmp_path: Path) -> None:
    directory = tmp_path / "train"
    identity = {"cache_version": 1, "split": "train"}
    write_json(
        directory / "metadata.json",
        {
            "status": "complete",
            "identity": identity,
            "cached_samples": 1,
            "shards": [{"filename": "shard_000000.pt", "samples": 1}],
        },
    )
    write_json(
        directory / "index.json",
        {"train/example": {"shard": "shard_000000.pt", "row": 0}},
    )
    targets = torch.randn(1, 8, 224)
    atomic_torch_save(
        directory / "shard_000000.pt",
        {
            "sample_ids": ["train/example"],
            "targets": targets,
            "lengths": torch.tensor([6]),
            "image_grid_thw": torch.tensor([[1, 2, 3]]),
        },
    )
    store = TeacherTargetStore(
        directory,
        expected_identity=identity,
        lru_shards=1,
    )
    item = store.get("train/example")
    assert item["target"].shape == (6, 224)
    assert item["image_grid_thw"].tolist() == [1, 2, 3]
    torch.testing.assert_close(item["target"], targets[0, :6])


def test_duts_pairing_and_nearest_mask_resize(tmp_path: Path) -> None:
    root = tmp_path / "DUTS-TR"
    image_dir = root / "DUTS-TR-Image"
    mask_dir = root / "DUTS-TR-Mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = Image.fromarray(np.full((11, 17, 3), 128, dtype=np.uint8))
    mask_values = np.zeros((11, 17), dtype=np.uint8)
    mask_values[:, :8] = 255
    mask = Image.fromarray(mask_values)
    image.save(image_dir / "sample.jpg")
    mask.save(mask_dir / "sample.png")
    pairs = _duts_pairs(root, "DUTS-TR")
    assert len(pairs) == 1
    settings = SimpleNamespace(
        image_size=224,
        augmentation_enabled=False,
    )
    resized_image, resized_mask = paired_saliency_transform(
        image,
        mask,
        settings,
        training=False,
    )
    assert resized_image.size == (224, 224)
    assert resized_mask.size == (224, 224)
    assert set(np.unique(np.asarray(resized_mask)).tolist()) <= {0, 255}


def test_backbone_has_three_stages_and_no_hidden_restore_adapter() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "coco_duts_pretrain_smoke.yaml"
    )
    settings.vision_hidden_size = 1024
    fake_visual = _FakeVisual()
    fake_model = _FakeModel(fake_visual)
    loaded = LoadedVisionBackbone(
        model=fake_model,
        visual=fake_visual,
        processor=None,
        device=torch.device("cpu"),
        load_time_sec=0.0,
    )
    backbone = OpticalVisionBackbone(loaded, settings)
    assert len(backbone.core.expert_layers) == 3
    assert not hasattr(backbone.core, "output_adapter")
    assert backbone.recombiner.linear.in_features == 224
    assert backbone.recombiner.linear.out_features == 224
    breakdown = optical_parameter_breakdown(backbone)
    assert breakdown["expert_phase_parameters"] == 3 * 16 * 224 * 224
    assert breakdown["global_phase_parameters"] == 986 * 986
    assert breakdown["removed_hidden_restore_224_to_1024_parameters"] == 0


class _KeywordIdentity(nn.Module):
    def forward(self, hidden: torch.Tensor, **_: object) -> torch.Tensor:
        return hidden


class _FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(1, 1)
        self.blocks = nn.ModuleList([_KeywordIdentity(), _KeywordIdentity()])


class _FakeModel(nn.Module):
    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(depth=2, hidden_size=1024)
        )
