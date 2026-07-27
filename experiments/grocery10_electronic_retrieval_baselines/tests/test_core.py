from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from experiments.grocery10_electronic_retrieval_baselines.modeling import (
    ElectronicRetrievalEncoder,
    parameter_report,
)
from experiments.grocery10_electronic_retrieval_baselines.settings import (
    load_settings,
)
from experiments.grocery10_electronic_retrieval_baselines.training import (
    load_checkpoint,
    save_checkpoint,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    gallery_retrieval_loss,
    supervised_contrastive_loss,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_all_formal_configs_use_the_same_replacement_ten_skus() -> None:
    values = []
    for name in (
        "resnet18_imagenet.yaml",
        "efficientnet_b0_imagenet.yaml",
        "mobilenet_v3_small_imagenet.yaml",
    ):
        settings = load_settings(EXPERIMENT / "configs" / name)
        values.append(settings.selected_skus)
        assert settings.weights == "imagenet1k"
        assert settings.embedding_dim == 64
        assert settings.output_dir.is_relative_to(EXPERIMENT)
    assert values[0] == values[1] == values[2]
    assert "Garant-Ecological-Standard-Milk" not in values[0]
    assert "Bravo-Apple-Juice" not in values[0]


def test_electronic_backbones_return_signed_normalized_64d_embeddings() -> None:
    for name in ("resnet18", "efficientnet_b0", "mobilenet_v3_small"):
        model = ElectronicRetrievalEncoder(name, 64, weights="none")
        model.eval()
        output = model(torch.randn(2, 3, 64, 64))
        assert output.shape == (2, 64)
        assert torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-5)
        retrieval_head_modules = list(model.feature_norm.modules()) + list(
            model.projection.modules()
        )
        assert not any(
            isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Softmax))
            for module in retrieval_head_modules
        )


def test_parameter_scale_order_and_head_breakdown() -> None:
    reports = {
        name: parameter_report(ElectronicRetrievalEncoder(name, 64, weights="none"))
        for name in ("resnet18", "efficientnet_b0", "mobilenet_v3_small")
    }
    assert (
        reports["mobilenet_v3_small"]["parameters"]
        < reports["efficientnet_b0"]["parameters"]
        < reports["resnet18"]["parameters"]
    )
    for report in reports.values():
        assert report["retrieval_head_parameters"] == (
            report["feature_norm_parameters"] + report["projection_parameters"]
        )
        assert report["parameters"] == report["trainable_parameters"]


def test_retrieval_losses_backpropagate_through_electronic_model() -> None:
    model = ElectronicRetrievalEncoder("mobilenet_v3_small", 64, weights="none")
    model.train()
    query = model(torch.randn(6, 3, 64, 64))
    gallery = model(torch.randn(3, 3, 64, 64))
    query_labels = torch.tensor([0, 0, 1, 1, 2, 2])
    gallery_labels = torch.tensor([0, 1, 2])
    values = torch.cat((query, gallery))
    labels = torch.cat((query_labels, gallery_labels))
    loss = supervised_contrastive_loss(values, labels, 0.07)
    loss = loss + 0.25 * gallery_retrieval_loss(
        query,
        query_labels,
        gallery,
        gallery_labels,
        0.15,
        stop_gradient_on_gallery=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    settings = load_settings(EXPERIMENT / "configs" / "mobilenet_v3_small_smoke.yaml")
    source = ElectronicRetrievalEncoder("mobilenet_v3_small", 64, weights="none")
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source, optimizer, None, 3, 1.25, settings)
    target = ElectronicRetrievalEncoder("mobilenet_v3_small", 64, weights="none")
    payload = load_checkpoint(path, target)
    assert payload["epoch"] == 3
    for left, right in zip(source.state_dict().values(), target.state_dict().values()):
        assert torch.equal(left, right)
