"""Structural/semantic CPU tests, not dataset performance evidence."""
import pytest
import torch
from torch import nn

from LightGenV2.tasks.t19_resnet20_baselines.models import (
    BasicBlock, ByteTextEncoder, OptionAShortcut, ResNet20Encoder,
    TASK_IDS, build_model, parameter_counts, tokenize_text,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_original_cifar_architecture_and_parameter_budget():
    encoder = ResNet20Encoder()
    assert sum(isinstance(m, BasicBlock) for m in encoder.modules()) == 9
    assert sum(isinstance(m, nn.Conv2d) for m in encoder.modules()) == 19
    assert sum(p.numel() for p in encoder.parameters()) == 269072
    assert sum(p.numel() for p in nn.Linear(64, 10).parameters()) + 269072 == 269722
    for m in encoder.modules():
        if isinstance(m, OptionAShortcut):
            assert not list(m.parameters())
    assert sum(p.numel() for p in ByteTextEncoder().parameters()) == 27104


def test_option_a_exact_subsample_and_zero_padding():
    x = torch.randn(2, 16, 7, 9)
    y = OptionAShortcut(16, 32, 2)(x)
    torch.testing.assert_close(y[:, 8:24], x[:, :, ::2, ::2])
    assert torch.count_nonzero(y[:, :8]) == torch.count_nonzero(y[:, 24:]) == 0


def test_text_encoding_padding_invariance_and_unicode():
    encoder = ByteTextEncoder().eval()
    with torch.inference_mode():
        a = encoder(tokenize_text(["更换背景"]))
        b = encoder(tokenize_text(["更换背景", "a much longer product description for testing"]))
    torch.testing.assert_close(a[0], b[0], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="byte limit"):
        tokenize_text(["x" * 513])
    with pytest.raises(ValueError, match="right-contiguous"):
        encoder(torch.tensor([[1, 0, 2]]))


@pytest.mark.parametrize("task", TASK_IDS)
def test_native_resolution_outputs_are_finite_and_small(task):
    model = build_model(task).eval()
    image = torch.rand(1, 3, 256 if task == "abo_conditional_edit" else 224,
                       256 if task == "abo_conditional_edit" else 224)
    tokens = tokenize_text(["Change the background to blue."])
    with torch.inference_mode():
        if task.startswith("lgvq_"):
            output = model(image[:, None].repeat(1, 4, 1, 1, 1))
            assert output["quality_logits"].shape == (1, 5)
            assert 1 <= output["quality_score"].item() <= 5
        elif task in {"openmoji", "abo_conditional_edit"}:
            output = model(image, tokens)
            if task == "openmoji":
                assert output["category_logits"].shape == (1, 17, 6, 6)
                assert output["edit_logits"].shape == (1, 6, 6)
                assert output["task_logits"].shape == (1, 4)
            else:
                assert output.shape == (1, 3, 256, 256)
                assert output.min() >= 0 and output.max() <= 1
        else:
            output = model(image)
            if task in {"lsp", "salicon"}:
                assert output.shape == ((1, 14, 56, 56) if task == "lsp" else (1, 1, 224, 224))
            else:
                assert output.shape == (1, 64)
                torch.testing.assert_close(output.norm(dim=-1), torch.ones(1))
                if hasattr(model, "encode_text"):
                    text = model.encode_text(tokens)
                    torch.testing.assert_close(text.norm(dim=-1), torch.ones(1))
        values = output.values() if isinstance(output, dict) else [output]
        assert all(torch.isfinite(x).all() for x in values)
    counts = parameter_counts(model)
    assert counts["total"] == counts["trainable"] < 500000
    assert counts["total"] == counts["visual_encoder"] + counts["text_encoder"] + counts["task_heads"]


@pytest.mark.parametrize("task", TASK_IDS)
def test_synthetic_gradients_reach_visual_and_text_encoders(task):
    model = build_model(task)
    image = torch.rand(2, 3, 32, 32)
    tokens = tokenize_text(["blue background", "replace object"])
    if task.startswith("lgvq_"):
        output = model(torch.rand(2, 4, 3, 32, 32))["quality_logits"]
    elif task in {"openmoji", "abo_conditional_edit"}:
        output = model(image, tokens)
        if isinstance(output, dict):
            output = output["category_logits"]
    else:
        output = model(image)
        if hasattr(model, "encode_text"):
            output = output @ model.encode_text(tokens).T
    # A nonconstant synthetic objective; no optimizer updates or real labels.
    weights = torch.linspace(-1, 1, output.numel()).reshape_as(output)
    (output * weights).mean().backward()
    assert model.encoder.stem[0].weight.grad.abs().sum() > 0
    if hasattr(model, "text_encoder"):
        assert model.text_encoder.embedding.weight.grad.abs().sum() > 0


def test_spatial_order_invariance_temporal_order_sensitivity():
    frames = torch.rand(2, 4, 3, 32, 32)
    spatial = build_model("lgvq_spatial").eval()
    temporal = build_model("lgvq_temporal").eval()
    with torch.inference_mode():
        torch.testing.assert_close(spatial(frames)["quality_score"],
                                   spatial(frames.flip(1))["quality_score"])
        assert not torch.allclose(temporal(frames)["quality_logits"],
                                  temporal(frames.flip(1))["quality_logits"])
    with pytest.raises(ValueError, match="four RGB"):
        temporal(frames[:, :3])


def test_quality_scale_and_bins_use_explicit_train_range():
    model = build_model("lgvq_spatial")
    boundaries = model.set_train_score_range(torch.tensor([10., 20., 30.]))
    torch.testing.assert_close(model.quality_levels, torch.tensor([10., 15., 20., 25., 30.]))
    torch.testing.assert_close(boundaries, torch.tensor([14., 18., 22., 26.]))
    # Out-of-TRAIN-range test MOS is bucketed without changing TRAIN bins.
    assert torch.bucketize(torch.tensor([-10., 100.]), boundaries, right=False).tolist() == [0, 4]
    with pytest.raises(ValueError, match="range"):
        model.set_train_score_range(torch.ones(3))
