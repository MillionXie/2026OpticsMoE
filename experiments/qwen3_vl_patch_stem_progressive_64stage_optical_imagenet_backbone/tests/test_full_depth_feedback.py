from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    rms_normalize,
)

from ..model import QwenStemProgressiveOpticalImageNetBackbone
from .test_progressive_model import common_config, fake_stem


@pytest.mark.parametrize("depth", (16, 32, 64, 100))
def test_random_feedback_has_one_independent_auditable_connector_per_stage(
    tmp_path: Path,
    depth: int,
) -> None:
    config = common_config()
    config.update({"num_stages": depth, "new_stage_alpha_init": 0.02})
    model = QwenStemProgressiveOpticalImageNetBackbone(
        fake_stem(tmp_path / f"stem_{depth}.pt"), config
    )
    model.configure_feedback("fa_random", random_seed=7301)
    manifest = model.feedback_manifest()

    assert manifest["method"] == "fa_random"
    assert manifest["connector_count"] == depth
    assert manifest["internal_interstage_connector_count"] == depth - 1
    assert manifest["adapter_input_connector_count"] == 1
    assert manifest["random_connector_seeds_are_unique"] is True
    assert len(model.feedback_connector_seeds) == depth
    assert len(set(model.feedback_connector_seeds)) == depth
    assert len(manifest["connections"]) == depth
    assert [item["connector_index_zero_based"] for item in manifest["connections"]] == list(
        range(depth)
    )
    assert manifest["connections"][0]["connector_role"] == "adapter_to_stage_input"
    assert all(
        item["connector_scale_control"]
        == "exact_complex_linear_spectrum_and_frobenius_norm_by_unitary_right_factor"
        for item in manifest["connections"]
    )
    assert not any(
        item["real_amplitude_jacobian_spectrum_claimed_equal"]
        for item in manifest["connections"]
    )
    assert all(item["frozen"] for item in manifest["connections"])
    assert all(
        item["random_substream_seed"] is not None
        for item in manifest["connections"]
    )
    assert all(
        slot.stage.feedback_mode == "fa_random" for slot in model.slots
    )
    assert all(
        torch.allclose(
            torch.exp(1j * slot.stage.feedback_phase).abs(),
            torch.ones_like(slot.stage.feedback_phase),
            rtol=0.0,
            atol=2.0e-7,
        )
        for slot in model.slots
    )

    state = model.state_dict()
    assert "feedback_source_phases" in state
    assert not any(name.endswith("feedback_phase") for name in state)
    assert model.feedback_source_phases.requires_grad is False


def test_random_feedback_is_repeatable_and_base_seed_independent_of_model_rng(
    tmp_path: Path,
) -> None:
    config = common_config()
    config.update({"num_stages": 16, "seed": 991})
    model = QwenStemProgressiveOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), config
    )

    model.configure_feedback("fa_random", random_seed=81)
    first = model.feedback_snapshot()
    first_seeds = model.feedback_connector_seeds
    model.configure_feedback("fa_random", random_seed=81)
    torch.testing.assert_close(model.feedback_snapshot(), first, rtol=0.0, atol=0.0)
    assert model.feedback_connector_seeds == first_seeds

    model.configure_feedback("fa_random", random_seed=82)
    second = model.feedback_snapshot()
    second_seeds = model.feedback_connector_seeds
    assert first_seeds != second_seeds
    assert all(left != right for left, right in zip(first_seeds, second_seeds))
    assert all(
        not torch.equal(first[index], second[index]) for index in range(16)
    )


def _backward_audit(
    model: QwenStemProgressiveOpticalImageNetBackbone,
    amplitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    value = amplitude.detach().clone().requires_grad_(True)
    output, _ = model.forward_field(value)
    spatial_weight = torch.linspace(0.2, 1.0, 224).view(1, 1, 224, 1)
    loss = (output.square() * spatial_weight).mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if value.grad is None:
        raise AssertionError("The optical input did not receive a gradient")
    return output.detach(), value.grad.detach().clone(), gradients


def test_source_feedback_is_exact_at_capture_and_random_keeps_all_new_phases_live(
    tmp_path: Path,
) -> None:
    config = common_config()
    config.update(
        {
            "num_stages": 16,
            "new_stage_alpha_init": 0.05,
            "activation_checkpointing": False,
        }
    )
    stem = fake_stem(tmp_path / "stem.pt")
    bp = QwenStemProgressiveOpticalImageNetBackbone(stem, config).eval()
    bp.capture_feedback_source(provenance={"capture": "unit_test_undrifted"})
    fa = QwenStemProgressiveOpticalImageNetBackbone(stem, config).eval()
    fa.load_state_dict(bp.state_dict(), strict=True)

    bp.configure_feedback("bp_current")
    fa.configure_feedback("fa_source")
    source_manifest = fa.feedback_manifest()
    assert source_manifest["source_match_is_exact_at_undrifted_capture"] is True
    assert source_manifest["feedback_equals_current_forward_phase"] is True
    assert source_manifest["runtime_feedback_phase_persistent"] is False

    generator = torch.Generator().manual_seed(1917)
    amplitude = rms_normalize(
        torch.rand(1, 3, 224, 224, generator=generator),
        1.0e-5,
    )
    bp_output, bp_input_gradient, bp_gradients = _backward_audit(bp, amplitude)
    fa_output, fa_input_gradient, fa_gradients = _backward_audit(fa, amplitude)

    torch.testing.assert_close(fa_output, bp_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        fa_input_gradient,
        bp_input_gradient,
        rtol=3.0e-5,
        atol=2.0e-8,
    )
    assert fa_gradients.keys() == bp_gradients.keys()
    for name in bp_gradients:
        torch.testing.assert_close(
            fa_gradients[name],
            bp_gradients[name],
            rtol=5.0e-5,
            atol=2.0e-8,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )

    new_phase_names = {
        f"slots.{slot.stage_index}.stage.raw_phase" for slot in fa.new_slots()
    }
    assert all(name in fa_gradients for name in new_phase_names)
    assert all(torch.isfinite(fa_gradients[name]).all() for name in new_phase_names)
    assert all(float(fa_gradients[name].norm()) > 0.0 for name in new_phase_names)

    fa.configure_feedback("fa_random", random_seed=31337)
    _, _, random_gradients = _backward_audit(fa, amplitude)
    assert all(name in random_gradients for name in new_phase_names)
    assert all(
        bool(torch.isfinite(random_gradients[name]).all())
        for name in new_phase_names
    )
    assert all(
        float(random_gradients[name].norm()) > 0.0 for name in new_phase_names
    )


def test_checkpoint_restores_source_but_requires_runtime_feedback_reconfiguration(
    tmp_path: Path,
) -> None:
    config = common_config()
    config.update({"num_stages": 16, "new_stage_alpha_init": 0.03})
    stem = fake_stem(tmp_path / "stem.pt")
    model = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    model.capture_feedback_source(provenance={"capture": "unit_test_source"})
    source = model.feedback_source_snapshot()
    model.configure_feedback("fa_random", random_seed=20260901)
    random_runtime = model.feedback_snapshot()

    state = model.state_dict()
    clone = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    clone.load_state_dict(state, strict=True)
    assert clone.feedback_method == "bp_current"
    torch.testing.assert_close(
        clone.feedback_source_snapshot(), source, rtol=0.0, atol=0.0
    )
    assert not torch.equal(clone.feedback_snapshot(), random_runtime)

    clone.configure_feedback("fa_source")
    torch.testing.assert_close(
        clone.feedback_snapshot(), source, rtol=0.0, atol=0.0
    )
