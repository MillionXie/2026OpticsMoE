from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    FairElectronicAmplitudeRouter,
    OpticalDetectorTopKRouter,
    sparsify_probabilities,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.geometry import (
    Aperture,
)


class _TinyGeometry:
    """Small, contract-equivalent geometry for fast CPU FFT tests."""

    canvas_size = 48
    active_size = 40
    expert_size = 16
    num_experts = 4

    @property
    def active_aperture(self) -> Aperture:
        margin = (self.canvas_size - self.active_size) // 2
        return Aperture(
            margin,
            margin + self.active_size,
            margin,
            margin + self.active_size,
        )


def _settings(
    *,
    top_k: int = 2,
    normalization: str = "legacy_l1",
    straight_through: bool = False,
    score_normalization: str = "log_energy_fraction",
) -> SimpleNamespace:
    return SimpleNamespace(
        top_k=top_k,
        router_pool_size=4,
        router_temperature=1.0,
        router_input_layernorm_enabled=True,
        router_input_layernorm_eps=1.0e-5,
        router_noise_std=0.0,
        router_gate_init_std=0.01,
        router_weight_normalization=normalization,
        router_straight_through=straight_through,
        optical_router_energy_eps=1.0e-12,
        optical_router_input_shift_pixels=0,
        optical_router_phase_shift_pixels=0,
        optical_router_ccd_shift_pixels=0,
        optical_router_phase_dropout_p=0.0,
        optical_router_phase_dropout_block_size=4,
        optical_router_capture_loss_scale=0.1,
        optical_router_score_normalization=score_normalization,
        # Row-major detector order must be TL, TR, BL, BR.
        optical_router_detector_intervals=((8, 14), (26, 32)),
        language_optical_pixel_pitch_um=17.0,
        language_optical_distance_m=0.10,
        language_optical_wavelength_nm=532.0,
        language_optical_k_space_enabled=False,
        language_optical_theta_max_deg=0.65,
    )


def _input_fields(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260901)
    value = torch.rand(
        batch,
        _TinyGeometry.expert_size,
        _TinyGeometry.expert_size,
        generator=generator,
    )
    # Break rotational/reflection symmetry so a zero gradient cannot be hidden
    # by the symmetric four-spot initialization.
    value[:, :5, 2:9] *= 1.8
    value[:, 9:, 11:] *= 0.35
    return value.add(0.05)


def _expected_selection(
    probabilities: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.topk(probabilities, top_k, dim=-1).indices
    selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(
        1, indices, True
    )
    return selected, indices


@pytest.mark.parametrize("normalization", ["legacy_l1", "power_l2"])
@pytest.mark.parametrize("straight_through", [False, True])
def test_sparsify_probabilities_has_exact_sparse_forward_contract(
    normalization: str, straight_through: bool
) -> None:
    probabilities = torch.tensor(
        [[0.10, 0.40, 0.20, 0.30], [0.55, 0.05, 0.15, 0.25]],
        requires_grad=True,
    )
    weights, selected, indices = sparsify_probabilities(
        probabilities,
        2,
        normalization=normalization,
        straight_through=straight_through,
    )

    expected_selected, expected_indices = _expected_selection(probabilities, 2)
    assert torch.equal(indices, expected_indices)
    assert torch.equal(selected, expected_selected)
    assert torch.count_nonzero(weights.masked_select(~selected)) == 0

    sparse = probabilities.detach() * expected_selected
    if normalization == "legacy_l1":
        expected = sparse / sparse.sum(dim=-1, keepdim=True)
        norm = weights.detach().sum(dim=-1)
    else:
        expected = sparse / sparse.square().sum(dim=-1, keepdim=True).sqrt()
        norm = weights.detach().square().sum(dim=-1).sqrt()
    torch.testing.assert_close(weights.detach(), expected)
    torch.testing.assert_close(norm, torch.ones_like(norm))


def test_top1_straight_through_keeps_sparse_forward_and_dense_gradient() -> None:
    probabilities = torch.tensor(
        [[0.10, 0.40, 0.20, 0.30]], requires_grad=True
    )
    weights, selected, _ = sparsify_probabilities(
        probabilities,
        1,
        normalization="legacy_l1",
        straight_through=True,
    )
    torch.testing.assert_close(weights.detach(), selected.float())

    coefficients = weights.new_tensor([[0.2, -0.7, 1.4, 0.5]])
    (weights * coefficients).sum().backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad) > 1


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("normalization", ["legacy_l1", "power_l2"])
def test_fair_electronic_router_obeys_topk_and_weight_norm(
    top_k: int, normalization: str
) -> None:
    router = FairElectronicAmplitudeRouter(
        _TinyGeometry(),
        _settings(top_k=top_k, normalization=normalization),
    ).eval()
    output = router(_input_fields())

    probabilities = output["probabilities"]
    weights = output["weights"]
    expected_selected, expected_indices = _expected_selection(probabilities, top_k)
    assert tuple(probabilities.shape) == (3, 4)
    assert torch.equal(output["selected_mask"], expected_selected)
    assert torch.equal(output["selected_indices"], expected_indices)
    assert torch.count_nonzero(weights.masked_select(~expected_selected)) == 0
    if normalization == "legacy_l1":
        norm = weights.sum(dim=-1)
    else:
        norm = weights.square().sum(dim=-1).sqrt()
    torch.testing.assert_close(norm, torch.ones_like(norm))


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("normalization", ["legacy_l1", "power_l2"])
def test_optical_router_output_is_finite_and_exactly_topk(
    top_k: int, normalization: str
) -> None:
    router = OpticalDetectorTopKRouter(
        _TinyGeometry(),
        _settings(top_k=top_k, normalization=normalization),
    ).eval()
    output = router(_input_fields())

    probabilities = output["probabilities"]
    weights = output["weights"]
    expected_selected, expected_indices = _expected_selection(probabilities, top_k)
    assert tuple(output["logits"].shape) == (3, 4)
    assert tuple(probabilities.shape) == (3, 4)
    assert tuple(weights.shape) == (3, 4)
    assert tuple(output["detector_energy"].shape) == (3, 4)
    assert tuple(output["detector_energy_fraction"].shape) == (3, 4)
    assert tuple(output["capture_fraction"].shape) == (3,)
    assert tuple(output["selected_indices"].shape) == (3, top_k)
    assert tuple(router.last_detector_intensity.shape) == (3, 40, 40)

    for key in (
        "logits",
        "probabilities",
        "weights",
        "detector_energy",
        "detector_energy_fraction",
        "capture_fraction",
        "capture_loss",
        "balance_loss",
        "importance_loss",
        "normalized_entropy",
    ):
        assert torch.isfinite(output[key]).all(), key
    assert (output["detector_energy"] >= 0).all()
    assert (output["capture_fraction"] >= 0).all()
    assert (output["capture_fraction"] <= 1).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(3), rtol=1.0e-5, atol=1.0e-6
    )
    assert torch.equal(output["selected_mask"], expected_selected)
    assert torch.equal(output["selected_indices"], expected_indices)
    assert (output["selected_mask"].sum(dim=-1) == top_k).all()
    assert torch.count_nonzero(weights.masked_select(~expected_selected)) == 0
    if normalization == "legacy_l1":
        norm = weights.sum(dim=-1)
    else:
        norm = weights.square().sum(dim=-1).sqrt()
    torch.testing.assert_close(norm, torch.ones_like(norm), rtol=1.0e-5, atol=1.0e-6)


def test_optical_router_detector_regions_are_tl_tr_bl_br_and_in_range() -> None:
    router = OpticalDetectorTopKRouter(_TinyGeometry(), _settings()).eval()
    assert router.detector_bounds == (
        (8, 8, 14, 14),
        (26, 8, 32, 14),
        (8, 26, 14, 32),
        (26, 26, 32, 32),
    )

    masks = router.detector_masks
    assert tuple(masks.shape) == (4, 40, 40)
    assert torch.equal(masks, masks.bool().to(masks.dtype))
    assert (masks.sum(dim=(-2, -1)) == 36).all()
    assert masks.sum(dim=0).amax() == 1
    for left, top, right, bottom in router.detector_bounds:
        assert 0 <= left < right <= router.active_size
        assert 0 <= top < bottom <= router.active_size

    phase = router.phase()
    active_phase = router.active_phase()
    margin = (router.active_size - router.input_size) // 2
    assert tuple(phase.shape) == (16, 16)
    assert tuple(active_phase.shape) == (40, 40)
    torch.testing.assert_close(
        active_phase[margin : margin + 16, margin : margin + 16], phase
    )
    assert torch.count_nonzero(active_phase[:margin]) == 0


def test_optical_router_top1_task_gradient_reaches_router_phase() -> None:
    router = OpticalDetectorTopKRouter(
        _TinyGeometry(),
        _settings(top_k=1, normalization="power_l2", straight_through=True),
    ).train()
    output = router(_input_fields(batch=2))
    coefficients = output["weights"].new_tensor([0.2, -0.9, 1.3, 0.6])
    loss = (output["weights"] * coefficients).sum()
    loss.backward()

    gradient = router.raw_router_phase.grad
    assert gradient is not None
    assert tuple(gradient.shape) == (16, 16)
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


@pytest.mark.parametrize(
    "score_normalization",
    ["log_energy_fraction", "standardized_region_energy"],
)
def test_optical_router_is_invariant_to_global_amplitude_gain(
    score_normalization: str,
) -> None:
    router = OpticalDetectorTopKRouter(
        _TinyGeometry(),
        _settings(score_normalization=score_normalization),
    ).eval()
    fields = _input_fields(batch=2)
    gain = 2.5
    reference = router(fields)
    amplified = router(gain * fields)

    # Propagation is linear in field amplitude, hence CCD energy scales by a^2;
    # routing is computed from relative/standardized energies and must not move.
    torch.testing.assert_close(
        amplified["detector_energy"],
        gain**2 * reference["detector_energy"],
        rtol=2.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        amplified["detector_energy_fraction"],
        reference["detector_energy_fraction"],
        rtol=2.0e-5,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        amplified["probabilities"],
        reference["probabilities"],
        rtol=2.0e-5,
        atol=1.0e-6,
    )
    assert torch.equal(amplified["selected_mask"], reference["selected_mask"])
