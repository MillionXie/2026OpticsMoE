"""Numerical and P11-boundary checks; run with unittest from the repository root."""

from __future__ import annotations

import copy
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from FFA.optics import (
    axis_propagate,
    axis_transfer_1d,
    install_full_response,
    intensity_forward,
    intensity_response,
    response_vjp,
    snapshot_source_phases,
    uninstall_full_response,
)


def _transfer(size, *, dtype=torch.float64, device="cpu"):
    # Nonunitary propagation including a blocked frequency makes inverse-vs-
    # adjoint errors observable; exact BP must work for this case as well.
    phase = torch.randn(size, dtype=dtype, device=device)
    magnitude = torch.linspace(0.0, 1.0, size, dtype=dtype, device=device)
    return magnitude * torch.exp(1j * phase)


def _electronic_boundary(intensity, amplitude):
    mean = intensity.mean(dim=(-2, -1), keepdim=True)
    variance = intensity.var(dim=(-2, -1), keepdim=True, unbiased=False)
    activated = F.relu((intensity - mean) * torch.rsqrt(variance + 1e-5))
    normalized = activated / activated.square().mean(
        dim=(-2, -1), keepdim=True
    ).add(1e-5).sqrt()
    return 0.6 * normalized + 0.4 * amplitude


class ResponseMathematicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1807)

    def test_single_layer_double_matches_autograd_both_axes(self):
        for axis, size in (("token", 5), ("channel", 6)):
            with self.subTest(axis=axis):
                amplitude = torch.rand(3, 2, 5, 6, dtype=torch.float64).requires_grad_()
                phase = torch.randn(2, 5, 6, dtype=torch.float64).requires_grad_()
                error = torch.randn_like(amplitude)
                transfer = _transfer(size)
                ordinary = intensity_forward(amplitude, phase, transfer, axis)
                expected = torch.autograd.grad(
                    ordinary, (amplitude, phase), error
                )
                custom = intensity_response(amplitude, phase, transfer, axis)
                actual = torch.autograd.grad(custom, (amplitude, phase), error)
                torch.testing.assert_close(custom, ordinary, rtol=0, atol=0)
                for observed, reference in zip(actual, expected):
                    torch.testing.assert_close(observed, reference, rtol=2e-12, atol=2e-12)

    def test_double_gradcheck_and_central_finite_differences(self):
        amplitude = torch.rand(2, 1, 3, 4, dtype=torch.float64).requires_grad_()
        phase = torch.randn(1, 3, 4, dtype=torch.float64).requires_grad_()
        transfer = _transfer(4)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, p: intensity_response(a, p, transfer, "channel"),
            (amplitude, phase), eps=1e-6, atol=2e-7, rtol=1e-5,
        ))
        error = torch.randn_like(amplitude)
        gradients = response_vjp(amplitude, phase, error, transfer, "channel")
        epsilon = 1e-6
        for argument, index, gradient in (
            (amplitude, (1, 0, 2, 1), gradients[0]),
            (phase, (0, 1, 2), gradients[1]),
        ):
            delta = torch.zeros_like(argument)
            delta[index] = epsilon
            if argument is amplitude:
                plus = intensity_forward(amplitude + delta, phase, transfer, "channel")
                minus = intensity_forward(amplitude - delta, phase, transfer, "channel")
            else:
                plus = intensity_forward(amplitude, phase + delta, transfer, "channel")
                minus = intensity_forward(amplitude, phase - delta, transfer, "channel")
            estimate = ((plus - minus) * error).sum() / (2 * epsilon)
            torch.testing.assert_close(estimate, gradient[index], rtol=2e-7, atol=2e-8)

    def test_batch_phase_reduction_and_per_sample_amplitude_credit(self):
        amplitude = torch.rand(4, 2, 5, 6, dtype=torch.float64)
        phase = torch.randn(2, 5, 6, dtype=torch.float64)
        error = torch.randn_like(amplitude)
        transfer = _transfer(5)
        ga, gp = response_vjp(amplitude, phase, error, transfer, "token")
        individual = [
            response_vjp(a[None], phase, e[None], transfer, "token")
            for a, e in zip(amplitude, error)
        ]
        torch.testing.assert_close(ga, torch.cat([g[0] for g in individual]))
        torch.testing.assert_close(gp, torch.stack([g[1] for g in individual]).sum(0))
        # A global phase shift on each independent line never changes intensity,
        # including with nonunitary P and signed detector-plane errors.
        torch.testing.assert_close(
            gp.sum(dim=-2), torch.zeros_like(gp.sum(dim=-2)),
            rtol=0, atol=3e-14,
        )

    def test_two_layers_with_live_electronics_match_double_bp(self):
        amplitude = torch.rand(2, 2, 5, 6, dtype=torch.float64).requires_grad_()
        raw_phases = [
            torch.randn(2, 5, 6, dtype=torch.float64).requires_grad_()
            for _ in range(2)
        ]
        transfers = [_transfer(5), _transfer(6)]
        error = torch.randn_like(amplitude)

        def evaluate(mode):
            value = amplitude
            for axis, raw_phase, transfer in zip(
                ("token", "channel"), raw_phases, transfers
            ):
                phase = 2 * math.pi * torch.sigmoid(raw_phase)
                if mode == "ordinary":
                    intensity = intensity_forward(value, phase, transfer, axis)
                else:
                    intensity = intensity_response(
                        value, phase, transfer, axis, mode=mode,
                        response_phase=phase.detach().clone(),
                    )
                value = _electronic_boundary(intensity, value)
            return value

        ordinary = evaluate("ordinary")
        expected = torch.autograd.grad(ordinary, (amplitude, *raw_phases), error)
        for mode in ("current_response", "source_full_response"):
            with self.subTest(mode=mode):
                output = evaluate(mode)
                actual = torch.autograd.grad(output, (amplitude, *raw_phases), error)
                torch.testing.assert_close(output, ordinary, rtol=0, atol=0)
                for observed, reference in zip(actual, expected):
                    torch.testing.assert_close(observed, reference, rtol=2e-11, atol=2e-11)

    def test_frozen_response_backward_never_uses_current_phase_or_field(self):
        amplitude = torch.rand(2, 2, 5, 6, dtype=torch.float64).requires_grad_()
        source_phase = torch.randn(2, 5, 6, dtype=torch.float64)
        phase_one = torch.randn_like(source_phase).requires_grad_()
        phase_two = (phase_one.detach() + torch.randn_like(source_phase)).requires_grad_()
        transfer = _transfer(5)
        error = torch.randn_like(amplitude)
        saved = []

        def pack(tensor):
            saved.append((tensor.data_ptr(), tensor.dtype, tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output_one = intensity_response(
                amplitude, phase_one, transfer, "token",
                mode="source_full_response", response_phase=source_phase,
            )
        self.assertEqual(len(saved), 3)
        self.assertNotIn(phase_one.data_ptr(), [item[0] for item in saved])
        self.assertEqual(sum(dtype.is_complex for _, dtype, _ in saved), 1)
        self.assertEqual(saved[-1][2], transfer.shape)
        first = torch.autograd.grad(output_one, (amplitude, phase_one), error)
        output_two = intensity_response(
            amplitude, phase_two, transfer, "token",
            mode="source_full_response", response_phase=source_phase,
        )
        second = torch.autograd.grad(output_two, (amplitude, phase_two), error)
        self.assertGreater((output_one - output_two).abs().max().item(), 0.01)
        for g_one, g_two in zip(first, second):
            torch.testing.assert_close(g_one, g_two, rtol=0, atol=0)
        expected = response_vjp(amplitude, source_phase, error, transfer, "token")
        for observed, reference in zip(first, expected):
            torch.testing.assert_close(observed, reference)

    def test_random_response_is_selected_explicitly(self):
        amplitude = torch.rand(2, 1, 5, 6, dtype=torch.float64).requires_grad_()
        phase = torch.randn(1, 5, 6, dtype=torch.float64).requires_grad_()
        random_phase = torch.randn_like(phase)
        error = torch.randn_like(amplitude)
        transfer = _transfer(6)
        output = intensity_response(
            amplitude, phase, transfer, "channel",
            mode="random_full_response", response_phase=random_phase,
        )
        actual = torch.autograd.grad(output, (amplitude, phase), error)
        expected = response_vjp(amplitude, random_phase, error, transfer, "channel")
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference)
        torch.testing.assert_close(output, intensity_forward(amplitude, phase, transfer, "channel"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_float32_current_response_matches_autograd(self):
        for axis in ("token", "channel"):
            amplitude = torch.rand(2, 3, 64, 64, device="cuda").requires_grad_()
            phase = torch.randn(3, 64, 64, device="cuda").requires_grad_()
            transfer = _transfer(64, dtype=torch.float32, device="cuda")
            error = torch.randn_like(amplitude)
            ordinary = intensity_forward(amplitude, phase, transfer, axis)
            expected = torch.autograd.grad(ordinary, (amplitude, phase), error)
            output = intensity_response(amplitude, phase, transfer, axis)
            actual = torch.autograd.grad(output, (amplitude, phase), error)
            for observed, reference in zip(actual, expected):
                torch.testing.assert_close(observed, reference, rtol=5e-5, atol=5e-6)


class _StagePair(nn.Module):
    def __init__(self):
        super().__init__()
        from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
            AxisOpticalOEOStage,
        )
        self.stages = nn.ModuleList([
            AxisOpticalOEOStage(
                optical_axis=axis, token_count=196, size=224, channels=1,
                wavelength_m=5.32e-7, pixel_size_m=1.6e-5, distance_m=0.05,
                phase_init_std=0.10, layernorm_eps=1e-5,
                residual_mode="constrained", residual_main_init=0.6,
                residual_main_min=0.5, normalize_branch_rms=True,
                random_seed=1807 + index, electronic_skip_mode="identity",
            )
            for index, axis in enumerate(("token", "channel"))
        ])

    def forward(self, amplitude):
        for stage in self.stages:
            amplitude = stage(amplitude)
        return amplitude


class P11BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1807)

    def test_one_dimensional_propagation_matches_original_two_dimensional_p11(self):
        backbone = _StagePair()
        field = torch.complex(torch.randn(2, 1, 224, 224), torch.randn(2, 1, 224, 224))
        for stage in backbone.stages:
            transfer = axis_transfer_1d(stage.propagator.transfer_function, stage.optical_axis)
            actual = axis_propagate(field, transfer, stage.optical_axis)
            expected = stage.propagator(field)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        invalid = torch.randn(5, 6, dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError, "not constant"):
            axis_transfer_1d(invalid, "token")

    def test_stage_pair_preserves_forward_gradient_electronics_and_token_layout(self):
        ordinary = _StagePair()
        current = copy.deepcopy(ordinary)
        initial_keys = tuple(current.state_dict())
        install_full_response(current, "current_response")
        self.assertEqual(initial_keys, tuple(current.state_dict()))
        current.load_state_dict(ordinary.state_dict(), strict=True)
        amplitude = torch.rand(2, 1, 224, 224)
        # Explicitly nonzero pad rows exercise the relayed-row preservation.
        amplitude[:, :, 196:, :] += 1.0
        inputs = [amplitude.clone().requires_grad_() for _ in range(2)]
        error = torch.randn_like(amplitude)
        expected_output, output = ordinary(inputs[0]), current(inputs[1])
        torch.testing.assert_close(output, expected_output, rtol=5e-5, atol=6e-6)
        expected = torch.autograd.grad(
            expected_output, (inputs[0], *ordinary.parameters()), error
        )
        actual = torch.autograd.grad(output, (inputs[1], *current.parameters()), error)
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, rtol=3e-4, atol=4e-5)
        # The old class remains untouched for unrelated P11 instances.
        self.assertNotIn("_optical_branch", ordinary.stages[0].__dict__)
        self.assertIn("_optical_branch", current.stages[0].__dict__)
        uninstall_full_response(current)
        restored = current(amplitude)
        torch.testing.assert_close(restored, ordinary(amplitude), rtol=0, atol=0)

    def test_source_buffer_is_independent_and_frozen_after_optimizer_steps(self):
        backbone = _StagePair()
        snapshots = snapshot_source_phases(backbone)
        before = [phase.clone() for phase in snapshots]
        install_full_response(backbone, "source_full_response", snapshots)
        for stage, source in zip(backbone.stages, snapshots):
            self.assertNotEqual(stage.ffa_response_phase.data_ptr(), source.data_ptr())
        optimizer = torch.optim.SGD(backbone.parameters(), lr=0.01)
        input_value = torch.rand(2, 1, 224, 224)
        error = torch.randn_like(input_value)
        raw_before = [stage.raw_phase.detach().clone() for stage in backbone.stages]
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (backbone(input_value) * error).mean().backward()
            optimizer.step()
        for stage, source, original, raw in zip(backbone.stages, snapshots, before, raw_before):
            torch.testing.assert_close(stage.ffa_response_phase, original, rtol=0, atol=0)
            torch.testing.assert_close(source, original, rtol=0, atol=0)
            self.assertGreater((stage.raw_phase.detach() - raw).abs().max().item(), 0)

    def test_random_defaults_and_unsupported_deployment_are_explicit(self):
        backbone = _StagePair()
        install_full_response(backbone, "random_full_response")
        amplitude = torch.rand(1, 1, 224, 224)
        for stage in backbone.stages:
            torch.testing.assert_close(stage.ffa_response_phase, stage.random_phase, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "ideal optics only"):
                stage(amplitude, detector_noise_relative_rms=0.01)
            with self.assertRaisesRegex(ValueError, "ideal optics only"):
                stage(amplitude, phase_override=stage.random_phase)

    def test_failed_install_does_not_partially_mutate_the_model(self):
        backbone = _StagePair()
        snapshots = list(snapshot_source_phases(backbone))
        snapshots[-1] = snapshots[-1][:, :10, :10]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            install_full_response(backbone, "source_full_response", snapshots)
        for stage in backbone.stages:
            self.assertNotIn("_optical_branch", stage.__dict__)
            self.assertNotIn("ffa_response_phase", stage._buffers)


if __name__ == "__main__":
    unittest.main()
