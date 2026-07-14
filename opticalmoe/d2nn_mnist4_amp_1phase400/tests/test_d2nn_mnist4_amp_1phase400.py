import math
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OPTICALMOE_ROOT = ROOT.parent
if str(OPTICALMOE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTICALMOE_ROOT))

from model import D2NNClassifier
from data import RemappedDigitSubset, mnist_transform
from optics import AngularSpectrumPropagator, PhaseLayer
from train_d2nn_mnist256 import forward_and_loss
from slm_bmp import export_plane_bmp
from utils import load_yaml


def base_config(phase_init="zeros"):
    return {
        "dataset": {"class_digits": [0, 1, 2, 3], "input_size": 400},
        "optics": {
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 16.0e-6,
            "canvas_size": 400,
            "input_size": 400,
            "phase_mask_size": 400,
            "phase_mask_mode": "full_canvas",
            "num_layers": 1,
            "phase_param": "sigmoid",
            "phase_init": phase_init,
            "init_std": 0.02,
            "evanescent_mode": "zero",
            "input_to_layer_distance_m": 0.0,
            "inter_layer_distance_m": 0.03,
            "detector_distance_m": 0.20,
        },
        "detector": {
            "detector_size": 50,
            "layout": "fixed_2x2",
            "start_pos_x": 75,
            "start_pos_y": 75,
            "N_det_sets": [2, 2],
            "det_steps_x": [150, 150],
            "det_steps_y": 150,
            "normalize_detector_energy": True,
        },
        "readout": {"type": "detector_only"},
        "regularization": {"phase_dropout": {"enabled": False}},
    }


def test_model_is_one_full_canvas_phase_layer_and_detector_only():
    model = D2NNClassifier(base_config(), num_classes=4)
    assert model.canvas_size == 400
    assert model.input_size == 400
    assert model.phase_mask_size == 400
    assert len(model.phase_layers) == 1
    assert tuple(model.phase_layers[0].raw_phase.shape) == (400, 400)
    assert model.phase_mask_region() == [0, 400, 0, 400]
    assert model.optical_parameter_count() == 400 * 400
    assert model.electronic_parameter_count() == 0


def test_zero_phase_baseline_disables_weight_decay():
    config = load_yaml(ROOT / "configs" / "config.yaml")
    assert float(config["optimizer"]["weight_decay"]) == 0.0


def test_forward_outputs_four_detector_energies_without_electronic_readout():
    model = D2NNClassifier(base_config(), num_classes=4)
    x = torch.rand(2, 1, 400, 400)
    logits, intermediates = model(x, return_intermediates=True)
    assert tuple(logits.shape) == (2, 4)
    assert tuple(intermediates["detector_energies"].shape) == (2, 4)
    assert torch.allclose(logits, intermediates["detector_energies"])
    assert torch.all(logits >= 0)


def test_fixed_detector_layout_matches_requested_regions():
    model = D2NNClassifier(base_config(), num_classes=4)
    masks = model.detector.masks
    # The notebook treats det_steps=150 as the empty gap after each 50-pixel
    # square: next start = 75 + 50 + 150 = 275.
    expected = [(75, 75), (75, 275), (275, 75), (275, 275)]
    for index, (y0, x0) in enumerate(expected):
        assert masks[index, y0 : y0 + 50, x0 : x0 + 50].sum().item() == 50 * 50
        assert masks[index].sum().item() == 50 * 50


def test_sigmoid_phase_constraint_is_between_zero_and_two_pi():
    for init in ["zeros", "uniform", "gaussian"]:
        layer = PhaseLayer(400, parameterization="sigmoid", init=init, init_std=0.02)
        phase = layer.get_phase()
        assert torch.all(phase >= 0)
        assert torch.all(phase <= 2.0 * math.pi)


def test_phase_initializations_are_raw_phase_initializations():
    zero = PhaseLayer(32, parameterization="sigmoid", init="zeros", init_std=0.02)
    uniform = PhaseLayer(32, parameterization="sigmoid", init="uniform", init_std=0.02)
    gaussian = PhaseLayer(32, parameterization="sigmoid", init="gaussian", init_std=0.02)
    assert torch.allclose(zero.raw_phase, torch.zeros_like(zero.raw_phase))
    assert float(uniform.raw_phase.min().item()) >= 0.0
    assert float(uniform.raw_phase.max().item()) <= 2.0 * math.pi
    assert abs(float(gaussian.raw_phase.mean().item())) < 0.01
    assert 0.01 < float(gaussian.raw_phase.std().item()) < 0.04


def test_zero_sigmoid_initialization_has_nonzero_phase_gradient():
    layer = PhaseLayer(16, parameterization="sigmoid", init="zeros", init_std=0.02)
    field = torch.rand(1, 16, 16, dtype=torch.complex64)
    output = layer(field)
    loss = output.real.sum()
    loss.backward()
    assert layer.raw_phase.grad is not None
    assert float(layer.raw_phase.grad.abs().max().item()) > 0.0


def test_phase_dropout_bypasses_post_sigmoid_modulation_not_raw_parameter():
    layer = PhaseLayer(8, parameterization="sigmoid", init="gaussian", init_std=0.2, phase_dropout_mode="phase_bypass", phase_dropout_p=1.0)
    field = torch.rand(2, 8, 8, dtype=torch.float32).to(torch.complex64)
    raw_before = layer.raw_phase.detach().clone()
    layer.train();output = layer(field)
    assert torch.equal(output, field)
    assert torch.equal(layer.raw_phase.detach(), raw_before)
    layer.eval();eval_output = layer(field)
    assert not torch.equal(eval_output, field)


def test_notebook_style_input_is_bicubic_336_then_zero_padded_to_400():
    transform = mnist_transform(
        {
            "preprocess_mode": "resize_then_pad",
            "resize_size": 336,
            "input_size": 400,
            "interpolation": "bicubic",
        }
    )
    image = Image.new("L", (28, 28), color=255)
    output = transform(image)
    assert tuple(output.shape) == (1, 400, 400)
    assert torch.count_nonzero(output[:, :32]).item() == 0
    assert torch.count_nonzero(output[:, -32:]).item() == 0
    assert torch.count_nonzero(output[:, :, :32]).item() == 0
    assert torch.count_nonzero(output[:, :, -32:]).item() == 0
    assert torch.allclose(output[:, 32:368, 32:368], torch.ones(1, 336, 336))


def test_per_class_sampling_is_balanced_and_configurable():
    class FakeDataset:
        targets = torch.tensor([0] * 10 + [1] * 9 + [2] * 8 + [3] * 7)

        def __getitem__(self, index):
            return torch.tensor(float(index)), int(self.targets[index])

    subset = RemappedDigitSubset(FakeDataset(), [0, 1, 2, 3], samples_per_class=5, seed=7)
    labels = [subset[index][1] for index in range(len(subset))]
    assert len(subset) == 20
    assert [labels.count(index) for index in range(4)] == [5, 5, 5, 5]


def test_zero_distance_propagation_is_exact_identity():
    layer = AngularSpectrumPropagator(532e-9, 16e-6, 32, 0.0)
    field = torch.randn(2, 32, 32, dtype=torch.complex64)
    assert torch.equal(layer(field), field)


def test_unshifted_propagation_matches_notebook_shifted_fft_convention():
    wavelength = 532e-9
    pixel_size = 16e-6
    distance = 0.2
    size = 32
    current = AngularSpectrumPropagator(wavelength, pixel_size, size, distance)
    field = torch.rand(2, size, size, dtype=torch.float32).to(torch.complex64)
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(size, d=pixel_size, dtype=torch.float64))
    fy, fx = torch.meshgrid(frequencies, frequencies, indexing="ij")
    argument = (2.0 * math.pi) ** 2 * ((1.0 / wavelength) ** 2 - fx.square() - fy.square())
    kz = torch.sqrt(argument.clamp_min(0.0))
    transfer_shifted = torch.exp(1j * kz * distance).to(torch.complex64)
    spectrum_shifted = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    notebook = torch.fft.ifft2(torch.fft.ifftshift(spectrum_shifted * transfer_shifted, dim=(-2, -1)))
    assert torch.allclose(current(field), notebook, atol=2e-5, rtol=2e-5)


def test_full_detector_plane_mse_backpropagates_to_phase():
    model = D2NNClassifier(base_config(), num_classes=4)
    images = torch.rand(1, 1, 400, 400)
    labels = torch.tensor([2])
    _, loss = forward_and_loss(model, images, labels, {"type": "detector_plane_mse", "scale": 100.0})
    loss.backward()
    gradient = model.phase_layers[0].raw_phase.grad
    assert gradient is not None
    assert float(gradient.abs().sum().item()) > 0.0


def test_k_space_constraint_filters_high_angle_frequencies():
    unconstrained = AngularSpectrumPropagator(532e-9, 16e-6, 64, 0.2, k_space_constraint_enabled=False)
    constrained = AngularSpectrumPropagator(
        532e-9,
        16e-6,
        64,
        0.2,
        k_space_constraint_enabled=True,
        theta_max_deg=0.5,
    )
    assert torch.all(unconstrained.k_space_mask)
    assert 0.0 < constrained.k_space_pass_fraction < 1.0
    assert constrained.max_sampled_angle_deg > constrained.theta_max_deg
    field = torch.rand(1, 64, 64, dtype=torch.float32).to(torch.complex64)
    assert not torch.allclose(unconstrained(field), constrained(field))


def test_angle_above_sampled_maximum_matches_disabled_constraint():
    unconstrained = AngularSpectrumPropagator(532e-9, 16e-6, 64, 0.2, k_space_constraint_enabled=False)
    wide = AngularSpectrumPropagator(
        532e-9,
        16e-6,
        64,
        0.2,
        k_space_constraint_enabled=True,
        theta_max_deg=2.0,
    )
    assert wide.k_space_pass_fraction == 1.0
    assert torch.equal(unconstrained.transfer_function, wide.transfer_function)


def test_nonzero_input_to_layer_distance_performs_real_propagation():
    zero = AngularSpectrumPropagator(532e-9, 16e-6, 64, 0.0)
    nonzero = AngularSpectrumPropagator(532e-9, 16e-6, 64, 0.03)
    field = torch.rand(1, 64, 64, dtype=torch.float32).to(torch.complex64)
    assert torch.equal(zero(field), field)
    assert not torch.allclose(nonzero(field), field)


def test_baseline_slm_bmp_export_scales_400_to_800_and_centers(tmp_path):
    from PIL import Image
    info = export_plane_bmp(torch.ones(400, 400), tmp_path / "phase.bmp", "phase", 2, 1920, 1200)
    assert Image.open(tmp_path / "phase.bmp").size == (1920, 1200)
    assert info["scaled_shape"] == [800, 800]
