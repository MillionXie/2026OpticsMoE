import torch

from opticalmoe.optics import (
    AngularSpectrumPropagator,
    DetectorArray,
    ElectronicReadout,
    OpticalClassifier,
    PhaseLayer,
)


def test_angular_spectrum_preserves_shape():
    prop = AngularSpectrumPropagator(
        wavelength_m=532e-9,
        pixel_size_m=8e-6,
        grid_size=32,
        distance_m=0.05,
    )
    field = torch.ones(2, 32, 32, dtype=torch.complex64)
    out = prop(field)
    assert out.shape == field.shape
    assert out.dtype == torch.complex64


def test_phase_layer_preserves_shape_and_has_parameters():
    layer = PhaseLayer(grid_size=32)
    field = torch.ones(2, 32, 32, dtype=torch.complex64)
    out = layer(field)
    assert out.shape == field.shape
    assert sum(p.numel() for p in layer.parameters()) > 0


def test_phase_layer_parameterizations():
    field = torch.ones(1, 16, 16, dtype=torch.complex64)
    for parameterization in ["unconstrained", "sigmoid", "cos"]:
        layer = PhaseLayer(grid_size=16, parameterization=parameterization)
        out = layer(field)
        wrapped = layer.get_phase_wrapped()
        assert out.shape == field.shape
        assert wrapped.min() >= 0.0
        assert wrapped.max() < 2.0 * torch.pi


def test_phase_layer_initialization_aliases():
    for init in [
        "identity",
        "uniform_0_2pi",
        "small_normal",
        "kaiming_phase",
    ]:
        layer = PhaseLayer(grid_size=16, init=init, init_std=0.01)
        assert layer.raw_phase.shape == (16, 16)
    identity = PhaseLayer(grid_size=8, init="identity")
    assert torch.all(identity.raw_phase == 0.0)


def test_detector_array_shape():
    detector = DetectorArray(num_classes=10, grid_size=32, detector_size=4, layout="grid")
    field = torch.ones(3, 32, 32, dtype=torch.complex64)
    energies = detector(field)
    assert energies.shape == (3, 10)


def test_electronic_readout_shapes():
    energies = torch.rand(4, 10)
    for readout_type in ["optical_only", "linear", "mlp"]:
        readout = ElectronicReadout(num_classes=10, readout_type=readout_type)
        logits = readout(energies)
        assert logits.shape == (4, 10)


def _tiny_classifier(num_layers=2, readout_type="optical_only"):
    return OpticalClassifier(
        num_classes=10,
        input_size=16,
        padding=8,
        grid_size=32,
        num_layers=num_layers,
        detector_size=4,
        readout_type=readout_type,
        distances_m={
            "input_to_prompt": 0.01,
            "prompt_to_first_layer": 0.01,
            "inter_layer": 0.01,
            "last_layer_to_detector": 0.01,
        },
    )


def test_optical_classifier_forward_tiny_batch():
    model = _tiny_classifier(num_layers=2)
    x = torch.rand(2, 1, 16, 16)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_optical_classifier_intermediates_keys():
    model = _tiny_classifier(num_layers=2)
    x = torch.rand(2, 1, 16, 16)
    logits, intermediates = model(x, return_intermediates=True)
    assert logits.shape == (2, 10)
    expected_keys = {
        "input_amplitude",
        "padded_input",
        "after_input_to_prompt",
        "after_prompt",
        "after_prompt_to_first_layer",
        "after_layer_1_modulation",
        "after_layer_1_propagation",
        "after_layer_2_modulation",
        "after_layer_2_propagation",
        "detector_field",
        "detector_intensity",
        "detector_energies",
    }
    assert expected_keys.issubset(set(intermediates.keys()))


def test_num_propagation_segments_equals_num_layers_plus_two():
    model = _tiny_classifier(num_layers=5)
    assert model.num_propagation_segments == model.num_layers + 2
