import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


_model_module = import_file("fulloptical_model_for_tests", ROOT / "model.py")
FullOpticalD2NNClassifier = _model_module.FullOpticalD2NNClassifier
SquareDetectionLayerNormReload = _model_module.SquareDetectionLayerNormReload
load_yaml = import_file("fulloptical_utils_for_tests", ROOT / "utils.py").load_yaml
_train_module = import_file("fulloptical_train_for_tests", ROOT / "train.py")
build_optimizer = _train_module.build_optimizer
detector_plane_mse_loss = _train_module.detector_plane_mse_loss


def load(name):
    return load_yaml(ROOT / "configs" / name)


def centers(detector):
    return torch.stack([mask.nonzero().float().mean(0) for mask in detector.masks])


def test_four_class_config_and_geometry():
    config = load("cifar10_4class.yaml")
    model = FullOpticalD2NNClassifier(config, 4)
    assert config["dataset"]["train_samples_per_class"] is None
    assert config["dataset"]["batch_size"] == 16
    assert model.input_size == 300 and model.canvas_size == 360
    assert len(model.phase_layers) == 6
    assert model.optical_parameter_count() == 6 * 360 * 360
    assert model.electronic_parameter_count() == 0
    assert model.detector.masks.shape == (4, 360, 360)
    assert torch.allclose(centers(model.detector).mean(0), torch.tensor([179.5, 179.5]))


def test_optimizer_type_supports_adam_and_adamw():
    config = load("cifar10_4class.yaml"); model = FullOpticalD2NNClassifier(config, 4)
    config["optimizer"]["type"] = "adamw"
    assert isinstance(build_optimizer(model, config), torch.optim.AdamW)
    config["optimizer"]["type"] = "adam"
    assert isinstance(build_optimizer(model, config), torch.optim.Adam)


def test_optoelectronic_20cm_config_has_five_interlayer_conversions():
    config=load("cifar10_4class_optoelectronic_interlayers_20cm.yaml")
    assert config["optics"]["inter_layer_distance_m"]==0.20
    assert config["optics"]["detector_distance_m"]==0.20
    model=FullOpticalD2NNClassifier(config,4)
    assert model.optoelectronic_enabled is True
    assert config["optoelectronic_interlayers"]["elementwise_affine"] is True
    assert config["optoelectronic_interlayers"]["affine_sharing"]=="per_stage"
    assert config["optoelectronic_interlayers"]["reapply_routing_amplitude"] is False
    assert len(model.interlayer_conversions)==5
    assert model.interlayer_conversion_parameter_count()==5*2*360*360
    assert model.electronic_parameter_count()==1_296_000
    assert all(tuple(module.affine_weight.shape)==(360,360) for module in model.interlayer_conversions)
    assert model.interlayer_conversions[0].affine_weight.data_ptr()!=model.interlayer_conversions[1].affine_weight.data_ptr()
    config["optics"]["inter_layer_distance_m"]=0.0;config["optics"]["detector_distance_m"]=0.0
    model=FullOpticalD2NNClassifier(config,4).eval()
    with torch.no_grad():logits,items=model(torch.rand(1,1,300,300),return_intermediates=True)
    assert logits.shape==(1,4)
    assert len(items["interlayer_detector_intensities"])==5
    assert len(items["interlayer_reloaded_amplitudes"])==5
    assert all(torch.all(value>=0) for value in items["interlayer_detector_intensities"])
    assert all(torch.all(value>=0) for value in items["interlayer_reloaded_amplitudes"])


def test_fulloptical_stage_affine_parameters_receive_gradients():
    module=SquareDetectionLayerNormReload(field_size=12,elementwise_affine=True)
    field=torch.randn(2,12,12,dtype=torch.complex64,requires_grad=True)
    output,details=module(field,return_details=True);output.real.mean().backward()
    assert module.affine_weight.grad is not None and module.affine_bias.grad is not None
    assert details["normalization_scope"]=="spatial_per_sample"
    assert details["routing_amplitude_reapplied"] is False


def test_ten_class_config_and_centered_detector():
    config = load("cifar10_10class.yaml")
    model = FullOpticalD2NNClassifier(config, 10)
    assert config["dataset"]["class_indices"] == list(range(10))
    assert model.detector.masks.shape == (10, 360, 360)
    assert config["detector"]["detector_size"] == 40
    assert config["detector"]["N_det_sets"] == [3, 4, 3]
    bounds=[]
    for mask in model.detector.masks:
        points=mask.nonzero();bounds.append((int(points[:,0].min()),int(points[:,0].max()+1),int(points[:,1].min()),int(points[:,1].max()+1)))
    assert bounds[:3] == [(80,120,90,130),(80,120,160,200),(80,120,230,270)]
    assert bounds[3:7] == [(160,200,55,95),(160,200,125,165),(160,200,195,235),(160,200,265,305)]
    assert bounds[7:] == [(240,280,90,130),(240,280,160,200),(240,280,230,270)]
    assert config["dataset"]["train_samples_per_class_per_epoch"] == 1000


def test_ten_class_optoelectronic_config_inherits_new_detector_and_normalized_mse():
    config = load("cifar10_10class_optoelectronic_interlayers_20cm.yaml")
    model = FullOpticalD2NNClassifier(config, 10)
    assert model.optoelectronic_enabled is True
    assert config["detector"]["detector_size"] == 40
    assert config["detector"]["N_det_sets"] == [3, 4, 3]
    assert config["loss"]["normalize_detector_plane_mse"] is True
    assert config["optics"]["inter_layer_distance_m"] == 0.20
    assert config["optics"]["detector_distance_m"] == 0.20
    assert config["optoelectronic_interlayers"]["elementwise_affine"] is True
    assert model.interlayer_conversion_parameter_count()==1_296_000


def test_detector_plane_mse_can_toggle_energy_normalization():
    intensity=torch.rand(2,8,8);target=torch.zeros_like(intensity);target[:,2:5,3:6]=1
    normalized=detector_plane_mse_loss(intensity,target,100.0,True,1.0e-8)
    normalized_scaled=detector_plane_mse_loss(9*intensity,target,100.0,True,1.0e-8)
    raw=detector_plane_mse_loss(intensity,target,100.0,False,1.0e-8)
    raw_scaled=detector_plane_mse_loss(9*intensity,target,100.0,False,1.0e-8)
    assert torch.allclose(normalized,normalized_scaled,rtol=1e-5,atol=1e-6)
    assert not torch.allclose(raw,raw_scaled)


def test_input_is_resized_and_center_zero_padded():
    config = load("cifar10_4class.yaml")
    model = FullOpticalD2NNClassifier(config, 4)
    canvas = model.prepare_input(torch.ones(1, 1, 300, 300)).real
    assert canvas.shape == (1, 360, 360)
    assert torch.all(canvas[:, 30:330, 30:330] == 1)
    assert int(torch.count_nonzero(canvas)) == 300 * 300


def test_forward_is_detector_only_for_four_and_ten_classes():
    for name, count in (("cifar10_4class.yaml", 4), ("cifar10_10class.yaml", 10)):
        config = load(name)
        # Keep the shape/API test quick while preserving the production config.
        config["optics"]["input_to_layer_distance_m"] = 0.0
        config["optics"]["inter_layer_distance_m"] = 0.0
        config["optics"]["detector_distance_m"] = 0.0
        model = FullOpticalD2NNClassifier(config, count).eval()
        with torch.no_grad():
            logits, items = model(torch.rand(1, 1, 300, 300), return_intermediates=True)
        assert logits.shape == (1, count)
        assert torch.all(logits >= 0)
        assert items["detector_intensity"].shape == (1, 360, 360)
        assert len(items["after_each_layer"]) == 6
