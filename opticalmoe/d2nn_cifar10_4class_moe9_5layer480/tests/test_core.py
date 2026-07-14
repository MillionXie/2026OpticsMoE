import sys
from pathlib import Path

from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
OPTICALMOE_ROOT=ROOT.parent
for value in [str(ROOT),str(OPTICALMOE_ROOT)]:
    if value not in sys.path:sys.path.insert(0,value)

from layout import MoELayout
from model import OpticalMoEClassifier, SquareDetectionLayerNormReload
from data import PerClassEpochSampler, RemappedCIFARSubset
from utils import load_yaml
from slm_bmp import export_plane_bmp, nearest_scale_uint8
from prompt import InputTopKRouter
from train import build_optimizer


def config():return load_yaml(ROOT/"configs"/"config.yaml")


def test_layout_matches_requested_480_450_3x3_geometry():
    layout=MoELayout();layout.validate()
    assert layout.active_start==15
    assert layout.expert_centers==[(y,x) for y in (90,240,390) for x in (90,240,390)]
    expected_starts=[(y,x) for y in (30,180,330) for x in (30,180,330)]
    assert [(v.y0,v.x0) for v in layout.expert_apertures]==expected_starts
    assert all(v.y1-v.y0==120 and v.x1-v.x0==120 for v in layout.expert_apertures)


def test_single_config_has_valid_phase_routing_and_regularization_defaults():
    value=load_yaml(ROOT/"configs"/"config.yaml")
    assert value["optics"]["phase_init"]=="zeros"
    assert value["prompt"]["top_k"]==3
    assert value["prompt"]["mode"]=="region_amplitude_global_lens"
    assert value["optics"]["k_space_constraint_enabled"] is False
    assert float(value["optimizer"]["weight_decay"])==0.0
    assert value["model"]["canvas_size"]==480
    assert value["optics"]["pixel_size_m"]==16e-6
    assert value["optics"]["distances_m"]["input_to_prompt"]==0.1444
    assert value["optics"]["distances_m"]["prompt_to_expert"]==0.1444
    assert value["optics"]["prompt_focal_length_m"]==0.0722
    assert value["dataset"]["resize_interpolation"]=="bicubic"
    assert value["dataset"]["resize_antialias"] is True
    assert value["training"]["phase_diagnostics_enabled"] is False
    assert float(value["loss"].get("router_importance_weight",0.0)) >= 0.0


def test_adamw_optimizer_is_honored_and_importance_loss_has_expected_limits():
    value=config();value["optimizer"].update({"type":"adamw","lr":0.005,"weight_decay":0.0})
    model=OpticalMoEClassifier(value,4)
    optimizer=build_optimizer(model,value)
    assert isinstance(optimizer,torch.optim.AdamW)
    router=InputTopKRouter(num_experts=9,top_k=3,pool_size=2)
    with torch.no_grad():
        router.gate.weight.zero_();router.gate.bias.zero_()
    uniform=router(torch.ones(8,1,4,4))
    assert float(uniform["importance_loss"])<1e-6
    with torch.no_grad():router.gate.bias[0]=20.0
    collapsed=router(torch.ones(8,1,4,4))
    assert float(collapsed["importance_loss"])>7.9
    assert float(collapsed["normalized_entropy"])<0.01


def test_optoelectronic_20cm_config_and_square_detection_reload():
    value=load_yaml(ROOT/"configs"/"config_optoelectronic_interlayers_20cm.yaml")
    assert value["optoelectronic_interlayers"]["enabled"] is True
    assert value["optics"]["distances_m"]["inter_layer"]==0.20
    assert value["optics"]["distances_m"]["last_expert_to_global_fc"]==0.20
    assert value["optics"]["distances_m"]["global_fc_to_detector"]==0.20
    model=OpticalMoEClassifier(value,4)
    assert model.optoelectronic_enabled is True
    assert sum(parameter.numel() for parameter in model.interlayer_conversion.parameters())==0
    field=torch.randn(2,12,12,dtype=torch.complex64,requires_grad=True)
    reloaded,details=SquareDetectionLayerNormReload()(field,return_details=True)
    assert reloaded.shape==field.shape and torch.is_complex(reloaded)
    assert torch.all(details["detector_intensity"]>=0)
    assert torch.all(details["reloaded_amplitude"]>=0)
    assert torch.allclose(reloaded.imag,torch.zeros_like(reloaded.imag))
    assert torch.allclose(details["layer_normalized"].mean((-2,-1)),torch.zeros(2),atol=1e-5)
    reloaded.abs().square().mean().backward()
    assert field.grad is not None and torch.count_nonzero(field.grad)>0


def test_model_has_nine_experts_global_fc_and_only_electronic_router():
    model=OpticalMoEClassifier(config(),4)
    assert len(model.expert_layers)==5
    assert all(len(layer.experts)==9 for layer in model.expert_layers)
    assert model.expert_phase_parameter_count()==9*5*120*120
    assert model.global_fc_parameter_count()==450*450
    assert model.optical_parameter_count()==850500
    assert model.router_parameter_count()==909
    assert model.electronic_parameter_count()==909
    assert model.prompt.phase_biases.requires_grad is False
    assert tuple(model.prompt.phase_map().shape)==(480,480)


def test_invalid_global_convolution_geometry_is_rejected():
    value=config();value["optics"]["prompt_focal_length_m"]=0.30
    try:OpticalMoEClassifier(value,4)
    except ValueError as error:assert "global fan-out convolution geometry" in str(error)
    else:raise AssertionError("Expected invalid prompt imaging geometry to fail")


def test_detector_is_centered_in_480_canvas():
    model=OpticalMoEClassifier(config(),4)
    bounds=[]
    for mask in model.detector.masks:
        points=mask.nonzero();bounds.append((int(points[:,0].min()),int(points[:,0].max()+1),int(points[:,1].min()),int(points[:,1].max()+1)))
    assert bounds==[(115,165,115,165),(115,165,315,365),(315,365,115,165),(315,365,315,365)]


def test_ten_class_config_has_centered_ten_detector_readout():
    value=load_yaml(ROOT/"configs"/"config_cifar10_10class.yaml")
    model=OpticalMoEClassifier(value,10)
    assert value["dataset"]["class_indices"]==list(range(10))
    assert value["dataset"]["train_samples_per_class"] is None
    assert model.detector.masks.shape==(10,480,480)
    assert value["detector"]["N_det_sets"]==[3,3,4]
    bounds=[]
    for mask in model.detector.masks:
        points=mask.nonzero();bounds.append((int(points[:,0].min()),int(points[:,0].max()+1),int(points[:,1].min()),int(points[:,1].max()+1)))
    assert bounds[:3]==[(140,180,140,180),(140,180,220,260),(140,180,300,340)]
    assert bounds[3:6]==[(220,260,140,180),(220,260,220,260),(220,260,300,340)]
    assert bounds[6:]==[(300,340,100,140),(300,340,180,220),(300,340,260,300),(300,340,340,380)]
    assert value["dataset"]["train_samples_per_class_per_epoch"]==1000
    importance=load_yaml(ROOT/"configs"/"config_cifar10_10class_importance_adamw.yaml")
    assert importance["optimizer"]["type"]=="adamw"
    assert importance["loss"]["router_importance_weight"]==0.1


def test_rotating_per_class_epoch_sampler_retains_and_covers_full_dataset():
    class FakeDataset:
        targets=[0]*5+[1]*5
        def __len__(self):return len(self.targets)
        def __getitem__(self,index):return torch.zeros(1,2,2),self.targets[index]
    dataset=RemappedCIFARSubset(FakeDataset(),[0,1],samples_per_class=None,seed=3)
    sampler=PerClassEpochSampler(dataset,samples_per_class=2,seed=11)
    epochs=[list(iter(sampler)) for _ in range(3)]
    assert len(dataset)==10 and all(len(epoch)==4 for epoch in epochs)
    for epoch in epochs:
        labels=[dataset.targets[index] for index in epoch]
        assert labels.count(0)==2 and labels.count(1)==2
    assert set(index for epoch in epochs for index in epoch)==set(range(10))


def test_input_100_plus_padding_is_centered_in_480():
    model=OpticalMoEClassifier(config(),4)
    image=torch.zeros(1,1,120,120);image[:,:,10:110,10:110]=1
    canvas=model.prepare_canvas_input(image).real
    assert tuple(canvas.shape)==(1,480,480)
    assert torch.all(canvas[:,190:290,190:290]==1)
    assert int(torch.count_nonzero(canvas).item())==100*100


def test_raw_input_resize_uses_smooth_bicubic_then_zero_padding():
    model=OpticalMoEClassifier(config(),4)
    image=torch.zeros(1,1,2,2);image[0,0,0,1]=1
    canvas=model.prepare_canvas_input(image).real
    crop=canvas[:,180:300,180:300]
    assert torch.all(crop[:,:10]==0) and torch.all(crop[:,-10:]==0)
    assert torch.all(crop[:,:,:10]==0) and torch.all(crop[:,:,-10:]==0)
    resized=crop[:,10:110,10:110]
    assert float(resized.min())>=0.0 and float(resized.max())<=1.0
    assert torch.unique(resized).numel()>2


def test_forward_is_pure_optical_four_class_output():
    model=OpticalMoEClassifier(config(),4).eval()
    with torch.no_grad():logits,items=model(torch.rand(1,1,120,120),return_intermediates=True)
    assert tuple(logits.shape)==(1,4)
    assert torch.all(logits>=0)
    assert tuple(items["detector_intensity"].shape)==(1,480,480)
    assert len(items["after_each_expert_layer"])==5
    assert tuple(items["global_fc_phase"].shape)==(450,450)
    assert int(items["routing_selected_mask"][0].sum())==3
    assert torch.count_nonzero(items["routing_weights"][0]).item()==3
    assert torch.allclose(items["routing_weights"][0].sum(),torch.tensor(1.0),atol=1e-6)
    amplitude=items["prompt_amplitude"][0]
    for index,weight in enumerate(items["routing_weights"][0]):
        row=index//3;col=index%3;y0=15+row*150;x0=15+col*150
        assert torch.allclose(amplitude[y0:y0+150,x0:x0+150],torch.full((150,150),weight),atol=1e-6)


def test_routing_weights_change_amplitude_cells_but_never_prompt_phase():
    model=OpticalMoEClassifier(config(),4).eval()
    first=torch.zeros(1,9);first[0,0]=1
    second=torch.zeros(1,9);second[0,4]=1
    phase_before=model.prompt.phase_map().clone()
    transmission_first=model.prompt.transmission(first)
    transmission_second=model.prompt.transmission(second)
    assert torch.equal(model.prompt.phase_map(),phase_before)
    assert not torch.allclose(transmission_first.abs(),transmission_second.abs())
    assert torch.allclose(transmission_first.abs()[0,15:165,15:165],torch.ones(150,150),atol=1e-6)
    assert torch.count_nonzero(transmission_first.abs()[0,15:165,165:315]).item()==0


def test_prompt_phase_is_one_global_formula_not_nine_local_phase_blocks():
    model=OpticalMoEClassifier(config(),4).eval();prompt=model.prompt
    n=prompt.layout.canvas_size;axis=(torch.arange(n,dtype=torch.float64)-n//2)*prompt.pixel_size_m
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    expected=torch.remainder(-torch.pi/(prompt.wavelength_m*prompt.focal_length_m)*(x.square()+y.square()),2*torch.pi).float()*prompt.active_mask
    assert torch.allclose(prompt.phase_map(),expected,atol=2e-5)
    assert not hasattr(prompt,"lens_phases")
    assert not hasattr(prompt,"grating_phases")


def test_global_fanout_convolution_preserves_batch_and_canvas_shape():
    model=OpticalMoEClassifier(config(),4).eval();field=torch.rand(2,480,480).to(torch.complex64)
    weights=torch.full((2,9),1/9);kernel=model.prompt.transmission(weights)
    output=model.global_fanout_convolution(field,kernel)
    assert output.shape==field.shape and torch.is_complex(output)


def test_equal_prompt_amplitudes_reach_nine_experts_with_equal_energy():
    model=OpticalMoEClassifier(config(),4).eval()
    image=torch.zeros(1,1,120,120);image[:,:,35:85,45:75]=1
    field=model.prepare_canvas_input(image);weights=torch.full((1,9),1/9)
    entrance=model.global_fanout_convolution(field,model.prompt.transmission(weights))
    ratios=model.expert_energy_ratios(entrance)[0]
    assert float(ratios.max()-ratios.min())<0.01
    assert float(ratios.sum())>0.95


def test_slm_bmp_is_1920_by_1200_and_centered(tmp_path):
    source=torch.ones(450,450)
    info=export_plane_bmp(source,tmp_path/"plane.bmp","amplitude",2,1920,1200)
    image=Image.open(tmp_path/"plane.bmp")
    assert image.size==(1920,1200)
    assert info["scaled_shape"]==[900,900]
    pixels=torch.frombuffer(bytearray(image.tobytes()),dtype=torch.uint8).view(1200,1920)
    assert torch.all(pixels[150:1050,510:1410]==255)
    assert int(torch.count_nonzero(pixels[:150]).item())==0


def test_slm_scaling_is_exact_nearest_neighbor_replication():
    source=torch.tensor([[0,64],[128,255]],dtype=torch.uint8)
    scaled=nearest_scale_uint8(source,2)
    expected=torch.tensor([[0,0,64,64],[0,0,64,64],[128,128,255,255],[128,128,255,255]],dtype=torch.uint8)
    assert torch.equal(scaled,expected)
