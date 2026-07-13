import sys
from pathlib import Path

from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
OPTICALMOE_ROOT=ROOT.parent
for value in [str(ROOT),str(OPTICALMOE_ROOT)]:
    if value not in sys.path:sys.path.insert(0,value)

from layout import MoELayout
from model import OpticalMoEClassifier
from utils import load_yaml
from slm_bmp import export_plane_bmp, nearest_scale_uint8


def config():return load_yaml(ROOT/"configs"/"config.yaml")


def test_layout_matches_requested_480_450_3x3_geometry():
    layout=MoELayout();layout.validate()
    assert layout.active_start==15
    assert layout.expert_centers==[(y,x) for y in (90,240,390) for x in (90,240,390)]
    expected_starts=[(y,x) for y in (30,180,330) for x in (30,180,330)]
    assert [(v.y0,v.x0) for v in layout.expert_apertures]==expected_starts
    assert all(v.y1-v.y0==120 and v.x1-v.x0==120 for v in layout.expert_apertures)


def test_single_config_defaults_to_zero_phase_top3_and_no_kspace_cutoff():
    value=load_yaml(ROOT/"configs"/"config.yaml")
    assert value["optics"]["phase_init"]=="zeros"
    assert value["prompt"]["top_k"]==3
    assert value["optics"]["k_space_constraint_enabled"] is False
    assert float(value["optimizer"]["weight_decay"])==0.0
    assert value["model"]["canvas_size"]==480
    assert value["optics"]["pixel_size_m"]==16e-6


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


def test_detector_is_centered_in_480_canvas():
    model=OpticalMoEClassifier(config(),4)
    bounds=[]
    for mask in model.detector.masks:
        points=mask.nonzero();bounds.append((int(points[:,0].min()),int(points[:,0].max()+1),int(points[:,1].min()),int(points[:,1].max()+1)))
    assert bounds==[(115,165,115,165),(115,165,315,365),(315,365,115,165),(315,365,315,365)]


def test_input_100_plus_padding_is_centered_in_480():
    model=OpticalMoEClassifier(config(),4)
    image=torch.zeros(1,1,120,120);image[:,:,10:110,10:110]=1
    canvas=model.prepare_canvas_input(image).real
    assert tuple(canvas.shape)==(1,480,480)
    assert torch.all(canvas[:,190:290,190:290]==1)
    assert int(torch.count_nonzero(canvas).item())==100*100


def test_raw_input_resize_uses_nearest_then_zero_padding():
    model=OpticalMoEClassifier(config(),4)
    image=torch.zeros(1,1,2,2);image[0,0,0,1]=1
    canvas=model.prepare_canvas_input(image).real
    crop=canvas[:,180:300,180:300]
    assert torch.all(crop[:,:10]==0) and torch.all(crop[:,-10:]==0)
    assert torch.all(crop[:,:,:10]==0) and torch.all(crop[:,:,-10:]==0)
    resized=crop[:,10:110,10:110]
    assert set(torch.unique(resized).tolist())=={0.0,1.0}


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
