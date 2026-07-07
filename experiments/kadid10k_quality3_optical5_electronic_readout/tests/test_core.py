from __future__ import annotations

import csv
import inspect
from pathlib import Path

import torch
from PIL import Image

from experiments.kadid10k_quality3_optical5_electronic_readout.data import load_data
from experiments.kadid10k_quality3_optical5_electronic_readout.models import Optical5ElectronicReadoutQualityClassifier,build_model
from experiments.kadid10k_quality3_optical5_electronic_readout.optics import OpticalDetectionIntensityLayer
from experiments.kadid10k_quality3_optical5_electronic_readout.settings import Settings,load_settings


def make_data(tmp_path:Path)->Path:
    root=tmp_path/"kadid";(root/"images").mkdir(parents=True);rows=[]
    for ref in range(10):
        for level in range(1,10):
            name=f"r{ref:02d}_{level}.png";Image.new("RGB",(16,16),(level*20,ref*10,100)).save(root/"images"/name)
            rows.append({"distorted_image":name,"ref_image":f"r{ref:02d}.png","dmos":level,"distortion_level":min(level,5),"distortion_type":"synthetic"})
    with (root/"dmos.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return root


def tiny_settings(tmp_path:Path)->Settings:
    return Settings(data_root=make_data(tmp_path),output_dir=tmp_path/"run",input_size=16,optical_field_size=16,
        optical_padding_size=20,detector_region_size=4,readout_channels=[4,8],readout_pool_size=2,
        readout_hidden_dim=16,batch_size=3,num_workers=0,epochs=1,save_interval_epochs=1,
        log_interval_batches=1,train_limit_per_class=6,test_limit_per_class=3,
        train_samples_per_class_per_epoch=3,device="cpu")


def test_config_parsing()->None:
    settings=load_settings(Path(__file__).parents[1]/"configs"/"kadid10k_quality3.json")
    assert settings.class_names==["high_quality","medium_quality","low_quality"]
    assert settings.optical_layers==5 and settings.model_type=="optical5_electronic_readout"
    assert settings.optical_field_size==64 and settings.optical_padding_size==128


def test_data_split_is_reference_disjoint(tmp_path:Path)->None:
    settings=tiny_settings(tmp_path);data=load_data(settings)
    assert data.metadata["reference_disjoint_train_validation_test"]
    assert data.class_names==["high_quality","medium_quality","low_quality"]


def test_intensity_layer_normalizes_and_has_no_sqrt_reload()->None:
    layer=OpticalDetectionIntensityLayer(16,20,532,8,5,"zeros",False,None)
    output=layer(torch.rand(2,16,16))
    assert output.shape==(2,16,16) and torch.all(output>=0)
    assert "sqrt" not in inspect.getsource(OpticalDetectionIntensityLayer.forward)


def test_optical5_electronic_readout_shape(tmp_path:Path)->None:
    model=build_model(tiny_settings(tmp_path));output,diagnostics=model(torch.rand(2,1,16,16),return_diagnostics=True)
    assert isinstance(model,Optical5ElectronicReadoutQualityClassifier)
    assert output.shape==(2,3) and len(diagnostics["after_layers"])==5
    assert diagnostics["detector_input"].shape==(2,16,16)


def test_model_has_small_two_convolution_readout(tmp_path:Path)->None:
    model=build_model(tiny_settings(tmp_path));convolutions=[module for module in model.readout.modules() if isinstance(module,torch.nn.Conv2d)]
    assert len(convolutions)==2
    assert all(parameter.requires_grad for parameter in model.parameters())
