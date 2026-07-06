from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.bdd100k_timeofday3_optical5_vs_cnn.data import DataBundle
from experiments.bdd100k_timeofday3_optical5_vs_cnn.data_prepare import normalize_timeofday_label
from experiments.bdd100k_timeofday3_optical5_vs_cnn.metrics import classification_metrics,write_json
from experiments.bdd100k_timeofday3_optical5_vs_cnn.models import ElectronicCNNTimeOfDayBaseline,Optical5ContinuousTimeOfDayClassifier,Optical5EnhancedTimeOfDayClassifier
from experiments.bdd100k_timeofday3_optical5_vs_cnn.optics import ClassRegionDetector,ContinuousOpticalPropagationLayer,OpticalDetectionIntensityLayer
from experiments.bdd100k_timeofday3_optical5_vs_cnn.training import train_model
from experiments.bdd100k_timeofday3_optical5_vs_cnn.settings import PhaseDropoutSettings
from experiments.bdd100k_timeofday3_optical5_vs_cnn.sampling import EpochClassMixedSampler


def test_timeofday_normalization()->None:
    assert normalize_timeofday_label("dawn/dusk")=="dawn_dusk"
    assert normalize_timeofday_label(" daytime ")=="daytime"
    assert normalize_timeofday_label("undefined") is None


def test_intensity_layer_has_no_detector_sqrt()->None:
    layer=OpticalDetectionIntensityLayer(8,10,532,17,5);output=layer(torch.rand(2,8,8))
    assert output.shape==(2,8,8) and torch.all(output>=0)
    assert torch.allclose(output.mean(dim=(-2,-1)),torch.ones(2),atol=1e-4)
    assert "sqrt" not in inspect.getsource(OpticalDetectionIntensityLayer.forward)


def test_phase_dropout_is_training_only()->None:
    cfg=PhaseDropoutSettings(enabled=True,p=0.5,block_size=2,batch_shared=True,start_epoch=1)
    layer=OpticalDetectionIntensityLayer(8,10,532,17,5,phase_dropout=cfg)
    layer.train();layer.set_phase_dropout_active(True);layer(torch.rand(2,8,8))
    assert layer.last_phase_dropout_mask is not None and layer.last_phase_dropout_mask.shape==(1,8,8)
    layer.eval();layer(torch.rand(2,8,8));assert layer.last_phase_dropout_mask is None


def test_optical_classifier_shape()->None:
    model=Optical5EnhancedTimeOfDayClassifier(field_size=16,padding_size=20,readout_channels=[4,8],readout_pool_size=2,readout_hidden_dim=8,detector_region_size=4)
    logits,aux=model(torch.rand(2,1,224,224),return_aux=True)
    assert logits.shape==(2,3) and aux["region_logits"].shape==(2,3)


def test_continuous_layer_preserves_complex_field_without_detection()->None:
    layer=ContinuousOpticalPropagationLayer(8,10,532,17,5);field=torch.complex(torch.rand(2,8,8),torch.zeros(2,8,8));output=layer(field)
    assert output.shape==(2,8,8) and torch.is_complex(output)
    source=inspect.getsource(ContinuousOpticalPropagationLayer.forward)
    assert ".abs()" not in source and "relu" not in source and "square" not in source


def test_continuous_classifier_shape_and_phase_gradients()->None:
    model=Optical5ContinuousTimeOfDayClassifier(field_size=16,padding_size=20,readout_channels=[4,8],readout_pool_size=2,readout_hidden_dim=8,detector_region_size=4)
    logits,aux=model(torch.rand(2,1,16,16),return_aux=True);loss=nn.CrossEntropyLoss()(logits,torch.tensor([0,2]))+nn.CrossEntropyLoss()(aux["region_logits"],torch.tensor([0,2]));loss.backward()
    assert logits.shape==(2,3)
    assert all(layer.phase_mask.grad is not None and float(layer.phase_mask.grad.abs().sum())>0 for layer in model.layers)


def test_class_region_detector_layout()->None:
    detector=ClassRegionDetector(16,["daytime","night","dawn_dusk"],4);assert detector.region_masks.shape==(3,16,16);assert torch.all(detector.region_masks.sum(0)<=1)
    intensity=torch.zeros(3,16,16)
    for index,box in enumerate(detector.boxes):intensity[index,box["y0"]:box["y1"],box["x0"]:box["x1"]]=1
    result=detector(intensity);assert result["region_logits"].argmax(1).tolist()==[0,1,2];assert torch.allclose(result["detector_fraction"],torch.ones(3))


def test_cnn_shape()->None:
    model=ElectronicCNNTimeOfDayBaseline([8,12,16,24],.1,3)
    assert model(torch.rand(1,1,224,224)).shape==(1,3)


def test_metrics()->None:
    result=classification_metrics([0,0,1,1,2],[0,1,1,1,2],["daytime","night","dawn_dusk"])
    assert result["top1_accuracy"]==.8
    assert len(result["confusion_matrix"])==3
    assert 0<=result["macro_f1"]<=1 and 0<=result["balanced_accuracy"]<=1


def test_epoch_sampler_mixes_classes_and_rotates_coverage()->None:
    labels=[0]*8+[1]*8+[2]*8;sampler=EpochClassMixedSampler(range(24),labels,3,6,42,4)
    sampler.set_epoch(1);first=list(sampler);sampler.set_epoch(2);second=list(sampler)
    assert len(first)==len(second)==12
    for start in range(0,12,6):assert {labels[index] for index in first[start:start+6]}=={0,1,2}
    for class_index in range(3):assert set(index for index in first if labels[index]==class_index).isdisjoint(index for index in second if labels[index]==class_index)


def test_json_writer_serializes_paths(tmp_path:Path)->None:
    output=tmp_path/"config.json";write_json(output,{"data_root":tmp_path/"data"})
    assert str(tmp_path/"data") in output.read_text(encoding="utf-8")


class TinyDataset(Dataset):
    def __init__(self):self.labels=[0,1,2,0,1,2]
    def __len__(self):return len(self.labels)
    def __getitem__(self,index):return torch.rand(1,4,4),self.labels[index],index,f"image_{index}.jpg"


class TinyModel(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(16,3))
    def forward(self,value):return self.net(value)


def test_one_epoch_writes_live_outputs(tmp_path:Path)->None:
    dataset=TinyDataset();data=DataBundle(dataset,dataset,dataset,["daytime","night","dawn_dusk"],{})
    settings=SimpleNamespace(batch_size=2,num_workers=0,seed=42,learning_rate=.001,weight_decay=0.0,epochs=1,
        log_interval_batches=10,save_predictions_interval_epochs=1,save_interval_epochs=1,output_dir=tmp_path)
    train_model(TinyModel(),data,settings,torch.device("cpu"))
    for path in ("metrics/training_history.csv","metrics/training_latest.json","checkpoints/best.pt","checkpoints/last.pt"):
        assert (tmp_path/path).is_file()
    assert "train_detector_region_loss" in (tmp_path/"metrics/training_history.csv").read_text(encoding="utf-8")
