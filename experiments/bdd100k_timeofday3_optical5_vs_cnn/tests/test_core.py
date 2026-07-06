from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.bdd100k_timeofday3_optical5_vs_cnn.data import DataBundle
from experiments.bdd100k_timeofday3_optical5_vs_cnn.data_prepare import normalize_timeofday_label
from experiments.bdd100k_timeofday3_optical5_vs_cnn.metrics import classification_metrics
from experiments.bdd100k_timeofday3_optical5_vs_cnn.models import ElectronicCNNTimeOfDayBaseline,Optical5EnhancedTimeOfDayClassifier
from experiments.bdd100k_timeofday3_optical5_vs_cnn.optics import OpticalDetectionIntensityLayer
from experiments.bdd100k_timeofday3_optical5_vs_cnn.training import train_model


def test_timeofday_normalization()->None:
    assert normalize_timeofday_label("dawn/dusk")=="dawn_dusk"
    assert normalize_timeofday_label(" daytime ")=="daytime"
    assert normalize_timeofday_label("undefined") is None


def test_intensity_layer_has_no_detector_sqrt()->None:
    layer=OpticalDetectionIntensityLayer(8,10,532,17,5);output=layer(torch.rand(2,8,8))
    assert output.shape==(2,8,8) and torch.all(output>=0)
    assert "sqrt" not in inspect.getsource(OpticalDetectionIntensityLayer.forward)


def test_optical_classifier_shape()->None:
    model=Optical5EnhancedTimeOfDayClassifier(field_size=16,padding_size=20,readout_channels=[4,8,8],readout_pool_size=2,readout_hidden_dim=8)
    assert model(torch.rand(2,1,224,224)).shape==(2,3)


def test_cnn_shape()->None:
    model=ElectronicCNNTimeOfDayBaseline([8,12,16,24],.1,3)
    assert model(torch.rand(1,1,224,224)).shape==(1,3)


def test_metrics()->None:
    result=classification_metrics([0,0,1,1,2],[0,1,1,1,2],["daytime","night","dawn_dusk"])
    assert result["top1_accuracy"]==.8
    assert len(result["confusion_matrix"])==3
    assert 0<=result["macro_f1"]<=1 and 0<=result["balanced_accuracy"]<=1


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

