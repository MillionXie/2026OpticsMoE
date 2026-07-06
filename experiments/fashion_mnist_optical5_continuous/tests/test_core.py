from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

from experiments.fashion_mnist_optical5_continuous.models import FashionMNISTOptical5Continuous
from experiments.fashion_mnist_optical5_continuous.optics import ContinuousOpticalPropagationLayer,GridClassRegionDetector
from experiments.fashion_mnist_optical5_continuous.sampling import EpochClassMixedSampler
from experiments.fashion_mnist_optical5_continuous.settings import load_settings


def smoke_settings():return load_settings(Path(__file__).parents[1]/"configs"/"fashion_mnist_smoke.json")


def test_dropout_is_explicitly_disabled()->None:
    settings=smoke_settings();assert not settings.phase_dropout.enabled
    model=FashionMNISTOptical5Continuous(settings);model.train();model.set_epoch(20);model(torch.rand(2,1,28,28))
    assert all(layer.last_phase_dropout_mask is None and not layer.phase_dropout_active for layer in model.layers)


def test_continuous_layer_has_no_intermediate_detection()->None:
    layer=ContinuousOpticalPropagationLayer(8,10,532,17,5);field=torch.complex(torch.rand(2,8,8),torch.zeros(2,8,8));output=layer(field);source=inspect.getsource(ContinuousOpticalPropagationLayer.forward)
    assert torch.is_complex(output) and output.shape==field.shape
    assert ".abs()" not in source and "relu" not in source and "square" not in source


def test_model_shape_and_phase_gradients()->None:
    settings=smoke_settings();model=FashionMNISTOptical5Continuous(settings);logits,aux=model(torch.rand(2,1,28,28),return_aux=True);loss=nn.CrossEntropyLoss()(logits,torch.tensor([0,9]))+nn.CrossEntropyLoss()(aux["region_logits"],torch.tensor([0,9]));loss.backward()
    assert logits.shape==(2,10)
    assert all(layer.phase_mask.grad is not None and torch.isfinite(layer.phase_mask.grad).all() for layer in model.layers)


def test_two_by_five_regions_are_disjoint()->None:
    names=[str(i) for i in range(10)];detector=GridClassRegionDetector(64,names,6);assert detector.region_masks.shape==(10,64,64);assert torch.all(detector.region_masks.sum(0)<=1)


def test_sampler_mixes_all_ten_classes()->None:
    labels=[cls for cls in range(10) for _ in range(4)];sampler=EpochClassMixedSampler(range(40),labels,10,20,42);indices=list(sampler)
    for start in range(0,40,20):assert {labels[index] for index in indices[start:start+20]}==set(range(10))
