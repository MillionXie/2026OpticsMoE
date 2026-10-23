"""Task decoders on a frozen official ImageNet ResNet-18 visual encoder.

Dense tasks use all four spatial stages rather than enlarging a 7x7 map.
This decoder is a new, explicitly disclosed profile; legacy heads are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .frozen_resnet18 import FrozenResNet18


FROZEN_VISUAL_TASKS = ('lgvq_temporal', 'lgvq_spatial', 'abo_image_to_image', 'lsp', 'salicon')


class FrozenImageRetrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenResNet18()
        self.projection = nn.Linear(512, 64)

    def forward(self, images):
        return F.normalize(self.projection(self.encoder(images)), dim=-1)


class FrozenDenseModel(nn.Module):
    def __init__(self, outputs, output_size):
        super().__init__()
        self.encoder = FrozenResNet18()
        self.lateral = nn.ModuleList(nn.Conv2d(channels, 32, 1)
                                     for channels in (64, 128, 256, 512))
        self.head = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, outputs, 1))
        self.output_size = output_size

    def forward(self, images):
        features = self.encoder.features(images)
        size = features[0].shape[-2:]
        fused = torch.stack([F.interpolate(project(x), size, mode='bilinear', align_corners=False)
                             for x, project in zip(features, self.lateral)]).sum(0)
        return F.interpolate(self.head(fused), (self.output_size, self.output_size),
                             mode='bilinear', align_corners=False)


def build_frozen_visual_model(task):
    if task == 'abo_image_to_image':
        return FrozenImageRetrieval()
    if task == 'lsp':
        return FrozenDenseModel(14, 56)
    if task == 'salicon':
        return FrozenDenseModel(1, 224)
    raise ValueError('Frozen pretrained profile currently supports pure visual tasks only; '
                     'text encoder protocol requires an explicit choice')
