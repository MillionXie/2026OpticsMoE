"""Trainable task decoders with both pretrained encoders strictly frozen."""
import torch
from torch import nn
from torch.nn import functional as F

from .frozen_resnet18 import FrozenResNet18
from .frozen_text import FrozenTextEncoder
from .frozen_visual_models import FrozenImageRetrieval


class FrozenCrossModalRetrieval(FrozenImageRetrieval):
    def __init__(self):
        super().__init__()
        self.text_encoder = FrozenTextEncoder()
        self.text_projection = nn.Linear(384, 64)

    def encode_text(self, tokens):
        return F.normalize(self.text_projection(self.text_encoder(tokens)), dim=-1)


class FrozenOpenMoji(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenResNet18()
        self.text_encoder = FrozenTextEncoder()
        self.image_projection = nn.Conv2d(512, 64, 1)
        self.text_projection = nn.Linear(384, 64)
        self.film = nn.Linear(64, 128)
        self.coordinates = nn.Conv2d(2, 64, 1)
        self.spatial = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.category = nn.Conv2d(64, 17, 1)
        self.edit = nn.Conv2d(64, 1, 1)
        self.task = nn.Linear(64, 4)

    def forward(self, images, tokens):
        x = self.image_projection(F.adaptive_avg_pool2d(self.encoder.features(images)[-1], (6, 6)))
        condition = self.text_projection(self.text_encoder(tokens))
        gamma, beta = self.film(condition).chunk(2, -1)
        x = x * (1 + torch.tanh(gamma)[:, :, None, None]) + beta[:, :, None, None]
        axis = torch.linspace(-1, 1, 6, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        x = self.spatial(x + self.coordinates(torch.stack((xx, yy))[None]))
        return {'category_logits': self.category(x), 'edit_logits': self.edit(x).squeeze(1),
                'task_logits': self.task(condition)}


class FrozenConditionalEditor(nn.Module):
    """Reference+instruction decoder with four frozen visual skip levels."""
    def __init__(self):
        super().__init__()
        self.encoder = FrozenResNet18()
        self.text_encoder = FrozenTextEncoder()
        self.lateral = nn.ModuleList(nn.Conv2d(channels, 64, 1)
                                     for channels in (64, 128, 256, 512))
        self.text_projection = nn.Linear(384, 64)
        self.film = nn.Linear(64, 128)
        self.decode = nn.ModuleList(nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                                   nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                                   for _ in range(3))
        self.rgb = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 3, 1))

    def forward(self, reference, tokens):
        levels = [project(x) for project, x in zip(self.lateral, self.encoder.features(reference))]
        x = levels[-1]
        gamma, beta = self.film(self.text_projection(self.text_encoder(tokens))).chunk(2, -1)
        x = x * (1 + torch.tanh(gamma)[:, :, None, None]) + beta[:, :, None, None]
        for decoder, skip in zip(self.decode, reversed(levels[:-1])):
            x = F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat((x, skip), 1))
        x = F.interpolate(x, reference.shape[-2:], mode='bilinear', align_corners=False)
        return self.rgb(x).sigmoid()


def build_frozen_multimodal_model(task):
    if task in {'abo_image_to_text', 'abo_text_to_image'}:
        return FrozenCrossModalRetrieval()
    if task == 'openmoji':
        return FrozenOpenMoji()
    if task == 'abo_conditional_edit':
        return FrozenConditionalEditor()
    raise ValueError(f'Unknown frozen pretrained task: {task}')
