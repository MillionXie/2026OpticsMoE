"""Frozen official RN18 + full-instruction byte GRU, for a fixed SDXL tail.

This is an explicitly distilled frontend replacement, not a scratch-trained
ResNet baseline. The old Qwen/optical frontend is absent from raw inference.
"""
import torch
from torch import nn
from ..t19_resnet20_baselines.models import ByteTextEncoder, tokenize_text
from ..t19_resnet20_baselines.frozen_resnet18 import FrozenResNet18
from .qwen3_scene_conditioning import QueryCondition

SCHEMA = 'scene_resnet18_byte_gru_fixed70_v1'
PARENT = 'aa2d277bd743d4dd831ec33852629dbf3001e772b55a8ac436a0d2a557a06fd8'
PAIRS = 'c2e30cbdbd871094df8827c78f4a02bc23305a0e39898ab53c6acf4178835fd0'
REQUESTS = 'f132238c3a59efb1308649909c96ce3669e02c7420d598bb29183536720f9f47'


class ResNetConditionReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = ByteTextEncoder()
        self.project = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 512))
        self.text_projection = nn.Linear(64, 512)
        self.text = QueryCondition(77, 2048)
        self.image = QueryCondition(257, 1280)
        self.pooled = nn.Linear(512, 1280)

    def initialize_readouts(self, parent_head):
        # Only the unchanged output interface is inherited; the RN18/GRU input
        # projections are new. Record this transfer in the scientific protocol.
        for name in ('text', 'image', 'pooled'):
            prefix = name + '.'
            getattr(self, name).load_state_dict(
                {k[len(prefix):]: v for k, v in parent_head.items() if k.startswith(prefix)}, strict=True)

    def forward(self, visual, tokens):
        if visual.ndim != 3 or visual.shape[1:] != (64, 512):
            raise ValueError('RN18 layer4 RGB256 must yield 64 spatial tokens of width512')
        context = self.project(visual)
        text_token = self.text_projection(self.text_encoder(tokens))[:, None]
        text, queries = self.text(torch.cat((context, text_token), 1))
        image, _ = self.image(context)
        return dict(text=text, pooled=self.pooled(queries.mean(1)), image=image)


class ResNetSceneFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenResNet18()
        self.readout = ResNetConditionReadout()

    def visual_tokens(self, rgb):
        if rgb.shape[1:] != (3, 256, 256):
            raise ValueError('Same white RGB256 input required')
        features = self.encoder.features(rgb)[-1]
        return features.flatten(2).transpose(1, 2).contiguous()

    def forward(self, rgb, tokens):
        return self.readout(self.visual_tokens(rgb), tokens)


def rgb_tensor(image):
    import numpy as np
    if image.mode != 'RGB' or image.size != (256, 256):
        raise ValueError('Original RGB256 required; no crop or replacement reference')
    value = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    return (value - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def condition_loss(pred, target):
    import torch.nn.functional as F
    parts = {}
    for field, weight in [('text', 1.), ('pooled', .25), ('image', 2.)]:
        actual, wanted = pred[field].float(), target[field].float().detach()
        if actual.shape != wanted.shape:
            raise ValueError('Matched SDXL condition shapes required')
        scale = wanted.square().mean().clamp_min(1e-6)
        parts[field] = weight * ((actual-wanted).square().mean()/scale
                                + .1*(1-F.cosine_similarity(actual, wanted, dim=-1)).mean())
    loss = sum(parts.values())
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite frontend loss')
    return loss


def validate_training_rows(rows):
    if len(rows) != 624 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Exactly the existing624 TRAIN contexts; no TEST gradients')
    if len({r['input_key'] for r in rows}) != 624 or len({r['reference_id'] for r in rows}) != 76:
        raise ValueError('Actual unique TRAIN identities and76 reference views required')
    tokenize_text([r['prompt'] for r in rows])  # fail on truncation, never silently shorten

