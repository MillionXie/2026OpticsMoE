"""A shared frozen CNN probability branch added to the phase-only experiment."""
from pathlib import Path
import sys
import torch
from torch import nn

PURE = Path(__file__).resolve().parents[1] / 'pure_optical'
sys.path.insert(0, str(PURE))
from models import PhaseOnly


class Electronic(nn.Module):
    def __init__(self, mean, std, seed=42):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).float().reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).float().reshape(1, 3, 1, 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 201)
            blocks = []
            for ci, co in [(3, 32), (32, 64), (64, 128)]:
                blocks.extend([nn.Conv2d(ci, co, 3, padding=1, bias=False),
                               nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2)])
            self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(128, 10))

    def forward(self, images):
        x = images.permute(0, 3, 1, 2).float() / 255.
        return self.head(self.features((x - self.mean) / self.std))


class FrozenFusion(nn.Module):
    def __init__(self, electronic, architecture, optical_config):
        super().__init__()
        self.electronic = electronic.requires_grad_(False).eval()
        self.optical = PhaseOnly(architecture, optical_config)

    def train(self, mode=True):
        super().train(mode)
        # Locks both dropout and BatchNorm running statistics, even during training.
        self.electronic.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            electronic = self.electronic(images).softmax(1)
        result = self.optical(images)
        optical = result['probabilities']
        result.update(electronic=electronic, optical=optical,
                      probabilities=.5 * electronic + .5 * optical)
        return result
