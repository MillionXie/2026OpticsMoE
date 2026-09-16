"""One frozen CNN feature extractor before either optical classifier."""
import importlib.util
from pathlib import Path
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_cnn_definition', TASK/'frozen_electronic/model.py')
cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn)
PhaseOnly = cnn.PhaseOnly


class SharedFrontend(nn.Module):
    def __init__(self, electronic_state):
        super().__init__()
        source = cnn.Electronic(electronic_state['mean'], electronic_state['std'])
        self.register_buffer('mean', source.mean)
        self.register_buffer('std', source.std)
        self.features = source.features
        # The dropout and 128->10 electronic classifier are not part of this model.
        # Accept either the original classifier checkpoint or the saved frontend alone.
        feature_state = {k:v for k,v in electronic_state.items() if k not in {'head.1.weight','head.1.bias'}}
        self.load_state_dict(feature_state, strict=True)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        x = images.permute(0, 3, 1, 2).float()/255.
        return self.features((x-self.mean)/self.std)


def encode_features(features, power=1.):
    """Nonnegative GAP channels 0..127 -> 16x8 cells -> 224x224 amplitude."""
    if features.ndim != 2 or features.shape[1] != 128:
        raise ValueError('Expected Bx128 CNN features')
    if not bool(torch.isfinite(features).all()) or bool((features < 0).any()):
        raise ValueError('CNN amplitude features must be finite and nonnegative')
    # Equal cell areas (14x28), no trainable projection, channel permutation or labels.
    field = features.reshape(-1, 16, 8).repeat_interleave(14, 1).repeat_interleave(28, 2)
    energy = field.square().sum((-2, -1), keepdim=True)
    if bool((energy <= 1e-12).any()):
        raise ValueError('Zero feature power')
    return field * (power/energy).sqrt()


class FrontendOptics(nn.Module):
    def __init__(self, frontend, architecture, optical_config):
        super().__init__()
        self.frontend = frontend.requires_grad_(False).eval()
        self.optical = PhaseOnly(architecture, optical_config)

    def forward(self, images):
        features = self.frontend(images)
        amplitude = encode_features(features, self.optical.cfg['main_input_power'])
        result = self.optical.forward_amplitude(amplitude)
        result['features'] = features
        return result
