"""Two-layer four-expert/full-aperture comparison with identical per-layer OEO."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.demo_check.pure_optical.models import PhaseOnly


class TextEncoder(nn.Module):
    def __init__(self, vocabulary, mode):
        super().__init__()
        self.mode = mode
        if mode == 'learned':
            self.embedding = nn.Embedding(vocabulary, 32, padding_idx=0)
            self.gru = nn.GRU(32, 32, batch_first=True)
        elif mode != 'fixed':
            raise ValueError(mode)

    def forward(self, token_ids):
        mask = token_ids.ne(0).float().unsqueeze(-1)
        if self.mode == 'fixed':
            return F.one_hot(token_ids, num_classes=64).float()*mask
        z, _ = self.gru(self.embedding(token_ids))
        z = z*mask
        return torch.cat((F.relu(z), F.relu(-z)), dim=-1)


def normalize_power(x, power):
    energy = x.square().sum((-2, -1), keepdim=True)
    return x*(power/energy.clamp_min(1e-20)).sqrt()


def encode(images, text):
    """RGB and text share 50/50 power, no fusion before optical propagation."""
    rgb = images.float().permute(0, 3, 1, 2)/255
    rgb = F.interpolate(rgb, (112, 112), mode='bilinear', align_corners=False)
    rgb = rgb*(0.5/rgb.square().sum((1,2,3), keepdim=True).clamp_min(1e-20)).sqrt()
    # 32x64 ->112x112 is nearest-neighbour enlargement, not feature averaging.
    text = F.interpolate(text[:,None], (112,112), mode='nearest')[:,0]
    text = normalize_power(text, 0.5)
    return torch.cat((torch.cat((rgb[:,0], rgb[:,1]), -1),
                      torch.cat((rgb[:,2], text), -1)), -2)


def enlarge_tiles(amplitude, side):
    """No resampling across modality/channel boundaries; restore tile powers."""
    half = amplitude.shape[-1]//2
    result = []
    for y in range(2):
        row = []
        for x in range(2):
            tile = amplitude[:,y*half:(y+1)*half,x*half:(x+1)*half]
            power = tile.square().sum((-2,-1), keepdim=True)
            tile = F.interpolate(tile[:,None], (side//2,side//2), mode='nearest')[:,0]
            row.append(normalize_power(tile, power))
        result.append(torch.cat(row,-1))
    return torch.cat(result,-2)


class OpticalOEO(PhaseOnly):
    def __init__(self, architecture, seed=17):
        cfg_path = Path(__file__).resolve().parents[2]/'demo_check/pure_optical/config.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf8'))
        cfg['seed'] = seed
        cfg.update(protocol='clevr_attribute_text_encoding_pilot_v1',no_intermediate_oeo=False,
                   detector_size=64,input_encoding='RGB tiles plus word-position grid; RGB/text power 0.5 each',
                   oeo='active intensity / spatial mean -> nonaffine LayerNorm -> ReLU -> Softsign -> unit-power amplitude; zero phase',
                   loss='NLL of two normalized detector energies',augmentation='none',
                   selection='minimum validation NLL',no_trainable_electronic_adapter_or_head=False)
        super().__init__('dynamic_four' if architecture == 'moe' else 'full_d2nn', cfg)
        self.class_centers = [(259,179),(259,339)]

    @staticmethod
    def oeo(field):
        # Identical active 478-square detector/SLM area for both models.
        intensity = field[:,20:498,20:498].abs().square()
        intensity = intensity/intensity.mean((-2,-1),keepdim=True).clamp_min(1e-20)
        value = F.softsign(F.relu(F.layer_norm(intensity, (478,478), eps=1e-5)))
        value = normalize_power(value, 1.0)
        return F.pad(value, (20,)*4).to(torch.complex64)

    def forward(self, amplitude):
        q, router_capture = self.route(amplitude)
        if self.architecture == 'full_d2nn':
            expanded = enlarge_tiles(amplitude, 478)
            first = F.pad(expanded*self.transmission(self.first_phase), (20,)*4)
        else:
            first = amplitude.new_zeros((len(amplitude),518,518),dtype=torch.complex64)
            for i,(y,x) in enumerate(self.apertures):
                first[:,y:y+224,x:x+224] = amplitude*q[:,i,None,None].sqrt()*self.transmission(self.first_phase[i])
        field = self.oeo(self.propagator(first))
        mask = F.pad(self.transmission(self.global_phase),(20,)*4,value=0)
        field = self.oeo(self.propagator(field*mask))
        energies = self.detect(field.abs().square(),self.class_centers,64)
        probs = (energies+1e-12)/(energies.sum(1,keepdim=True)+2e-12)
        return dict(probabilities=probs, route_power=q, capture=energies.sum(1),
                    router_capture=router_capture)


def loss(output, targets):
    return F.nll_loss(output['probabilities'].clamp_min(1e-12).log(), targets)
