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
        elif mode == 'fixed_dense':
            assert vocabulary<=32
            h=torch.ones(1,1)
            while h.shape[0]<32:
                h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
            self.register_buffer('codes',h)
        elif mode != 'fixed':
            raise ValueError(mode)

    def forward(self, token_ids):
        mask = token_ids.ne(0).float().unsqueeze(-1)
        if self.mode == 'fixed':
            return F.one_hot(token_ids, num_classes=64).float()*mask
        if self.mode=='fixed_dense':
            z=self.codes[token_ids]
        else:
            z, _ = self.gru(self.embedding(token_ids))
        z = z*mask
        return torch.cat((F.relu(z), F.relu(-z)), dim=-1)


def normalize_power(x, power):
    energy = x.square().sum((-2, -1), keepdim=True)
    return x*(power/energy.clamp_min(1e-20)).sqrt()


def encode(images, text, layout='legacy'):
    """RGB and text share 50/50 power, no fusion before optical propagation."""
    if layout in ['two_band','interleaved','left_right']:
        if images.ndim==2:
            sensor=images.reshape(-1,1,16,8)
        else:
            # Only monochrome spectrograms; never silently discard RGB information.
            assert images.ndim==4 and images.shape[-1]==3
            assert torch.equal(images[...,0],images[...,1]) and torch.equal(images[...,0],images[...,2])
            sensor=images[...,0].float()[:,None]/255
        target=(224,112) if layout=='left_right' else (112,224)
        sensor=F.interpolate(sensor,target,mode='bilinear',align_corners=False)[:,0]
        words=F.interpolate(text[:,None],target,mode='nearest')[:,0]
        sensor=normalize_power(sensor,.5);words=normalize_power(words,.5)
        if layout=='interleaved':
            return torch.stack((sensor,words),dim=-2).reshape(-1,224,224)
        if layout=='left_right':
            return torch.cat((sensor,words),-1)
        return torch.cat((sensor,words),-2)
    assert layout=='legacy'
    if images.ndim==2:
        # Same frozen GAP128 visual feature vector in each of the three slots.
        # Repetition does not supply additional information to either architecture.
        rgb=images.reshape(-1,1,16,8).expand(-1,3,-1,-1)
    else:
        rgb = images.float().permute(0, 3, 1, 2)/255
    if images.ndim==2:
        rgb=F.interpolate(rgb,(112,112),mode='nearest')
    else:
        rgb = F.interpolate(rgb, (112, 112), mode='bilinear', align_corners=False)
    rgb = rgb*(0.5/rgb.square().sum((1,2,3), keepdim=True).clamp_min(1e-20)).sqrt()
    # 32x64 ->112x112 is nearest-neighbour enlargement, not feature averaging.
    text = F.interpolate(text[:,None], (112,112), mode='nearest')[:,0]
    text = normalize_power(text, 0.5)
    return torch.cat((torch.cat((rgb[:,0], rgb[:,1]), -1),
                      torch.cat((rgb[:,2], text), -1)), -2)


def enlarge_tiles(amplitude, side, layout='legacy'):
    """No resampling across modality/channel boundaries; restore tile powers."""
    if layout in ['two_band','interleaved','left_right']:
        bands=[]
        if layout=='interleaved': source=(amplitude[:,0::2],amplitude[:,1::2])
        elif layout=='left_right': source=amplitude.chunk(2,dim=-1)
        else: source=amplitude.chunk(2,dim=-2)
        for band in source:
            power=band.square().sum((-2,-1),keepdim=True)
            target=(side,side//2) if layout=='left_right' else (side//2,side)
            bands.append(normalize_power(F.interpolate(band[:,None],target,mode='nearest')[:,0],power))
        if layout=='interleaved':return torch.stack(bands,dim=-2).reshape(-1,side,side)
        return torch.cat(bands,-1 if layout=='left_right' else -2)
    assert layout=='legacy'
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
    def __init__(self, architecture, seed=17, phase_dropout=0., phase_dropout_block=8, input_layout='legacy', oeo_activation='relu'):
        cfg_path = Path(__file__).resolve().parents[2]/'demo_check/pure_optical/config.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf8'))
        cfg['seed'] = seed
        cfg.update(protocol='clevr_attribute_text_encoding_pilot_v1',no_intermediate_oeo=False,
                   detector_size=64,input_encoding='RGB tiles plus word-position grid; RGB/text power 0.5 each',
                   oeo=f'active intensity / spatial mean -> nonaffine LayerNorm -> {oeo_activation} -> Softsign -> unit-power amplitude; zero phase',
                   loss='NLL of two normalized detector energies',augmentation='none',
                   selection='minimum validation NLL',no_trainable_electronic_adapter_or_head=False)
        if oeo_activation=='intensity_softsign':
            cfg['oeo']='active intensity / spatial mean -> Softsign -> unit-power amplitude; zero phase; no centered LayerNorm'
        super().__init__('dynamic_four' if architecture == 'moe' else 'full_d2nn', cfg)
        self.class_centers = [(259,179),(259,339)]
        self.phase_dropout = phase_dropout
        self.phase_dropout_block = phase_dropout_block
        self.input_layout = input_layout
        assert oeo_activation in ['relu','softplus','intensity_softsign']
        self.oeo_activation = oeo_activation

    def main_transmission(self, raw):
        value = self.transmission(raw)
        if self.training and self.phase_dropout:
            h,w = raw.shape[-2:]; b = self.phase_dropout_block
            # Per-layer, batch-shared blocks bypass phase, not amplitude.
            mask = torch.rand((1,1,(h+b-1)//b,(w+b-1)//b), device=raw.device)<self.phase_dropout
            mask = mask.repeat_interleave(b,-2).repeat_interleave(b,-1)[0,0,:h,:w]
            value = torch.where(mask, torch.ones_like(value), value)
        return value

    def oeo(self,field):
        # Identical active 478-square detector/SLM area for both models.
        intensity = field[:,20:498,20:498].abs().square()
        intensity = intensity/intensity.mean((-2,-1),keepdim=True).clamp_min(1e-20)
        if self.oeo_activation=='intensity_softsign':
            # Nonnegative monotone response: no mean subtraction, threshold, or pedestal.
            value=intensity
        else:
            z=F.layer_norm(intensity, (478,478), eps=1e-5)
            value=F.relu(z) if self.oeo_activation=='relu' else F.softplus(z)
        value = F.softsign(value)
        value = normalize_power(value, 1.0)
        return F.pad(value, (20,)*4).to(torch.complex64)

    def forward(self, amplitude):
        q, router_capture = self.route(amplitude)
        if self.architecture == 'full_d2nn':
            expanded = enlarge_tiles(amplitude, 478, self.input_layout)
            first = F.pad(expanded*self.main_transmission(self.first_phase), (20,)*4)
        else:
            first = amplitude.new_zeros((len(amplitude),518,518),dtype=torch.complex64)
            for i,(y,x) in enumerate(self.apertures):
                first[:,y:y+224,x:x+224] = amplitude*q[:,i,None,None].sqrt()*self.main_transmission(self.first_phase[i])
        field = self.oeo(self.propagator(first))
        mask = F.pad(self.main_transmission(self.global_phase),(20,)*4,value=0)
        field = self.oeo(self.propagator(field*mask))
        energies = self.detect(field.abs().square(),self.class_centers,64)
        probs = (energies+1e-12)/(energies.sum(1,keepdim=True)+2e-12)
        return dict(probabilities=probs, route_power=q, capture=energies.sum(1),
                    router_capture=router_capture)


def loss(output, targets):
    return F.nll_loss(output['probabilities'].clamp_min(1e-12).log(), targets)
