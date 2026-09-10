"""532 nm / 10 cm optics, four 224-pixel experts inside a 478-pixel field.

Logical pitch is 17 um. The 518 canvas is numerical padding, not the SLM size.
CCD log/LayerNorm/ReLU are deliberately retained to reproduce the audited best.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

CANVAS, ACTIVE, EXPERT, BORDER, GAP = 518, 478, 224, 20, 30
APERTURES = [(y, x) for y in (20, 274) for x in (20, 274)]
DEFAULT_NOISE = dict(router_bypass=.05,phase_bypass=.08,router_logit_std=.1,
                     dc_min=.2,dc_max=.3,gain_min=.4,gain_max=2.5,
                     ccd_mean=.03,ccd_std=.03,ccd_low=-.03,ccd_high=.12)


def noise_settings(config):
    result=DEFAULT_NOISE.copy()
    if config is not None:
        if set(config)-set(result):raise ValueError('Unknown optical noise settings')
        result.update(config)
    if not 0<=result['dc_min']<=result['dc_max']<1:raise ValueError('Invalid DC intensity fractions')
    if not 0<result['gain_min']<=result['gain_max']:raise ValueError('Invalid camera gain range')
    if result['ccd_std']<=0 or result['ccd_low']>=result['ccd_high']:raise ValueError('Invalid truncated CCD noise')
    return result


class Propagation(nn.Module):
    def __init__(self):
        super().__init__()
        frequency = torch.fft.fftfreq(CANVAS, d=17e-6, dtype=torch.float64)
        fy, fx = torch.meshgrid(frequency, frequency, indexing='ij')
        argument = (2 * math.pi) ** 2 * ((1 / 532e-9) ** 2 - fx.square() - fy.square())
        transfer = torch.exp(1j * (0.1 * argument.clamp_min(0).sqrt())).to(torch.complex64)
        self.register_buffer('transfer', torch.where(argument >= 0, transfer, 0), persistent=False)

    def forward(self, field):
        return torch.fft.ifft2(torch.fft.fft2(field.to(torch.complex64)) * self.transfer).to(torch.complex64)


def phase_modulation(raw):
    return torch.exp(1j * (2 * math.pi * torch.sigmoid(raw))).to(torch.complex64)


def block_bypass(modulation, probability, *, batch=1):
    size = modulation.shape[-1]
    keep = torch.rand(batch, math.ceil(size / 8), math.ceil(size / 8), device=modulation.device) >= probability
    keep = keep.repeat_interleave(8, -2).repeat_interleave(8, -1)[..., :size, :size].to(torch.complex64)
    return keep * modulation + (1 - keep)


def truncated_noise(reference, mean=.03, std=.03, low=-.03, high=.12):
    noise = torch.empty_like(reference).normal_(mean, std)
    invalid = (noise < low) | (noise > high)
    while bool(invalid.any()):
        noise[invalid] = torch.empty(int(invalid.sum()), device=noise.device).normal_(mean, std)
        invalid = (noise < low) | (noise > high)
    return noise


class Router(nn.Module):
    def __init__(self, noise_config=None):
        super().__init__()
        self.raw_router_phase = nn.Parameter(torch.zeros(EXPERT, EXPERT))
        self.propagator = Propagation()
        masks = torch.zeros(4, ACTIVE, ACTIVE)
        for i, (y, x) in enumerate([(y, x) for y in (164, 255) for x in (164, 255)]):
            masks[i, y:y+59, x:x+59] = 1
        self.register_buffer('detectors', masks, persistent=False)
        self.last = {}
        self.measured_ccd = None
        self.noise_config = noise_settings(noise_config)

    def forward(self, amplitude):
        field = F.pad(amplitude.float(), (147, 147, 147, 147))
        modulation = phase_modulation(self.raw_router_phase).unsqueeze(0).expand(len(field), -1, -1)
        if self.training:
            # Router bypass is sampled separately for every sample.
            modulation = block_bypass(modulation, self.noise_config['router_bypass'], batch=len(field))
        plane = torch.ones_like(field, dtype=torch.complex64)
        plane[:, 147:371, 147:371] = modulation
        intensity = (self.propagator(field.to(torch.complex64) * plane).abs().square().float()
                     [:, BORDER:-BORDER, BORDER:-BORDER]) if self.measured_ccd is None else self.measured_ccd.to(field.device).float()
        if tuple(intensity.shape) != (len(field), ACTIVE, ACTIVE):
            raise ValueError('Router measured CCD must be canonical [B,478,478]')
        energy = torch.einsum('bhw,ehw->be', intensity, self.detectors)
        centered = energy - energy.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(1e-8).sqrt()
        if self.training:
            logits = logits + torch.randn_like(logits)*self.noise_config['router_logit_std']
        probabilities = torch.softmax(logits / 2.0, dim=-1)
        indices = torch.topk(probabilities, 2, dim=-1).indices
        selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(1, indices, True)
        sparse = probabilities * selected
        hard = sparse / sparse.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        dense = probabilities / probabilities.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        weights = hard.detach() + dense - dense.detach()
        capture = (energy.sum(-1) / intensity.sum((-2, -1)).clamp_min(1e-8)).clamp(0, 1)
        importance = probabilities.mean(0)
        load = selected.float().mean(0) / 2
        surrogate_load = load + importance - importance.detach()
        self.last = dict(weights=weights, probabilities=probabilities, selected_mask=selected,
                         selected_indices=indices, energy=energy, intensity=intensity.detach(),
                         balance=4*(importance*load).sum()+.1*(1-capture).mean(),
                         importance=4*importance.square().sum()-1,
                         hard_balance=4*surrogate_load.square().sum()-1)
        return weights


class OpticalPath(nn.Module):
    def __init__(self, input_rms=0.25, noise_config=None):
        super().__init__()
        self.input_adapter = nn.Linear(192, 224)
        self.input_norm = nn.LayerNorm(224)
        self.noise_config = noise_settings(noise_config)
        self.eval_ccd_noise = False
        self.router = Router(noise_config)
        self.experts = nn.ParameterList([nn.Parameter(torch.zeros(224, 224)) for _ in range(4)])
        self.global_phase = nn.Parameter(torch.zeros(478, 478))
        self.expert_output = nn.Linear(224, 192)
        self.global_output = nn.Linear(224, 192)
        self.propagator = Propagation()
        self.input_rms = float(input_rms)
        self.measured = {}  # Stage -> canonical raw [B,478,478], no display enhancement.
        self.last_ccd = {}
        self.operating_losses = []
        indices = []
        for y, x in APERTURES:
            indices.append((torch.arange(y, y+224)[:, None]*518+torch.arange(x, x+224)[None]).reshape(-1))
        self.register_buffer('indices', torch.stack(indices).reshape(-1), persistent=False)

    def encode(self, latent):
        b, length, _ = latent.shape
        if length > 224:
            raise ValueError('Token length exceeds optical rows; truncation is forbidden')
        projected = F.softplus(self.input_norm(self.input_adapter(latent.reshape(-1, 192).float())))
        field = F.pad(projected.reshape(b, length, 224), (0, 0, 0, 224-length))
        rms = field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        return field * (self.input_rms / rms)

    def fanout(self, amplitude, weights):
        values = (amplitude.float()[:, None] * weights.float().clamp_min(0)[:, :, None, None]).reshape(len(amplitude), -1)
        canvas = amplitude.new_zeros(len(amplitude), 518*518).float().scatter(
            1, self.indices[None].expand(len(amplitude), -1), values).reshape(-1, 518, 518)
        return torch.complex(canvas, torch.zeros_like(canvas))

    def propagate(self, field, final):
        name = 'global' if final else 'expert'
        plane = torch.ones_like(field, dtype=torch.complex64)
        support = torch.zeros_like(field.real, dtype=torch.bool)
        if final:
            modulation = phase_modulation(self.global_phase)
            if self.training:
                modulation = block_bypass(modulation, self.noise_config['phase_bypass'])
            plane[:, 20:498, 20:498] = modulation
            support[:, 20:498, 20:498] = True
        else:
            for raw, (y, x) in zip(self.experts, APERTURES):
                modulation = phase_modulation(raw)
                if self.training:
                    modulation = block_bypass(modulation, self.noise_config['phase_bypass'])
                plane[:, y:y+224, x:x+224] = modulation
                support[:, y:y+224, x:x+224] = True
        if name in self.measured:
            intensity = self.measured[name].to(field.device).float()
        else:
            if self.training:
                # Fractions are intensity fractions; mix fields with square roots.
                shape = (len(field), 1, 1)
                amp_eta = field.real.new_empty(shape).uniform_(self.noise_config['dc_min'], self.noise_config['dc_max'])
                phase_eta = field.real.new_empty(shape).uniform_(self.noise_config['dc_min'], self.noise_config['dc_max'])
                incident = torch.zeros_like(field)
                incident[:, 20:498, 20:498] = self.input_rms * torch.exp(1j*field.real.new_empty(shape).uniform_(-math.pi, math.pi))
                field = (1-amp_eta).sqrt()*field + amp_eta.sqrt()*incident
                leak = torch.exp(1j*field.real.new_empty(shape).uniform_(-math.pi, math.pi))
                plane = torch.where(support, (1-phase_eta).sqrt()*plane+phase_eta.sqrt()*leak, plane)
            intensity = self.propagator(field*plane).abs().square().float()[:, 20:498, 20:498]
        if tuple(intensity.shape) != (len(field), 478, 478):
            raise ValueError('Measured CCD must be [B,478,478]')
        self.last_ccd[name] = intensity.detach()
        self.operating_losses.append(intensity.mean((-2, -1)).clamp_min(1e-8).log())
        if (self.training or self.eval_ccd_noise) and name not in self.measured:
            reference = intensity.mean((-2, -1), keepdim=True).detach()
            cfg=self.noise_config
            gain = intensity.new_empty(len(field), 1, 1).uniform_(cfg['gain_min'], cfg['gain_max'])
            intensity = (gain*intensity + truncated_noise(intensity,cfg['ccd_mean'],cfg['ccd_std'],cfg['ccd_low'],cfg['ccd_high'])*reference).clamp_min(0)
        return intensity

    def decode(self, intensity, length, dtype, final):
        value = intensity.float().clamp_min(0)
        relative = (value/value.mean((-2, -1), keepdim=True).clamp_min(1e-6)).clamp_max(12)
        pooled = F.adaptive_avg_pool2d(torch.log1p(relative)[:, None], (224, 224))[:, 0]
        readout = F.relu(F.layer_norm(pooled, (224,), eps=1e-5))
        packed = torch.cat([row[:length] for row in readout], dim=0)
        output = self.global_output if final else self.expert_output
        return output(packed).to(dtype).reshape(len(value), length, 192)

    def expert(self, latent):
        self.operating_losses = []
        amplitude = self.encode(latent)
        weights = self.router(amplitude)
        intensity = self.propagate(self.fanout(amplitude, weights), False)
        return self.decode(intensity, latent.shape[1], latent.dtype, False), weights

    def global_stage(self, latent, weights):
        field = self.fanout(self.encode(latent), weights)
        intensity = self.propagate(field, True)
        return self.decode(intensity, latent.shape[1], latent.dtype, True)
