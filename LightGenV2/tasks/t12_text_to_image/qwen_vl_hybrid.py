"""Native Qwen TF-span replacement with one physical router per modality.

Original tokenizer, patch embedding, mergers/DeepStack, retained Qwen layers and
generation head stay in their native positions. No RGB/VAE input bypass exists.
"""
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .audited_unified import optical_path
from .lab_shs8um.bounded_amplitude import encode as bounded_encode
from .modeling import ScaleMatchedFusion


def feature_loss(actual, target):
    """RMS-normalized MSE plus token cosine; preserve scale separately."""
    a, b = actual.float(), target.float()
    rms = b.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
    return ((a-b)/rms).square().mean()+(.5*(1-F.cosine_similarity(a, b, dim=-1))).mean()


class TokenCodec(nn.Module):
    def __init__(self, modality, width):
        super().__init__()
        self.modality = modality
        self.channels = 196 if modality == 'vision' else 224
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, self.channels)
        self.output = nn.Linear(self.channels, width)

    def pack(self, hidden):
        projected = self.input(self.norm(hidden.float()))
        b, n, c = projected.shape
        if self.modality == 'vision':
            if n != 256:
                raise ValueError('Native 256 visual tokens required; never truncate')
            return projected.reshape(b, 224, 224)
        if n > 224:
            raise ValueError('Language exceeds one optical field; no silent truncation')
        return projected

    def unpack(self, packed, tokens):
        if self.modality == 'vision':
            if tokens != 256 or packed.shape[1:] != (224, 224):
                raise ValueError('Visual field reordering contract differs')
            packed = packed.reshape(len(packed), 256, 196)
        elif packed.shape[1] != tokens:
            raise ValueError('Language token count changed')
        return self.output(packed)


class ElectronicResidual(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.normal_(self.up.weight, std=.002)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x.float()+self.up(F.silu(self.down(self.norm(x.float()))))


def prepare_clean_path(path, first):
    """Use audited full geometry, with an explicit clean starting curriculum."""
    path.eval()
    path.input_shift_pixels = path.phase_shift_pixels = path.ccd_shift_pixels = 0
    path.gain_min = path.gain_max = 1.
    path.offset_fraction = path.read_noise_fraction = 0.
    path.zero_order_enabled = False
    if first:
        router = path.core.router
        router.noise_std = router.phase_dropout_p = 0.
        router.input_shift_pixels = router.phase_shift_pixels = router.ccd_shift_pixels = 0
        router._t12_physical_simulate = router._simulate
        def bounded_router(fields):
            return router._t12_physical_simulate(bounded_encode(fields, {'kind': 'tanh', 'scale': .5}))
        router._simulate = bounded_router
    else:
        # Only the first pair OWNS a router; later pairs reuse its selection.
        path.core.router = nn.Identity()
    path._t12_physical_simulate = path._simulate_detector_roi
    def bounded_detector(field, modulation, shifts, *, phase_support=None):
        return path._t12_physical_simulate(bounded_encode(field, {'kind': 'tanh', 'scale': .5}),
                        modulation, shifts, phase_support=phase_support)
    path._simulate_detector_roi = bounded_detector


class OpticalMacro(nn.Module):
    def __init__(self, modality, width, pairs=1, residual_width=256, path_factory=optical_path):
        super().__init__()
        if modality not in ('vision', 'language') or pairs not in (1, 2):
            raise ValueError('Only approved modalities and one/two E-G pairs')
        self.modality, self.width, self.pairs = modality, width, pairs
        self.paths = nn.ModuleList([path_factory(224, 224) for _ in range(pairs)])
        for i, path in enumerate(self.paths):
            prepare_clean_path(path, i == 0)
        self.codecs = nn.ModuleList([TokenCodec(modality, width) for _ in range(2*pairs)])
        self.electronic = nn.ModuleList([ElectronicResidual(width, residual_width) for _ in range(2*pairs)])
        self.fusions = nn.ModuleList([ScaleMatchedFusion(.65, .60, .90, 1e-6) for _ in range(2*pairs)])
        self.routing = None
        self.last_stages = []
        self.optical_enabled = True
        self.router_calls = 0
        self.register_buffer('routing_mean', torch.full((4,), .25))

    def train(self, mode=True):
        super().train(mode)
        # Optical weights still receive gradients in eval; this disables ONLY
        # inherited dc20 random corruption until a separately recorded curriculum.
        for path in self.paths:
            path.eval()
        return self

    def stage(self, hidden, stage):
        if not 0 <= stage < 2*self.pairs:
            raise ValueError('Wrong stage')
        if stage == 0:
            self.routing = None
            self.last_stages = []
            self.router_calls = 0
        elif self.routing is None:
            raise RuntimeError('Expert stage must execute before later stages')
        x = hidden.float()
        electronic = self.electronic[stage](x)
        with torch.autocast(x.device.type, enabled=False):
            codec, path = self.codecs[stage], self.paths[stage//2]
            packed = codec.pack(x)
            padding = torch.zeros(packed.shape[:2], dtype=torch.bool, device=x.device)
            fields, lengths = path._encode_input_fields(packed, padding)
            if stage == 0:
                field, self.routing = path.core.begin(fields)
                self.router_calls += 1
            else:
                field = path.core.fanout(fields, self.routing)
            expert = stage % 2 == 0
            phase = path._expert_phase_modulation(field) if expert else path._global_phase_modulation(field)
            support = 'expert' if expert else 'global'
            intensity = path._simulate_detector_roi(field, phase, path._draw_stage_shifts(support), phase_support=support)
            # Normalization and detector readout are the audited electronic OEO
            # components; they are not hidden inside a fake optical matrix.
            delta = path._readout_delta(intensity, lengths, torch.float32, final=not expert)
            optical_delta = codec.unpack(delta.reshape(len(x), packed.shape[1], 224), x.shape[1])
            optical = x+optical_delta
            if not self.optical_enabled:
                optical = x  # Explicit ablation only, never the deploy default.
            fused = self.fusions[stage](electronic, optical)
        result = fused.to(hidden.dtype)
        self.last_stages.append(result)
        return result

    def forward(self, hidden):
        for i in range(2*self.pairs):
            hidden = self.stage(hidden, i)
        return hidden

    def balance_loss(self):
        if self.routing is None:
            return self.fusions[0].raw_alpha*0
        probabilities = self.routing.get('probabilities', self.routing.get('soft_probs'))
        if probabilities is None:
            probabilities = self.routing['weights']
        mean = probabilities.float().mean(0)
        # B1 training must balance ACROSS examples, not force every instruction
        # to select a uniform mixture (which would collapse hard top-2 ties).
        reference = self.routing_mean.detach().clone()
        if self.training:
            with torch.no_grad():
                self.routing_mean.mul_(.97).add_(mean.detach(), alpha=.03)
        return 4*(mean*reference).sum()-1

    def diagnostics(self):
        route = self.routing or {}
        result = {'alpha': [float(f.alpha.detach()) for f in self.fusions],
                  'router_calls_per_forward': self.router_calls,
                  'optical_cycles_per_forward': 1+2*self.pairs,
                  'physical_active_roi': 478, 'propagation_canvas': 518,
                  'expert_tile': 224, 'top_k': 2}
        for key in ('selected_mask', 'weights', 'probabilities', 'soft_probs'):
            if key in route:
                result[key] = route[key].detach().float().cpu().tolist()
        return result


class IdentityBlock(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class MacroStageBlock(nn.Module):
    def __init__(self, macro, stage, vision):
        super().__init__()
        self._macro_ref = weakref.ref(macro)
        self.stage_index, self.vision = stage, vision

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states.unsqueeze(0) if self.vision else hidden_states
        # The installed B1 text path has no padded tokens. Do not silently
        # interpret a multi-image packed vision batch as one image.
        if self.vision and x.shape[1] != 256:
            raise ValueError('One native 16x16 visual field per forward required')
        out = self._macro_ref().stage(x, self.stage_index)
        return out[0] if self.vision else out


def apply_hybrid(qwen, vision_pairs=1, language_pairs=1, residual_width=256):
    if hasattr(qwen.model, 'hybrid'):
        raise ValueError('Native Qwen spans already replaced')
    qwen.requires_grad_(False)
    device = next(qwen.parameters()).device
    macros = nn.ModuleDict({'vision': OpticalMacro('vision', 1024, vision_pairs, residual_width),
                           'language': OpticalMacro('language', 2048, language_pairs, residual_width)}).to(device)
    qwen.model.hybrid = macros
    for name, blocks, start, stop, vision in (
            ('vision', qwen.model.visual.blocks, 6, 18, True),
            ('language', qwen.model.language_model.layers, 4, 24, False)):
        macro = macros[name]
        # For vision K=1, E is at6/G12, native DeepStack captures at11/17.
        # K=2: E6/G9/E12/G15, preserving both native capture interfaces.
        starts = [start+(stop-start)*i//(2*macro.pairs) for i in range(2*macro.pairs)]
        for i in range(start, stop):
            blocks[i] = MacroStageBlock(macro, starts.index(i), vision) if i in starts else IdentityBlock()
    qwen.eval()
    return macros


def hybrid_config(vision_pairs=1, language_pairs=1):
    return {'schema': 'qwen_vl_native_hybrid_v1', 'vision_replaced': list(range(6, 18)),
            'language_replaced': list(range(4, 24)), 'vision_pairs': vision_pairs,
            'language_pairs': language_pairs, 'alpha_min': .60, 'alpha_max': .90,
            'bounded_amplitude': {'kind': 'tanh', 'scale': .5}, 'one_router_per_modality': True,
            'original_vision_tokens': 256, 'visual_packed_shape': [224, 224],
            'reference_vae_inference': False}
