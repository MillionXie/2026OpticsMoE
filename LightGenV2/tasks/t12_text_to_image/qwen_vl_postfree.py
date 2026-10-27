"""No TF after either optical macro; parameter-light instruction-aware readout."""
import math

import torch
from torch import nn

from .qwen_vl_hybrid import IdentityBlock, apply_hybrid
from .modeling import ScaleMatchedFusion


class QuietFusion(ScaleMatchedFusion):
    """Same fusion arithmetic, without per-forward GPU-to-CPU scalar logging."""
    def forward(self, electronic, optical):
        if electronic.shape != optical.shape:
            raise ValueError('Parallel branches must have identical output shapes')
        electronic32, optical32 = electronic.float(), optical.float()
        re, ro = self._rms(electronic32).detach(), self._rms(optical32).detach()
        mixture = (1-self.alpha)*electronic32/re+self.alpha*optical32/ro
        fused = re*mixture/self._rms(mixture).detach()
        return fused.to(electronic.dtype)


def apply_postfree(qwen, vision_pairs=1, language_pairs=1, residual_width=256):
    macros = apply_hybrid(qwen, vision_pairs, language_pairs, residual_width)
    for macro in macros.values():
        for i, fusion in enumerate(macro.fusions):
            quiet = QuietFusion(.65, .60, .90, 1e-6).to(fusion.raw_alpha.device)
            quiet.load_state_dict(fusion.state_dict(), strict=True)
            macro.fusions[i] = quiet
    for blocks, start in ((qwen.model.visual.blocks, 18),
                          (qwen.model.language_model.layers, 24)):
        for index in range(start, len(blocks)):
            blocks[index] = IdentityBlock()
    return macros


def instruction_bounds(ids, vision_end_id, message_end_id):
    values = ids.tolist()
    ends = [i for i, token in enumerate(values) if token == vision_end_id]
    if len(ends) != 1:
        raise ValueError('Require exactly one verified native image')
    start = ends[0]+1
    stop = next((i for i in range(start, len(values)) if values[i] == message_end_id), None)
    if stop is None or stop <= start:
        raise ValueError('Missing user instruction span')
    return start, len(values)-stop


def focus_features(hidden, valid, positions, text_start, suffix_tokens, gain):
    lengths = valid.sum(1)
    steps = torch.arange(hidden.shape[1], device=hidden.device)[None]
    mask = valid & (steps >= text_start) & (steps < lengths[:, None]-suffix_tokens)
    if not mask.any(1).all() or mask[:, positions].any():
        raise ValueError('Instruction mask empty or overlaps visual tokens')
    visual = hidden[:, positions].float()
    context = (hidden.float()*mask[:, :, None]).sum(1)/mask.sum(1, keepdim=True)
    vrms = visual.square().mean((1, 2), keepdim=True).sqrt().detach().clamp_min(1e-6)
    crms = context.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
    # One scalar, no additional TF or wide electronic network. Qwen text states
    # already attend to the reference image; all image inputs still enter Qwen.
    conditioned_visual = visual+gain*(context/crms)[:, None]*vrms
    return conditioned_visual, mask


class FocusedHead(nn.Module):
    def __init__(self, base, positions, text_start, suffix_tokens, initial_gain=.1):
        super().__init__()
        self.base = base
        self.positions = list(positions)
        self.text_start, self.suffix_tokens = text_start, suffix_tokens
        self.raw_text_gain = nn.Parameter(torch.tensor(math.log(initial_gain/(1-initial_gain))))

    @property
    def vae(self):
        return self.base.vae

    @property
    def text_gain(self):
        return self.raw_text_gain.sigmoid()

    def forward(self, hidden, valid, noise):
        visual, text_mask = focus_features(hidden, valid, self.positions,
            self.text_start, self.suffix_tokens, self.text_gain)
        spatial = self.base.spatial(visual)
        condition = self.base.adapter(hidden, text_mask)
        scheduler, unet, vae = self.base.scheduler, self.base.unet, self.base.vae
        scheduler.set_timesteps(1, device=noise.device)
        timestep = scheduler.timesteps[0]
        latent = noise.to(unet.dtype)*scheduler.init_noise_sigma
        noisy = scheduler.scale_model_input(latent, timestep)
        inputs = torch.cat([noisy, spatial.to(noisy.dtype)], 1)
        with torch.autocast(noise.device.type, dtype=torch.bfloat16, enabled=noise.is_cuda):
            prediction = unet(inputs, timestep, encoder_hidden_states=condition.to(unet.dtype)).sample
        generated = scheduler.step(prediction, timestep, latent).prev_sample
        image = vae.decode(generated.to(vae.dtype)/vae.config.scaling_factor).sample
        return generated.float(), image.float()


def short_prompt(row, description):
    # Only concise grammar is changed; no blind truncation of distinguishing
    # target attributes, and no product identifier or new object is introduced.
    description = description.replace(', entirely visible', '').replace('entirely visible', '').strip().rstrip('.')
    if row['mode'] == 'background':
        room, tone, brightness, direction = row['target_scene']
        return f"Keep the {row['category']}. Use a {brightness}, {tone} {room.replace('_', ' ')} with {direction} window light."
    instruction = f"Use {description}."
    if row['mode'] == 'object':
        return instruction+' Keep the background.'
    room, tone, brightness, direction = row['target_scene']
    return instruction+f" Use a {brightness}, {tone} {room.replace('_', ' ')} with {direction} window light."
