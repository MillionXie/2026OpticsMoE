"""Full multimodal Qwen conditioning -> native one-step SD-Turbo -> VAE RGB.

No reference VAE, GT access, template lookup, mask or RGB paste in forward.
"""
from __future__ import annotations

import torch
from torch import nn


class QueryResampler(nn.Module):
    def __init__(self, input_dim=2048, hidden=256, output_dim=1024, queries=77, layers=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, hidden)
        self.queries = nn.Parameter(torch.randn(queries, hidden) * .02)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(hidden, 8, hidden*4, dropout=0,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.output = nn.Linear(hidden, output_dim)

    def forward(self, hidden, valid):
        memory = self.project(self.input_norm(hidden.float()))
        x = self.queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        for block in self.blocks:
            x = block(x, memory, memory_key_padding_mask=~valid.bool())
        return self.output(x)


class NativeTurboHead(nn.Module):
    def __init__(self, unet, vae, scheduler, adapter):
        super().__init__()
        self.unet, self.vae, self.scheduler, self.adapter = unet, vae, scheduler, adapter

    def forward(self, hidden, valid, noise):
        conditions = self.adapter(hidden, valid)
        return self.generate_from_conditions(conditions, noise)

    def generate_from_conditions(self, conditions, noise):
        """Native generator; exposed for a TRAIN-only native-condition probe."""
        # Reset scheduler state for EVERY forward. Its native epsilon/v/sample
        # contract and scaling remain intact; no hand-written residual edit.
        self.scheduler.set_timesteps(1, device=noise.device)
        t = self.scheduler.timesteps[0]
        latents = noise.to(self.unet.dtype) * self.scheduler.init_noise_sigma
        model_input = self.scheduler.scale_model_input(latents, t)
        prediction = self.unet(model_input, t, encoder_hidden_states=conditions.to(self.unet.dtype)).sample
        generated = self.scheduler.step(prediction, t, latents).prev_sample
        rgb = self.vae.decode(generated.to(self.vae.dtype) / self.vae.config.scaling_factor).sample
        return generated.float(), rgb.float()


def image_tensor(image):
    import numpy as np
    return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float() / 127.5 - 1


def pack_hidden(rows, device):
    length = max(r.shape[0] for r in rows)
    h = torch.zeros(len(rows), length, rows[0].shape[-1], device=device, dtype=rows[0].dtype)
    valid = torch.zeros(len(rows), length, device=device, dtype=torch.bool)
    for i, row in enumerate(rows):
        h[i, :len(row)] = row.to(device)
        valid[i, :len(row)] = True
    return h, valid


@torch.no_grad()
def encode_reference(model, processor, reference, instruction, device):
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': reference},
                 {'type': 'text', 'text': instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = processor(text=[text], images=[reference], return_tensors='pt').to(device)
    if 'pixel_values' not in inputs or 'image_grid_thw' not in inputs:
        raise ValueError('Qwen visual path must be present')
    output = model.model(**inputs, use_cache=False, return_dict=True)
    valid = inputs['attention_mask'][0].bool()
    return output.last_hidden_state[0, valid].detach().bfloat16().cpu()
