"""Continuous static expert generation; no sample or batch inputs."""
from __future__ import annotations

import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen W plus alpha/r * BA, initialized to the original linear map."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        if x.dtype == self.lora_a.dtype:
            return self.base(x) + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale
        # Optional FP32 trainable adapters on a frozen BF16 backbone.
        base = self.base(x)
        with torch.autocast(x.device.type, enabled=False):
            delta = ((x.float() @ self.lora_a.T) @ self.lora_b.T) * self.scale
        return base + delta.to(base.dtype)


def install_lora(model: nn.Module, rank: int):
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in {'q_proj', 'v_proj'}:
            parent_name, child_name = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), child_name, LoRALinear(module, rank, 2 * rank))
            names.append(name)
    if not names:
        raise RuntimeError('No Qwen q_proj/v_proj modules found')
    return names


class ExpertLinear(nn.Module):
    """Separate output projections for the eight fixed expert identities."""
    def __init__(self, width, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(8, outputs, width))
        self.bias = nn.Parameter(torch.zeros(8, outputs))
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.einsum('epi,eoi->epo', x, self.weight) + self.bias[:, None]


class PatchDecoder(nn.Module):
    def __init__(self, context_dim, size=224, patch=16, width=128, expert_specific_heads=False):
        super().__init__()
        if size % patch:
            raise ValueError('size must be divisible by patch')
        self.size, self.patch, self.grid = size, patch, size // patch
        self.condition = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, width))
        self.positions = nn.Parameter(torch.randn(self.grid**2, width) * 0.02)
        head = ExpertLinear(width, patch**2) if expert_specific_heads else nn.Linear(width, patch**2)
        self.decode = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), head)
        nn.init.normal_(self.decode[-1].weight, std=0.002)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(self, context):
        x = self.condition(context.float())[:, None] + self.positions[None]
        patches = self.decode(x).reshape(8, self.grid, self.grid, self.patch, self.patch)
        return patches.permute(0, 1, 3, 2, 4).reshape(2, 4, self.size, self.size)


class StaticGenerator(nn.Module):
    def __init__(self, method='small_hyper', source=None, device='cpu', rank=8, size=224, patch=16, seed=42, task_description=None, expert_specific_heads=False):
        super().__init__()
        self.method = method
        self.lora_modules = []
        self.source = str(source)
        if method == 'small_hyper':
            width = 128
            self.tokens = nn.Parameter(torch.randn(8, 4, width) * 0.02)
            block = nn.TransformerEncoderLayer(width, 4, 256, dropout=0, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        elif method in {'qwen_frozen', 'qwen_lora'}:
            from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
            full = Qwen3VLForConditionalGeneration.from_pretrained(
                source, local_files_only=True, torch_dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32,
                attn_implementation='eager')
            self.encoder = full.model.language_model
            self.encoder.requires_grad_(False)
            del full
            self.encoder.to(device)
            if method == 'qwen_lora':
                self.lora_modules = install_lora(self.encoder, rank)
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, padding_side='right')
            task_description = task_description or 'Caltech101 ten-category image retrieval'
            prompts = [f'Design a fixed optical expert bank for {task_description}. Modality: {m}. Expert: {e}. Return a continuous design representation.'
                       for m in ('vision', 'language') for e in range(4)]
            encoded = tokenizer(prompts, padding=True, return_tensors='pt')
            self.register_buffer('input_ids', encoded['input_ids'].to(device))
            self.register_buffer('attention_mask', encoded['attention_mask'].to(device))
            width = self.encoder.config.hidden_size
        else:
            raise ValueError(method)
        self.decoder = PatchDecoder(width, size, patch, expert_specific_heads=expert_specific_heads).to(device)
        self.register_buffer('initial_reference', torch.zeros(2, 4, size, size, device=device))
        rng = torch.Generator().manual_seed(seed)
        self.register_buffer('anchor', (torch.randn(2, 4, size, size, generator=rng) * 0.02).to(device))
        self.to(device)
        self.eval()
        with torch.no_grad():
            self.initial_reference.copy_(self.decode_raw())

    def context(self):
        if self.method == 'small_hyper':
            return self.encoder(self.tokens)[:, -1]
        self.encoder.eval()  # disables dropout, does NOT disable LoRA autograd
        result = self.encoder(input_ids=self.input_ids, attention_mask=self.attention_mask, use_cache=False, return_dict=True)
        indexes = self.attention_mask.sum(-1) - 1
        return result.last_hidden_state[torch.arange(8, device=indexes.device), indexes]

    def decode_raw(self):
        return self.decoder(self.context())

    def forward(self):
        # Common fixed random anchor ensures exact same optical initialization
        # for all methods. Neither anchor nor initial_reference is trainable.
        return self.anchor + (self.decode_raw() - self.initial_reference)

    def compact_state(self):
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu().clone() for n, t in self.state_dict().items()
                if n in trainable or not n.startswith('encoder.')}

    def load_compact_state(self, state):
        result = self.load_state_dict(state, strict=False)
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        if result.unexpected_keys or any(n in trainable or not n.startswith('encoder.') for n in result.missing_keys):
            raise RuntimeError(f'Invalid compact state: {result}')
