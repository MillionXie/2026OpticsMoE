"""Frozen Qwen patch/position/merger and compact token embeddings, no TF classes.

Fixed 224x224 image and audited fixed prompt only. Changing the prompt requires
re-exporting its embedding rows, rather than silently mapping unknown tokens.
"""
import torch
from torch import nn


class Frontend(nn.Module):
    def __init__(self, token_count):
        super().__init__()
        self.patch = nn.Conv3d(3, 1024, (2, 16, 16), stride=(2, 16, 16), bias=True)
        self.position = nn.Embedding(2304, 1024)
        self.merger_norm = nn.LayerNorm(1024, eps=1e-6)
        self.merger_fc1 = nn.Linear(4096, 4096)
        self.merger_fc2 = nn.Linear(4096, 2048)
        self.gelu = nn.GELU()
        self.tokens = nn.Embedding(token_count, 2048)
        self.register_buffer('token_ids', torch.zeros(token_count, dtype=torch.long))
        h = torch.linspace(0, 47, 14)
        floor = h.int()
        ceil = (floor + 1).clip(max=47)
        delta = h - floor
        ids = torch.stack([(a[:, None]*48+b[None]).flatten() for a,b in
                           [(floor,floor),(floor,ceil),(ceil,floor),(ceil,ceil)]]).long()
        weights = torch.stack([(a[:, None]*b[None]).flatten() for a,b in
                               [(1-delta,1-delta),(1-delta,delta),(delta,1-delta),(delta,delta)]])
        self.register_buffer('position_ids', ids, persistent=False)
        self.register_buffer('position_weights', weights, persistent=False)
        self.requires_grad_(False)

    def patches(self, pixels, batch):
        x = self.patch(pixels.view(-1,3,2,16,16).to(self.patch.weight.dtype)).view(-1,1024)
        ids = self.position_ids.repeat(1, batch)
        weights = self.position_weights.to(self.position.weight.dtype).repeat(1, batch)
        p = self.position(ids)*weights[:, :, None]
        p = p[0]+p[1]+p[2]+p[3]
        # Original Qwen stores 2x2 neighboring patches consecutively.
        p = p.reshape(batch,7,2,7,2,1024).permute(0,1,3,2,4,5).flatten(0,4)
        return (x+p).reshape(batch,196,1024)

    def merge(self, vision):
        x = self.merger_norm(vision.reshape(-1,1024)).view(-1,4096)
        return self.merger_fc2(self.gelu(self.merger_fc1(x)))

    def embed(self, ids):
        index = torch.searchsorted(self.token_ids, ids)
        index = index.clamp_max(len(self.token_ids)-1)
        if not torch.equal(self.token_ids[index], ids):
            raise ValueError('Prompt contains unexported tokens; re-export the frontend')
        return self.tokens(index)

