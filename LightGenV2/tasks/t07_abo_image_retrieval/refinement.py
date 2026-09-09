"""Small, auditable T07 refinements; no attention or additional optical path."""
from __future__ import annotations

import math
import random
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from PIL import ImageEnhance, ImageOps, Image

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import GroceryRetrievalDataset
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import ElectronicRetrievalReadout


class WeakAugmentationDataset(GroceryRetrievalDataset):
    def _transform(self, image):
        # First retain the EXACT evaluation crop/aspect, then perturb slightly.
        image = super()._transform(image)
        scale = random.uniform(.94, 1.)
        side = round(self.image_size * scale)
        left, top = [random.randint(0, self.image_size-side) for _ in range(2)]
        image = image.crop((left, top, left+side, top+side)).resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(.95, 1.05))
        return ImageEnhance.Contrast(image).enhance(random.uniform(.95, 1.05))


class CrossProductBatchSampler:
    """All ten classes; each positive pair comes from DIFFERENT train products."""
    def __init__(self, samples, steps, seed):
        self.groups = defaultdict(lambda: defaultdict(list))
        for index, sample in enumerate(samples):
            self.groups[sample.sku_index][sample.sku_name].append(index)
        if len(self.groups) != 10 or any(len(g) < 2 for g in self.groups.values()):
            raise ValueError("Need ten classes and >=2 train products per class")
        self.steps, self.seed, self.epoch = steps, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.steps):
            batch = []
            for products in self.groups.values():
                for product in rng.sample(list(products), 2):
                    batch.append(rng.choice(products[product]))
            rng.shuffle(batch)
            yield batch


def lr_multiplier(epoch, epochs, warmup=5, floor=.1):
    if epoch <= warmup:
        return epoch / warmup
    progress = min(1., (epoch-warmup) / max(1, epochs-warmup))
    return floor + (1-floor) * .5 * (1+math.cos(math.pi*progress))


class TrainProductBank:
    """Detached train-only memory. No test samples or inference-time branch."""
    def __init__(self, samples, teacher_vectors, device):
        if any(s.split != "train" for s in samples):
            raise ValueError("Gallery memory must contain training samples only")
        products = sorted({s.product_id for s in samples})
        lookup = {p: i for i, p in enumerate(products)}
        self.product_ids = torch.tensor([lookup[s.product_id] for s in samples], device=device)
        self.labels = torch.tensor([s.category_id for s in samples], device=device)
        self.product_labels = torch.tensor([
            next(s.category_id for s in samples if s.product_id == p) for p in products], device=device)
        if any(len({s.category_id for s in samples if s.product_id == p}) != 1 for p in products):
            raise ValueError("Product has conflicting training categories")
        self.counts = torch.bincount(self.product_ids, minlength=len(products)).float()[:, None]
        if len(teacher_vectors) != len(samples):
            raise ValueError("Training teacher memory length mismatch")
        self.teacher = F.normalize(teacher_vectors.detach().float().to(device), dim=-1)
        self.teacher_centers = self.centers(self.teacher)
        self.memory = None

    def centers(self, vectors):
        totals = vectors.new_zeros((len(self.product_labels), vectors.shape[-1]))
        totals.index_add_(0, self.product_ids, F.normalize(vectors.float(), dim=-1))
        return F.normalize(totals / self.counts, dim=-1)

    @torch.no_grad()
    def refresh(self, vectors):
        if len(vectors) != len(self.product_ids):
            raise ValueError("Training memory refresh length mismatch")
        self.memory = F.normalize(vectors.detach().float().to(self.teacher.device), dim=-1).clone()

    @torch.no_grad()
    def update(self, indices, vectors, momentum=.5):
        indices = torch.as_tensor(indices, device=self.teacher.device)
        if len(indices.unique()) != len(indices):
            raise ValueError("Memory update batch must not repeat an image")
        new = F.normalize(vectors.detach().float(), dim=-1)
        self.memory[indices] = F.normalize(momentum*self.memory[indices] + (1-momentum)*new, dim=-1)

    def losses(self, query, indices, temperature=.1, teacher_temperature=.1):
        if self.memory is None or temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("Initialize memory and use positive temperatures")
        indices = torch.as_tensor(indices, device=query.device)
        valid = torch.arange(len(self.product_labels), device=query.device)[None, :] != self.product_ids[indices, None]
        positives = self.labels[indices, None].eq(self.product_labels[None, :]) & valid
        if not bool(positives.any(1).all()):
            raise ValueError("Every query needs a different positive training product")
        # Loss is probability mass of any relevant OTHER product, not uniform
        # matching of all within-class vectors. Gradients flow through queries.
        logits = F.normalize(query.float(), dim=-1) @ self.centers(self.memory).T / temperature
        logits = logits.masked_fill(~valid, -1e4)
        ranking = (logits.logsumexp(1) - logits.masked_fill(~positives, -1e4).logsumexp(1)).mean()
        with torch.no_grad():
            teacher_logits = self.teacher[indices] @ self.teacher_centers.T / teacher_temperature
            distribution = teacher_logits.masked_fill(~valid, -1e4).softmax(1)
        relational = F.kl_div(logits.log_softmax(1), distribution, reduction="batchmean")
        return ranking, relational


def inflate_convolution(convolution, new_size, causal):
    """Zero-pad learned kernels: initial outputs stay identical (also causal L)."""
    old_size = convolution.kernel_size[0]
    if new_size < old_size or new_size % 2 != 1:
        raise ValueError("New kernel must be odd and not smaller")
    cls = type(convolution)
    enlarged = cls(convolution.in_channels, convolution.out_channels, new_size,
                   groups=convolution.groups, bias=convolution.bias is not None).to(
                       device=convolution.weight.device, dtype=convolution.weight.dtype)
    with torch.no_grad():
        enlarged.weight.zero_()
        if isinstance(convolution, nn.Conv2d):
            start = (new_size-old_size)//2
            enlarged.weight[..., start:start+old_size, start:start+old_size].copy_(convolution.weight)
        else:
            start = new_size-old_size if causal else (new_size-old_size)//2
            enlarged.weight[..., start:start+old_size].copy_(convolution.weight)
        if convolution.bias is not None:
            enlarged.bias.copy_(convolution.bias)
    return enlarged


class ResidualRetrievalReadout(ElectronicRetrievalReadout):
    def __init__(self, original, hidden=512):
        super().__init__(original.detector_dim, original.embedding_dim)
        self.norm, self.projection = original.norm, original.projection
        self.correction = nn.Sequential(nn.Linear(self.detector_dim, hidden), nn.GELU(),
                                        nn.Dropout(.1), nn.Linear(hidden, self.embedding_dim))
        self.correction.to(device=self.projection.weight.device, dtype=self.projection.weight.dtype)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward_unnormalized(self, features):
        normalized = self.norm(features.float())
        return self.projection(normalized) + self.correction(normalized)

    def specification(self):
        return {**super().specification(),
                "architecture": f"LN({self.detector_dim}) -> [Linear64 + MLP{self.correction[0].out_features}_GELU_64] -> L2Normalize",
                "residual_mlp_hidden": self.correction[0].out_features,
                "attention": False, "zero_initialized_correction": True}


def enhance_graph(replacement, readout, options):
    if not options.get("enhanced_electronics", False):
        return readout
    kernel = int(options.get("residual_kernel_size", 9))
    for name in ("vision", "language"):
        core = getattr(replacement, name+"_surrogate").core
        for block in core.blocks:
            if not block.token_mixer_enabled:
                raise RuntimeError("Expected existing convolutional electronic residual")
            block.token_depthwise = inflate_convolution(block.token_depthwise, kernel, name == "language")
            block.token_mixer_kernel_size = kernel
        core.token_mixer_kernel_size = kernel
    readout = ResidualRetrievalReadout(readout, int(options.get("readout_hidden", 512)))
    replacement.checkpoint_architecture += f"_t07_conv{kernel}_readout{readout.correction[0].out_features}"
    return readout
