"""Official ImageNet ResNet-18 with a strictly frozen visual encoder.

This module is a separate profile from the historical random-initialized
ResNet-18 comparison. It keeps torchvision's original state-dict names: an
owning model still saves ``encoder.conv1.weight``, not an extra wrapper prefix.
The removed ImageNet classifier is replaced by an identity operation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn


OFFICIAL_CHECKPOINT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_state_sha256(encoder: nn.Module) -> str:
    """Hash every parameter and persistent buffer, including BN counters.

    Tensor names, dtypes and shapes are included before their contiguous CPU
    bytes, so the identity does not depend on torch.save/pickle metadata.
    """
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        descriptor = [name, str(value.dtype), list(value.shape)]
        digest.update(json.dumps(descriptor, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_identity(encoder: nn.Module, weights) -> dict:
    """Bind loaded tensors to the actual checked official cache file."""
    filename = Path(urlparse(weights.url).path).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint.is_file():
        raise RuntimeError(f"Official pretrained checkpoint is absent: {checkpoint}")
    match = re.fullmatch(r"resnet18-([a-f0-9]{8,64})\.pth", filename)
    if match is None:
        raise RuntimeError(f"Unexpected official ResNet-18 filename: {filename}")
    actual_sha256 = file_sha256(checkpoint)
    if not actual_sha256.startswith(match.group(1)):
        raise RuntimeError("Official ResNet-18 checkpoint failed its URL SHA256 prefix")
    if actual_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError("Official ResNet-18 checkpoint failed its full SHA256 identity")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    modules = dict(encoder.named_modules())
    restored_legacy_bn_counters = []
    for name, actual in encoder.state_dict().items():
        expected = checkpoint_state.get(name)
        # The original official checkpoint predates persistent BN counters.
        # torchvision's strict loader restores those counters to scalar int64
        # zero. Accept exactly that compatibility value, never a missing model
        # parameter, BN mean/variance, nonzero counter or arbitrary buffer.
        if (expected is None and name.endswith(".num_batches_tracked")
                and isinstance(modules.get(name.rsplit(".", 1)[0]),
                               nn.modules.batchnorm._BatchNorm)
                and actual.dtype == torch.int64 and actual.shape == torch.Size([])
                and actual.item() == 0):
            restored_legacy_bn_counters.append(name)
            continue
        if (not isinstance(expected, torch.Tensor) or expected.dtype != actual.dtype
                or expected.shape != actual.shape or not torch.equal(expected, actual.cpu())):
            raise RuntimeError(f"Loaded encoder does not match pretrained checkpoint: {name}")
    return {
        "weights_enum": f"{type(weights).__name__}.{weights.name}",
        "weights_url": weights.url,
        "checkpoint_cache_path": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_url_hash_prefix_verified": True,
        "checkpoint_full_sha256_verified": True,
        "checkpoint_tensors_match_encoder": True,
        "torchvision_restored_zero_bn_counters": restored_legacy_bn_counters,
        "pretraining_dataset": "ImageNet-1K",
        "classification_head_removed": True,
    }


class FrozenResNet18(nn.Module):
    """512-dimensional frozen encoder; ``features`` exposes four ResNet stages.

    Both parameters and BatchNorm statistics stay fixed when the owning task
    model calls ``train()``. Encoder inference never builds an autograd graph.
    Construct on CPU first; the caller selects the admitted compute device.
    """

    output_dim = 512
    feature_channels = (64, 128, 256, 512)

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1
        official = resnet18(weights=weights)
        official.fc = nn.Identity()
        identity = _checkpoint_identity(official, weights)
        # Direct child registration retains all original torchvision key names.
        for name, module in official.named_children():
            self.add_module(name, module)
        self.requires_grad_(False)
        self.train(False)
        initial_sha256 = encoder_state_sha256(self)
        self.pretrained_provenance = {
            **identity,
            "encoder_state_sha256": initial_sha256,
            "encoder_parameters": sum(value.numel() for value in self.parameters()),
            "encoder_trainable_parameters": 0,
            "parameters_frozen": True,
            "batch_norm_running_statistics_frozen": True,
            "state_dict_key_prefix_added": False,
        }

    def train(self, mode: bool = True):
        # Parent model.train() recursively calls this method. Refuse to enable
        # encoder/BN training, including during the decoder's optimization steps.
        super().train(False)
        return self

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB [B,3,H,W]")
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        return a, b, c, self.layer4(c)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(images)[-1]), 1)


def assert_encoder_unchanged(encoder: nn.Module, expected_sha256: str) -> dict:
    """Fail closed if parameters, buffers, mode or gradient policy drifted."""
    if any(value.requires_grad for value in encoder.parameters()):
        raise RuntimeError("Frozen encoder has trainable parameters")
    if any(value.grad is not None for value in encoder.parameters()):
        raise RuntimeError("Frozen encoder accumulated parameter gradients")
    if any(module.training for module in encoder.modules()):
        raise RuntimeError("Frozen encoder contains a module in training mode")
    actual_sha256 = encoder_state_sha256(encoder)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Frozen encoder parameters or buffers changed")
    return {
        "expected_encoder_state_sha256": expected_sha256,
        "actual_encoder_state_sha256": actual_sha256,
        "encoder_unchanged": True,
        "encoder_trainable_parameters": 0,
        "encoder_parameter_gradients_present": False,
        "encoder_all_modules_eval": True,
    }
