"""CPU-reviewable task models, pending architecture approval.

The visual encoder is CIFAR ResNet-20, not a narrow ImageNet ResNet-18:
3x3 stem; [3,3,3] basic blocks; [16,32,64] channels; option-A shortcuts.
Task heads and the byte-GRU are explicit additions, counted in full.
No pretrained model, Qwen feature cache, download, or device selection occurs.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class OptionAShortcut(nn.Module):
    """Original parameter-free subsampling and zero-padded channels."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.padding = (out_channels - in_channels) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.padding, self.padding))


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels
                         else OptionAShortcut(in_channels, out_channels, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


class ResNet20Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU())
        self.stage1 = self._stage(16, 16, 1)
        self.stage2 = self._stage(16, 32, 2)
        self.stage3 = self._stage(32, 64, 2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _stage(in_channels: int, out_channels: int, stride: int):
        return nn.Sequential(BasicBlock(in_channels, out_channels, stride),
                             BasicBlock(out_channels, out_channels),
                             BasicBlock(out_channels, out_channels))

    def features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB [B,3,H,W]")
        a = self.stage1(self.stem(images))
        b = self.stage2(a)
        return a, b, self.stage3(b)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)[-1].mean(dim=(-2, -1))


def tokenize_text(texts: list[str], max_bytes: int = 512) -> torch.Tensor:
    """UTF-8 bytes, PAD=0/BOS=1/EOS=2; no class/SKU ID or target input.

    Preserve full instructions; overlength input fails rather than truncates.
    The caller moves the returned CPU tensor to the model's device.
    """
    if not texts:
        raise ValueError("Empty text batch")
    sequences = []
    for value in texts:
        encoded = value.encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError("Text exceeds declared byte limit; review the input contract")
        sequences.append([1, *(v + 3 for v in encoded), 2])
    batch = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        batch[i, :len(sequence)] = torch.tensor(sequence)
    return batch


class ByteTextEncoder(nn.Module):
    """Trainable 259x32 embedding plus one unidirectional GRU(32,64)."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(259, 32, padding_idx=0)
        self.gru = nn.GRU(32, 64, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("Expected nonempty [B,L] byte tokens")
        lengths = (tokens != 0).sum(1).cpu()
        if bool((lengths < 2).any()):
            raise ValueError("Every sequence needs BOS/EOS")
        expected = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths.to(tokens.device)[:, None]
        if not torch.equal(tokens != 0, expected):
            raise ValueError("Padding must be right-contiguous")
        packed = pack_padded_sequence(self.embedding(tokens), lengths,
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden[-1]


class QualityModel(nn.Module):
    def __init__(self, target: str, backbone: str = "resnet20"):
        super().__init__()
        if target not in {"spatial", "temporal"}:
            raise ValueError(target)
        self.target = target
        if backbone == "resnet20":
            self.encoder = ResNet20Encoder()
            self.feature_dim = 64
        elif backbone == "resnet18":
            from torchvision.models import resnet18
            self.encoder = resnet18(weights=None)
            self.encoder.fc = nn.Identity()
            self.feature_dim = 512
        else:
            raise ValueError(backbone)
        self.head = nn.Sequential(nn.Linear(self.feature_dim * (1 if target == "spatial" else 7), 64),
                                  nn.ReLU(), nn.Linear(64, 5))
        self.register_buffer("quality_levels", torch.arange(1., 6.))

    def set_train_score_range(self, train_mos: torch.Tensor) -> torch.Tensor:
        """Bind score scale/bins using TRAIN only, as in the original baseline.

        Default [1..5] levels are for the architecture prototype only. The data
        adapter must call this before training/evaluation on actual MOS values.
        Returns four equally-spaced bin boundaries for bucketize(right=False).
        """
        values = torch.as_tensor(train_mos).detach().float()
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("TRAIN MOS must be nonempty and finite")
        low, high = values.min().item(), values.max().item()
        if high <= low:
            raise ValueError("TRAIN MOS range must be positive")
        self.quality_levels.copy_(torch.linspace(low, high, 5, device=self.quality_levels.device))
        return torch.linspace(low, high, 6, device=self.quality_levels.device)[1:-1]

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[1:3] != (4, 3):
            raise ValueError("Exactly four RGB frames per video are required")
        batch, count, channels, height, width = frames.shape
        z = self.encoder(frames.reshape(batch * count, channels, height, width)).reshape(batch, count, self.feature_dim)
        features = (z.mean(1) if self.target == "spatial" else
                    torch.cat((z.flatten(1), (z[:, 1:] - z[:, :-1]).flatten(1)), 1))
        logits = self.head(features)
        return {"quality_logits": logits,
                "quality_score": logits.softmax(-1) @ self.quality_levels}


class ImageRetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet20Encoder()
        self.projection = nn.Linear(64, 64)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.encoder(images)), dim=-1)


class CrossModalRetrievalModel(ImageRetrievalModel):
    def __init__(self):
        super().__init__()
        self.text_encoder = ByteTextEncoder()
        self.text_projection = nn.Linear(64, 64)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projection(self.text_encoder(tokens)), dim=-1)


class DenseModel(nn.Module):
    def __init__(self, outputs: int, output_size: int):
        super().__init__()
        self.encoder = ResNet20Encoder()
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, outputs, 1))
        self.output_size = output_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.encoder.features(images)[-1])
        return F.interpolate(logits, (self.output_size, self.output_size),
                             mode="bilinear", align_corners=False)


class OpenMojiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet20Encoder()
        self.text_encoder = ByteTextEncoder()
        self.film = nn.Linear(64, 128)
        self.coordinates = nn.Conv2d(2, 64, 1)
        self.spatial = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.category = nn.Conv2d(64, 17, 1)
        self.edit = nn.Conv2d(64, 1, 1)
        self.task = nn.Linear(64, 4)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.adaptive_avg_pool2d(self.encoder.features(images)[-1], (6, 6))
        condition = self.text_encoder(tokens)
        gamma, beta = self.film(condition).chunk(2, -1)
        x = x * (1 + torch.tanh(gamma)[:, :, None, None]) + beta[:, :, None, None]
        axis = torch.linspace(-1, 1, 6, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        x = self.spatial(x + self.coordinates(torch.stack((xx, yy))[None]))
        return {"category_logits": self.category(x),
                "edit_logits": self.edit(x).squeeze(1), "task_logits": self.task(condition)}


class ConditionalEditor(nn.Module):
    """Paired reference-image + instruction editor; not text-only synthesis.

    Image-only skip features carry spatial detail. Targets and GT masks are
    never inputs. Two bilinear upsampling steps invert the encoder's /4 scale.
    """

    def __init__(self):
        super().__init__()
        self.encoder = ResNet20Encoder()
        self.text_encoder = ByteTextEncoder()
        self.film = nn.Linear(64, 128)
        self.decode2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.decode1 = nn.Sequential(nn.Conv2d(32 + 16, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.rgb = nn.Conv2d(16, 3, 1)

    def forward(self, reference: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        a, b, x = self.encoder.features(reference)
        gamma, beta = self.film(self.text_encoder(tokens)).chunk(2, -1)
        x = x * (1 + torch.tanh(gamma)[:, :, None, None]) + beta[:, :, None, None]
        x = F.interpolate(x, b.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode2(torch.cat((x, b), 1))
        x = F.interpolate(x, a.shape[-2:], mode="bilinear", align_corners=False)
        return self.rgb(self.decode1(torch.cat((x, a), 1))).sigmoid()


TASK_IDS = ("lgvq_temporal", "lgvq_spatial", "abo_image_to_text", "abo_image_to_image",
            "abo_text_to_image", "lsp", "salicon", "openmoji", "abo_conditional_edit")


def build_model(task: str) -> nn.Module:
    if task == "lgvq_temporal_resnet18":
        return QualityModel("temporal", backbone="resnet18")
    if task.startswith("lgvq_"):
        return QualityModel(task.removeprefix("lgvq_"))
    if task in {"abo_image_to_text", "abo_text_to_image"}:
        return CrossModalRetrievalModel()
    if task == "abo_image_to_image":
        return ImageRetrievalModel()
    if task == "lsp":
        return DenseModel(14, 56)
    if task == "salicon":
        return DenseModel(1, 224)
    if task == "openmoji":
        return OpenMojiModel()
    if task == "abo_conditional_edit":
        return ConditionalEditor()
    raise ValueError(f"Unknown task: {task}")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    visual = sum(p.numel() for p in model.encoder.parameters())
    text = sum(p.numel() for p in model.text_encoder.parameters()) if hasattr(model, "text_encoder") else 0
    return {"total": total, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "visual_encoder": visual, "text_encoder": text, "task_heads": total - visual - text}
