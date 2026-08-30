from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)
from experiments.qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone.model import (
    qwen_tokens_to_grid,
)


DownstreamTask = Literal["caltech101", "isic2016", "lsp"]
FeedbackMethod = Literal[
    "noft",
    "bp",
    "bp_current",
    "fa_pretrained",
    "fa_random",
]

_P11_SIGNATURE_KEY = "p11_separable_architecture_signature"
_P11_SIGNATURE = torch.tensor([11, 1, 2, 4], dtype=torch.int64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_physical_phases(
    phases: torch.Tensor,
    expected_shape: tuple[int, int, int, int],
    *,
    name: str,
) -> None:
    if tuple(phases.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(phases.shape)}")
    if not bool(torch.isfinite(phases).all()):
        raise ValueError(f"{name} contains non-finite values")
    tolerance = 1.0e-5
    minimum = float(phases.detach().amin().cpu())
    maximum = float(phases.detach().amax().cpu())
    if minimum < -tolerance or maximum > 2.0 * math.pi + tolerance:
        raise ValueError(
            f"{name} must contain physical phase in [0, 2pi], got [{minimum}, {maximum}]"
        )


def _checkpoint_backbone_state(payload: Any) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError("P11 checkpoint must contain a mapping")
    metadata: Mapping[str, Any] = payload
    if "backbone" in payload:
        candidate = payload["backbone"]
    elif "model" in payload:
        candidate = {
            name: value
            for name, value in payload["model"].items()
            if not str(name).startswith("readout.")
        }
    else:
        candidate = payload
        metadata = {}
    if not isinstance(candidate, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in candidate.items()
    ):
        raise TypeError("P11 backbone state must be a string-to-tensor mapping")
    return dict(candidate), metadata


def load_strict_p11_backbone(
    model: QwenStemSeparableOpticalImageNetBackbone,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Load the complete P11 reusable backbone and reject silent partial loads.

    The exported P11 state deliberately omits only the ImageNet classifier.
    In particular, the learned 1024->224 adapter and all electronic residual
    mixers are part of the backbone and must never be filtered out.
    """

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"P11 source checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state, metadata = _checkpoint_backbone_state(payload)
    signature = state.get(_P11_SIGNATURE_KEY)
    if signature is None or not torch.equal(signature.detach().cpu(), _P11_SIGNATURE):
        raise RuntimeError("Source checkpoint does not carry the required P11 architecture signature")

    expected_missing = {
        f"readout.{name}" for name in model.readout.state_dict().keys()
    }
    incompatible = model.load_state_dict(state, strict=False)
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing:
        raise RuntimeError(
            "P11 backbone load was incomplete: "
            f"expected only {sorted(expected_missing)}, got {sorted(actual_missing)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"P11 backbone has unexpected keys: {sorted(incompatible.unexpected_keys)}"
        )

    report = metadata.get("model_report") if isinstance(metadata, Mapping) else None
    if isinstance(report, Mapping):
        variant = report.get("optical_mixer_variant")
        if variant not in {None, "separable_token_channel_axis"}:
            raise RuntimeError(f"Checkpoint reports an incompatible optical variant: {variant}")
    expected_stem_hash = metadata.get("stem_checkpoint_sha256") if isinstance(metadata, Mapping) else None
    if expected_stem_hash is not None and str(expected_stem_hash) != model.stem.checkpoint_sha256:
        raise RuntimeError(
            "Frozen Qwen stem SHA-256 differs from the P11 source checkpoint: "
            f"expected {expected_stem_hash}, got {model.stem.checkpoint_sha256}"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "model_report": dict(report) if isinstance(report, Mapping) else None,
        "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
    }


class OpticalBankFusion(nn.Module):
    """Learn one convex combination of the three optical banks."""

    def __init__(self, banks: int = 3) -> None:
        super().__init__()
        self.banks = int(banks)
        self.bank_logits = nn.Parameter(torch.zeros(self.banks))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.bank_logits, dim=0)

    def forward(self, active_tokens: torch.Tensor) -> torch.Tensor:
        if active_tokens.ndim != 4 or active_tokens.shape[1] != self.banks:
            raise ValueError(
                f"Expected [B,{self.banks},T,C] optical tokens, got {tuple(active_tokens.shape)}"
            )
        weights = self.weights().to(device=active_tokens.device, dtype=active_tokens.dtype)
        return (active_tokens * weights.view(1, self.banks, 1, 1)).sum(dim=1)


class GlobalTransferHead(nn.Module):
    """Final-stage 448-D global descriptor, classifier and retrieval embedding."""

    def __init__(
        self,
        *,
        token_count: int,
        token_dim: int,
        optical_banks: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.fusion = OpticalBankFusion(optical_banks)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.descriptor_norm = nn.LayerNorm(2 * self.token_dim)
        self.hidden = nn.Sequential(
            nn.Linear(2 * self.token_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))
        self.descriptor_dim = 2 * self.token_dim
        self.embedding_dim = int(hidden_dim)

    def forward(self, final_field: torch.Tensor) -> dict[str, torch.Tensor]:
        active = _active_tokens(final_field, self.token_count, self.token_dim)
        tokens = self.token_norm(self.fusion(active))
        descriptor = self.descriptor_norm(
            torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)
        )
        hidden = self.hidden(descriptor)
        return {
            "logits": self.classifier(hidden),
            "embedding": F.normalize(hidden, dim=-1),
        }


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ProgressiveUpsampleBlock(nn.Module):
    """Bilinear upsampling followed by an attention-free separable convolution."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            int(input_channels),
            int(input_channels),
            kernel_size=3,
            padding=1,
            groups=int(input_channels),
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(
            _group_count(int(input_channels)), int(input_channels)
        )
        self.pointwise = nn.Conv2d(
            int(input_channels), int(output_channels), kernel_size=1, bias=False
        )
        self.output_norm = nn.GroupNorm(
            _group_count(int(output_channels)), int(output_channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        value = F.gelu(self.depthwise_norm(self.depthwise(value)))
        return F.gelu(self.output_norm(self.pointwise(value)))


class ProgressiveDenseHead(nn.Module):
    """Final-stage 14x14 token decoder for segmentation or keypoint heatmaps."""

    def __init__(
        self,
        *,
        token_count: int,
        token_dim: int,
        optical_banks: int,
        output_channels: int,
        output_size: int,
        decoder_channels: Sequence[int] = (192, 160, 128, 96, 64),
    ) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.grid_size = math.isqrt(self.token_count)
        if self.grid_size * self.grid_size != self.token_count:
            raise ValueError("Dense decoding requires a square token grid")
        self.output_size = int(output_size)
        ratio = self.output_size / self.grid_size
        upsample_count = int(round(math.log2(ratio))) if ratio >= 1.0 else -1
        if upsample_count < 0 or self.grid_size * (2**upsample_count) != self.output_size:
            raise ValueError(
                f"output_size must be {self.grid_size} times a non-negative power of two"
            )
        widths = tuple(int(value) for value in decoder_channels)
        if len(widths) < upsample_count + 1 or any(value <= 0 for value in widths):
            raise ValueError("decoder_channels does not cover all progressive upsampling stages")

        self.fusion = OpticalBankFusion(optical_banks)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.input_projection = nn.Sequential(
            nn.Conv2d(self.token_dim, widths[0], kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(widths[0]), widths[0]),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ProgressiveUpsampleBlock(widths[index], widths[index + 1])
            for index in range(upsample_count)
        )
        self.output_projection = nn.Conv2d(
            widths[upsample_count], int(output_channels), kernel_size=1
        )
        if self.parameter_count() >= 1_000_000:
            raise ValueError("The temporary dense head must remain below one million parameters")

    def fused_grid(self, final_field: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
        active = _active_tokens(final_field, self.token_count, self.token_dim)
        tokens = self.fusion(active)
        if normalize:
            tokens = self.token_norm(tokens)
        return qwen_tokens_to_grid(tokens, grid_size=self.grid_size)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, final_field: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.input_projection(self.fused_grid(final_field, normalize=True))
        for block in self.blocks:
            value = block(value)
        logits = self.output_projection(value)
        if tuple(logits.shape[-2:]) != (self.output_size, self.output_size):
            raise RuntimeError("Progressive decoder produced an unexpected spatial size")
        return {"logits": logits}


def _active_tokens(
    final_field: torch.Tensor,
    token_count: int,
    token_dim: int,
) -> torch.Tensor:
    if final_field.ndim != 4 or final_field.shape[-2] < int(token_count):
        raise ValueError(f"Expected a padded [B,banks,H,{token_dim}] P11 field")
    if final_field.shape[-1] != int(token_dim):
        raise ValueError(f"Expected token dimension {token_dim}, got {final_field.shape[-1]}")
    return final_field[:, :, : int(token_count), :]


class P11DownstreamModel(nn.Module):
    """Strict P11 source backbone plus one temporary downstream task head."""

    def __init__(
        self,
        *,
        stem_checkpoint: str | Path,
        source_checkpoint: str | Path,
        p11_config: Mapping[str, Any],
        task: DownstreamTask,
        num_outputs: int | None = None,
        global_hidden_dim: int = 256,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.10,
        dense_output_size: int | None = None,
        output_size: int | None = None,
        decoder_width: int = 64,
    ) -> None:
        super().__init__()
        if task not in {"caltech101", "isic2016", "lsp"}:
            raise ValueError(f"Unsupported downstream task: {task}")
        self.task: DownstreamTask = task
        self.backbone = QwenStemSeparableOpticalImageNetBackbone(
            stem_checkpoint, dict(p11_config)
        )
        self.source_manifest = load_strict_p11_backbone(
            self.backbone, source_checkpoint
        )
        source_phases = self.backbone.phase_snapshot().float()
        _validate_physical_phases(source_phases, self._phase_shape(), name="source_phases")
        self.register_buffer("source_phases", source_phases.clone(), persistent=True)

        # The ImageNet readout is neither part of the reusable checkpoint nor
        # the downstream optimization. Removing it also avoids unused DDP
        # parameters when only forward_features is called.
        self.backbone.readout = nn.Identity()
        self.backbone.stem.requires_grad_(False)

        default_outputs = {"caltech101": 101, "isic2016": 1, "lsp": 14}[task]
        outputs = default_outputs if num_outputs is None else int(num_outputs)
        if outputs <= 0:
            raise ValueError("num_outputs must be positive")
        if task == "caltech101":
            hidden_dim = (
                int(global_hidden_dim)
                if head_hidden_dim is None
                else int(head_hidden_dim)
            )
            self.head: nn.Module = GlobalTransferHead(
                token_count=self.backbone.stem.token_count,
                token_dim=self.backbone.token_dim,
                optical_banks=self.backbone.optical_channels,
                num_classes=outputs,
                hidden_dim=hidden_dim,
                dropout=float(head_dropout),
            )
        else:
            default_size = 224 if task == "isic2016" else 56
            if (
                dense_output_size is not None
                and output_size is not None
                and int(dense_output_size) != int(output_size)
            ):
                raise ValueError("dense_output_size and output_size disagree")
            selected_size = (
                int(dense_output_size)
                if dense_output_size is not None
                else int(output_size) if output_size is not None else default_size
            )
            width = int(decoder_width)
            if width <= 0:
                raise ValueError("decoder_width must be positive")
            decoder_channels = (3 * width, 5 * width // 2, 2 * width, 3 * width // 2, width)
            self.head = ProgressiveDenseHead(
                token_count=self.backbone.stem.token_count,
                token_dim=self.backbone.token_dim,
                optical_banks=self.backbone.optical_channels,
                output_channels=outputs,
                output_size=selected_size,
                decoder_channels=decoder_channels,
            )
        self.feedback_method: FeedbackMethod = "bp_current"
        self.feedback_random_seed: int | None = None
        self.configure_feedback("bp_current")

    def _phase_shape(self) -> tuple[int, int, int, int]:
        return (
            self.backbone.num_stages,
            self.backbone.optical_channels,
            self.backbone.canvas_size,
            self.backbone.canvas_size,
        )

    def configure_feedback(
        self,
        method: FeedbackMethod,
        *,
        random_seed: int = 0,
        pretrained_phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct runtime-only feedback state after build or resume."""

        if method not in {"noft", "bp", "bp_current", "fa_pretrained", "fa_random"}:
            raise ValueError(f"Unsupported feedback method: {method}")
        if method in {"noft", "bp", "bp_current"}:
            for stage in self.backbone.stages:
                stage.set_feedback("bp")
            self.feedback_random_seed = None
            self.feedback_method = method
            return self.phase_snapshot()

        if method == "fa_pretrained":
            phases = self.source_phases if pretrained_phases is None else pretrained_phases
            _validate_physical_phases(phases, self._phase_shape(), name="pretrained_phases")
            mode = "fa_pretrained"
            self.feedback_random_seed = None
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
            phases = 2.0 * math.pi * torch.rand(
                self._phase_shape(), generator=generator, dtype=torch.float32
            )
            mode = "fa_random"
            self.feedback_random_seed = int(random_seed)
        for index, stage in enumerate(self.backbone.stages):
            stage.set_feedback(mode, phases[index])
        self.feedback_method = method
        return self.feedback_snapshot()

    def feedback_snapshot(self) -> torch.Tensor:
        return torch.stack(
            [stage.feedback_phase.detach().cpu().clone() for stage in self.backbone.stages]
        )

    def feedback_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.feedback_method,
            "random_seed": self.feedback_random_seed,
        }
        if self.feedback_method == "fa_pretrained":
            payload["phase_sha256"] = _sha256_tensor(self.source_phases)
        elif self.feedback_method == "fa_random":
            payload["phase_sha256"] = _sha256_tensor(self.feedback_snapshot())
        return payload

    def phase_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.backbone.phase_parameters()

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.backbone.adapter_parameters()

    def residual_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.backbone.residual_parameters()

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        """Yield non-phase trainable backbone parameters for a separate LR group."""

        yield from self.adapter_parameters()
        yield from self.residual_parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.head.parameters()

    def set_backbone_trainable(self, enabled: bool) -> None:
        self.backbone.adapter.requires_grad_(enabled)
        self.backbone.stages.requires_grad_(enabled)
        self.backbone.stem.requires_grad_(False)

    def phase_snapshot(self) -> torch.Tensor:
        return self.backbone.phase_snapshot()

    def phase_report(self) -> dict[str, Any]:
        current = self.phase_snapshot()
        source = self.source_phases.detach().cpu()
        displacement = torch.atan2(
            torch.sin(current - source), torch.cos(current - source)
        ).abs()
        flattened = displacement.flatten(1)
        return {
            "mean_absolute_rad": float(displacement.mean()),
            "median_absolute_rad": float(displacement.median()),
            "fraction_over_0p1_rad": float((displacement > 0.1).float().mean()),
            "per_stage_mean_absolute_rad": [float(value) for value in flattened.mean(dim=1)],
            "per_stage_rms_rad": [
                float(value) for value in flattened.square().mean(dim=1).sqrt()
            ],
        }

    def parameter_report(self) -> dict[str, Any]:
        optical = sum(parameter.numel() for parameter in self.phase_parameters())
        adapter = sum(parameter.numel() for parameter in self.adapter_parameters())
        residual = sum(parameter.numel() for parameter in self.residual_parameters())
        electronic_backbone = adapter + residual
        head = sum(parameter.numel() for parameter in self.head_parameters())
        reusable = optical + electronic_backbone
        report = {
            "task": self.task,
            "source_checkpoint": self.source_manifest,
            "feedback": self.feedback_manifest(),
            "optical_phase_parameters": optical,
            "adapter_electronic_parameters": adapter,
            "residual_electronic_parameters": residual,
            "electronic_backbone_parameters": electronic_backbone,
            "temporary_head_parameters": head,
            "reusable_backbone_trainable_parameters": reusable,
            "total_trainable_parameters_including_head": reusable + head,
            "optical_fraction_of_reusable_backbone": optical / max(reusable, 1),
            "optical_fraction_including_temporary_head": optical
            / max(reusable + head, 1),
            "frozen_qwen_stem_parameters": self.backbone.stem.parameter_report()[
                "frozen_parameters"
            ],
            "old_imagenet_readout_parameters": 0,
            "head_contains_attention": False,
            "minimum_optical_gate": min(self.backbone.optical_gates()),
            "axis_schedule": self.backbone.axis_schedule,
            "source_phase_sha256": _sha256_tensor(self.source_phases),
        }
        if isinstance(self.head, GlobalTransferHead):
            report.update(
                {
                    "global_descriptor_dim": self.head.descriptor_dim,
                    "retrieval_embedding_dim": self.head.embedding_dim,
                }
            )
        else:
            report.update(
                {
                    "dense_output_size": self.head.output_size,
                    "dense_head_below_one_million": head < 1_000_000,
                }
            )
        return report

    def forward_features(
        self, images: torch.Tensor, *, ablation: str = "normal"
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return self.backbone.forward_features(images, ablation=ablation)

    def forward(
        self, images: torch.Tensor, *, ablation: str = "normal"
    ) -> dict[str, torch.Tensor]:
        final, _ = self.forward_features(images, ablation=ablation)
        return self.head(final)
