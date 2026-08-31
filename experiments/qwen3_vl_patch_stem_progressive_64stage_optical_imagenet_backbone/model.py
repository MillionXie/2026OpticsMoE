from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    rms_normalize,
)
from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.model import (
    CLIP_MEAN,
    CLIP_STD,
    TokenAdapter,
    TokenClassificationReadout,
)
from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    StaticQwenPatchStem,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    AxisOpticalOEOStage,
    OpticalAxis,
)
from experiments.qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone.model import (
    SlimSpatialTokenMixerSkip,
)


Ablation = Literal[
    "normal",
    "optical_off",
    "phase_random",
    "electronic_skip_off",
]
FeedbackMethod = Literal["bp_current", "fa_source", "fa_random"]

P13_SUPPORTED_DEPTHS = (16, 32, 64, 100)
P11_SOURCE_STAGE_COUNT = 8
P11_SOURCE_PAIR_COUNT = P11_SOURCE_STAGE_COUNT // 2
P13_FEEDBACK_FORMAT = "p13-full-depth-optical-feedback-v1"


def _sha256_tensor(value: torch.Tensor) -> str:
    """Hash one tensor without depending on its current device."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _phase_sequence_sha256(phases: torch.Tensor) -> str:
    if phases.ndim != 4:
        raise ValueError("A feedback phase sequence must have shape [D,C,H,W]")
    digest = hashlib.sha256()
    for index, phase in enumerate(phases):
        value = phase.detach().cpu().contiguous()
        digest.update(f"{index}:{value.dtype}:{tuple(value.shape)}:".encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _connector_random_seed(
    *,
    base_seed: int,
    num_stages: int,
    stage_index: int,
    axis: OpticalAxis,
) -> int:
    """Derive an auditable independent PRNG substream for one connector."""

    payload = (
        f"{P13_FEEDBACK_FORMAT}|base={int(base_seed)}|depth={int(num_stages)}|"
        f"stage={int(stage_index)}|axis={axis}"
    ).encode("utf-8")
    # torch.Generator.manual_seed accepts signed 64-bit seeds. Cryptographic
    # derivation avoids coupling feedback masks to model-initialization RNG
    # consumption or to the number/order of other connectors.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def anchor_stage_indices(num_stages: int) -> tuple[int, ...]:
    """Distribute the four ordered P11 token/channel pairs through P13.

    Every old pair remains consecutive and lands on the same token/channel
    parity. New pairs between anchors are exact bypasses while their outer
    depth alpha is zero, so the eight source stages still execute in their
    original order and produce the exact P11 body function.
    """

    depth = int(num_stages)
    if depth not in P13_SUPPORTED_DEPTHS:
        raise ValueError(
            f"P13 depth must be one of {P13_SUPPORTED_DEPTHS}, got {depth}"
        )
    pair_count = depth // 2
    pair_indices = tuple(
        round(source_pair * (pair_count - 1) / (P11_SOURCE_PAIR_COUNT - 1))
        for source_pair in range(P11_SOURCE_PAIR_COUNT)
    )
    if len(set(pair_indices)) != P11_SOURCE_PAIR_COUNT:
        raise RuntimeError("P13 anchor-pair mapping is not one-to-one")
    return tuple(
        stage
        for pair in pair_indices
        for stage in (2 * pair, 2 * pair + 1)
    )


class ProgressiveOpticalStageSlot(nn.Module):
    """One optical stage plus a non-trainable function-preserving depth gate."""

    def __init__(
        self,
        stage: AxisOpticalOEOStage,
        *,
        stage_index: int,
        source_stage_index: int | None,
        initial_alpha: float,
    ) -> None:
        super().__init__()
        alpha = float(initial_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Depth alpha must lie in [0,1]")
        self.stage = stage
        self.stage_index = int(stage_index)
        self.source_stage_index = (
            None if source_stage_index is None else int(source_stage_index)
        )
        self.is_anchor = self.source_stage_index is not None
        if self.is_anchor and alpha != 1.0:
            raise ValueError("Migrated P11 anchor stages must have alpha=1")
        self.register_buffer(
            "depth_alpha",
            torch.tensor(alpha, dtype=torch.float32),
            persistent=True,
        )
        # The growth schedule is controller state, not a learned scalar. Keep
        # a synchronized Python value so a deep CUDA forward does not incur a
        # device-to-host synchronization at every added stage.
        self._alpha_value = alpha

    @property
    def alpha_value(self) -> float:
        return self._alpha_value

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._alpha_value = float(self.depth_alpha.detach().cpu())

    def set_alpha(self, value: float) -> None:
        alpha = float(value)
        if self.is_anchor and alpha != 1.0:
            raise ValueError("P11 anchor alpha is locked to one")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Depth alpha must lie in [0,1]")
        self.depth_alpha.fill_(alpha)
        self._alpha_value = alpha

    def forward(
        self,
        amplitude: torch.Tensor,
        *,
        phase_override: torch.Tensor | None = None,
        optical_off: bool = False,
        disable_electronic_skip: bool = False,
    ) -> torch.Tensor:
        if self.is_anchor:
            return self.stage(
                amplitude,
                phase_override=phase_override,
                optical_off=optical_off,
                disable_electronic_skip=disable_electronic_skip,
            )

        # This branch is intentionally an exact Python-level bypass. Computing
        # Stage(x) and multiplying it by zero would waste almost all deep-model
        # compute and could turn a NaN into a non-identity 0*NaN result.
        if self._alpha_value == 0.0:
            return amplitude
        transformed = self.stage(
            amplitude,
            phase_override=phase_override,
            optical_off=optical_off,
            disable_electronic_skip=disable_electronic_skip,
        )
        alpha = self.depth_alpha.to(device=amplitude.device, dtype=amplitude.dtype)
        return amplitude + alpha * (transformed - amplitude)


class QwenStemProgressiveOpticalImageNetBackbone(nn.Module):
    """P13: a progressively grown 16/32/64/100-stage P11 optical body.

    Only eight distributed anchor stages own the migrated width-96 P11
    electronic mixers. Every added stage has an identity ElectronicSkipProcessor
    with zero trainable transform parameters, plus the one constrained optical
    fusion scalar already intrinsic to an OpticalOEOStage. Outer depth alphas
    are buffers rather than trainable electronics and are forced by a growth
    schedule toward one.
    """

    def __init__(self, stem_checkpoint: str | Path, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.stem_checkpoint = Path(stem_checkpoint).expanduser().resolve()
        self.stem = StaticQwenPatchStem(self.stem_checkpoint)
        self.stem.requires_grad_(False)
        self.canvas_size = int(config.get("canvas_size", 224))
        self.optical_channels = int(config.get("optical_channels", 3))
        self.num_stages = int(config.get("num_stages", 64))
        self.token_dim = int(config.get("token_dim", 224))
        self.num_classes = int(config.get("num_classes", 1000))
        self.mixer_width = int(config.get("mixer_width", 96))
        self.new_stage_alpha_init = float(config.get("new_stage_alpha_init", 0.0))
        self.new_stage_alpha_epsilon = float(
            config.get("new_stage_alpha_epsilon", 0.01)
        )
        self.new_stage_ramp_epochs = int(config.get("new_stage_ramp_epochs", 10))
        self.activation_checkpointing = bool(
            config.get("activation_checkpointing", False)
        )
        if (
            self.canvas_size != 224
            or self.optical_channels != 3
            or self.token_dim != 224
        ):
            raise ValueError("P13 retains the P11 three-bank 224x224 optical geometry")
        if self.num_stages not in P13_SUPPORTED_DEPTHS:
            raise ValueError(
                f"P13 num_stages must be one of {P13_SUPPORTED_DEPTHS}"
            )
        if self.mixer_width != 96:
            raise ValueError("P13 retains the eight width-96 P11 anchor mixers")
        if not 0.0 <= self.new_stage_alpha_init <= 1.0:
            raise ValueError("new_stage_alpha_init must lie in [0,1]")
        if not 0.0 < self.new_stage_alpha_epsilon <= 1.0:
            raise ValueError("new_stage_alpha_epsilon must lie in (0,1]")
        if self.new_stage_ramp_epochs <= 0:
            raise ValueError("new_stage_ramp_epochs must be positive")

        self.adapter = TokenAdapter(self.stem.hidden_size, self.token_dim)
        self.anchor_indices = anchor_stage_indices(self.num_stages)
        anchor_source = {
            target_index: source_index
            for source_index, target_index in enumerate(self.anchor_indices)
        }
        token_distance = float(
            config.get("token_axis_propagation_distance_m", 0.05)
        )
        channel_distance = float(
            config.get("channel_axis_propagation_distance_m", 0.05)
        )
        if token_distance <= 0.0 or channel_distance <= 0.0:
            raise ValueError("P13 axis propagation distances must be positive")
        self.token_axis_propagation_distance_m = token_distance
        self.channel_axis_propagation_distance_m = channel_distance

        slots: list[ProgressiveOpticalStageSlot] = []
        schedule: list[dict[str, Any]] = []
        for index in range(self.num_stages):
            axis: OpticalAxis = "token" if index % 2 == 0 else "channel"
            source_index = anchor_source.get(index)
            stage = self._make_stage(
                index=index,
                axis=axis,
                has_anchor_mixer=source_index is not None,
            )
            slot = ProgressiveOpticalStageSlot(
                stage,
                stage_index=index,
                source_stage_index=source_index,
                initial_alpha=(
                    1.0 if source_index is not None else self.new_stage_alpha_init
                ),
            )
            slots.append(slot)
            schedule.append(
                {
                    "stage": index + 1,
                    "axis": axis,
                    "distance_m": (
                        token_distance if axis == "token" else channel_distance
                    ),
                    "is_p11_anchor": source_index is not None,
                    "p11_source_stage": (
                        None if source_index is None else source_index + 1
                    ),
                    "has_width96_electronic_mixer": source_index is not None,
                }
            )
        self.slots = nn.ModuleList(slots)
        # Alias the conceptual stage sequence without registering every module
        # twice in state_dict. Callers should iterate optical_stages().
        self.axis_schedule = tuple(schedule)
        self.readout = TokenClassificationReadout(
            token_dim=self.token_dim,
            hidden_dim=int(config.get("head_hidden_dim", 448)),
            num_classes=self.num_classes,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "p13_progressive_architecture_signature",
            torch.tensor([13, 1, 2, self.num_stages], dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "p13_anchor_stage_indices",
            torch.tensor(self.anchor_indices, dtype=torch.int64),
            persistent=True,
        )
        # A complete source connector requires one frozen physical phase for
        # every target-depth stage. This is deliberately not an eight-stage
        # P11 sequence repeated through the deeper model. The source snapshot
        # is persistent because it is required to reconstruct FA after resume;
        # each stage's active ``feedback_phase`` remains runtime-only in the
        # inherited OpticalOEOStage and is excluded from state_dict.
        self.register_buffer(
            "feedback_source_phases",
            torch.stack(
                [slot.stage.phase().detach().clone() for slot in self.slots]
            ),
            persistent=True,
        )
        self.feedback_source_provenance: dict[str, Any] = {
            "capture": "p13_deterministic_initialization",
            "model_initialization_seed": int(config.get("seed", 2026)),
        }
        self.feedback_method: FeedbackMethod = "bp_current"
        self.feedback_random_base_seed: int | None = None
        self.feedback_connector_seeds: tuple[int, ...] = ()
        self.configure_feedback("bp_current")
        self.register_load_state_dict_post_hook(self._feedback_state_dict_loaded)
        self.migration_manifest: dict[str, Any] | None = None

    def _make_stage(
        self,
        *,
        index: int,
        axis: OpticalAxis,
        has_anchor_mixer: bool,
    ) -> AxisOpticalOEOStage:
        distance = (
            self.token_axis_propagation_distance_m
            if axis == "token"
            else self.channel_axis_propagation_distance_m
        )
        config = self.config
        stage = AxisOpticalOEOStage(
            optical_axis=axis,
            token_count=self.stem.token_count,
            size=self.canvas_size,
            channels=self.optical_channels,
            wavelength_m=float(config.get("wavelength_m", 5.32e-7)),
            pixel_size_m=float(config.get("pixel_size_m", 1.6e-5)),
            distance_m=distance,
            phase_init_std=float(config.get("phase_init_std", 0.10)),
            layernorm_eps=float(config.get("layernorm_eps", 1.0e-5)),
            residual_mode="constrained",
            residual_main_init=float(config.get("optical_gate_init", 0.60)),
            residual_main_min=float(config.get("optical_gate_min", 0.50)),
            normalize_branch_rms=True,
            random_seed=int(config.get("seed", 2026)) + 1009 * index,
            electronic_skip_mode="identity",
            long_skip_enabled=False,
            long_skip_weight_init=0.0,
            long_skip_weight_max=0.0,
        )
        if has_anchor_mixer:
            stage.electronic_skip = SlimSpatialTokenMixerSkip(
                field_size=self.canvas_size,
                token_count=self.stem.token_count,
                token_dim=self.token_dim,
                optical_banks=self.optical_channels,
                width=self.mixer_width,
                expansion=float(config.get("mixer_expansion", 2.0)),
                kernel_size=int(config.get("mixer_kernel_size", 3)),
                dropout=float(config.get("mixer_dropout", 0.10)),
                spatial_gate_init=float(
                    config.get("mixer_spatial_gate_init", 0.10)
                ),
                channel_gate_init=float(
                    config.get("mixer_channel_gate_init", 0.10)
                ),
                output_scale_init=float(config.get("residual_scale_init", 0.10)),
                output_scale_max=float(config.get("residual_scale_max", 0.25)),
                eps=float(config.get("layernorm_eps", 1.0e-5)),
            )
        return stage

    def p11_reference_config(self) -> dict[str, Any]:
        config = dict(self.config)
        config["num_stages"] = P11_SOURCE_STAGE_COUNT
        config["canvas_size"] = self.canvas_size
        config["optical_channels"] = self.optical_channels
        config["token_dim"] = self.token_dim
        config["mixer_width"] = self.mixer_width
        return config

    def train(self, mode: bool = True):
        super().train(mode)
        self.stem.eval()
        return self

    def optical_stages(self) -> tuple[AxisOpticalOEOStage, ...]:
        return tuple(slot.stage for slot in self.slots)

    def new_slots(self) -> tuple[ProgressiveOpticalStageSlot, ...]:
        return tuple(slot for slot in self.slots if not slot.is_anchor)

    def anchor_slots(self) -> tuple[ProgressiveOpticalStageSlot, ...]:
        return tuple(slot for slot in self.slots if slot.is_anchor)

    def _feedback_state_dict_loaded(
        self,
        module: nn.Module,
        incompatible_keys: Any,
    ) -> None:
        """Make the runtime-only connector contract explicit after resume."""

        del incompatible_keys
        if module is not self:
            raise RuntimeError("P13 feedback post-load hook received another module")
        self.feedback_source_provenance = {
            "capture": "loaded_from_persistent_state_dict",
            "original_provenance": "read_the_external_checkpoint_manifest",
        }
        # Runtime feedback phases and Python modes are intentionally absent
        # from state_dict. A resumed model therefore starts in current-BP mode
        # until the training controller explicitly reconstructs FA.
        self.configure_feedback("bp_current")

    def _validate_feedback_phases(
        self,
        phases: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        expected = (
            self.num_stages,
            self.optical_channels,
            self.canvas_size,
            self.canvas_size,
        )
        if not isinstance(phases, torch.Tensor) or tuple(phases.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        value = phases.detach().float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains a non-finite value")
        tolerance = 16.0 * torch.finfo(value.dtype).eps
        if bool((value < -tolerance).any()) or bool(
            (value > 2.0 * math.pi + tolerance).any()
        ):
            raise ValueError(f"{name} must contain physical phases in [0,2pi]")
        return value

    def capture_feedback_source(
        self,
        phases: torch.Tensor | None = None,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Freeze one distinct source connector for every target-depth stage.

        This operation is intended immediately after strict P11 migration or
        after a future deep-source pretraining run. It never expands/repeats
        the eight P11 anchor phases: all added-stage phases are captured from
        their own target slots in their own physical layouts.
        """

        selected = self.phase_snapshot() if phases is None else phases
        value = self._validate_feedback_phases(
            selected,
            name="feedback source phases",
        )
        self.feedback_source_phases.copy_(
            value.to(
                device=self.feedback_source_phases.device,
                dtype=self.feedback_source_phases.dtype,
            )
        )
        self.feedback_source_provenance = dict(provenance or {})
        self.feedback_source_provenance.setdefault(
            "capture", "explicit_current_full_depth_snapshot"
        )
        # Invalidate any old FA runtime copy. The caller must explicitly choose
        # fa_source/fa_random after the new source has been frozen.
        self.configure_feedback("bp_current")
        return self.feedback_source_manifest()

    def feedback_source_snapshot(self) -> torch.Tensor:
        return self.feedback_source_phases.detach().cpu().clone()

    def feedback_source_manifest(self) -> dict[str, Any]:
        phases = self.feedback_source_snapshot()
        return {
            "format": P13_FEEDBACK_FORMAT,
            "depth": self.num_stages,
            "connector_count": self.num_stages,
            "internal_interstage_connector_count": self.num_stages - 1,
            "adapter_input_connector_count": 1,
            "phase_shape": list(phases.shape),
            "phase_sequence_sha256": _phase_sequence_sha256(phases),
            "per_connector_phase_sha256": [
                _sha256_tensor(phase) for phase in phases
            ],
            "persistent_in_backbone_state_dict": True,
            "requires_grad": bool(self.feedback_source_phases.requires_grad),
            "provenance": dict(self.feedback_source_provenance),
        }

    def configure_feedback(
        self,
        method: FeedbackMethod,
        *,
        random_seed: int = 0,
    ) -> None:
        """Configure full-depth optical feedback without changing the forward.

        ``bp_current`` uses the current optical Jacobian. ``fa_source`` freezes
        every stage-input optical connector at the full-depth source snapshot.
        ``fa_random`` uses one separately seeded unit-modulus phase mask per
        stage. Since the complex-field connector is
        ``H_i diag(exp(j*phi_i))`` and the diagonal factor is unitary, every
        random connector has exactly the same complex-linear singular values
        and Frobenius norm as the corresponding source connector using the same
        propagation operator ``H_i``. The trainable amplitude is real-valued,
        so this must not be over-stated as an unconditional equality of the
        singular spectrum of every downstream real Jacobian.
        """

        if method not in {"bp_current", "fa_source", "fa_random"}:
            raise ValueError(f"Unsupported P13 feedback method: {method}")

        connector_seeds: list[int] = []
        if method == "bp_current":
            for slot in self.slots:
                slot.stage.set_feedback("bp")
            base_seed: int | None = None
        elif method == "fa_source":
            source = self._validate_feedback_phases(
                self.feedback_source_phases,
                name="persistent feedback source phases",
            )
            for index, slot in enumerate(self.slots):
                slot.stage.set_feedback("fa_pretrained", source[index])
            base_seed = None
        else:
            base_seed = int(random_seed)
            for index, slot in enumerate(self.slots):
                seed = _connector_random_seed(
                    base_seed=base_seed,
                    num_stages=self.num_stages,
                    stage_index=index,
                    axis=slot.stage.optical_axis,
                )
                connector_seeds.append(seed)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                random_phase = 2.0 * math.pi * torch.rand(
                    (
                        self.optical_channels,
                        self.canvas_size,
                        self.canvas_size,
                    ),
                    generator=generator,
                    dtype=torch.float32,
                )
                slot.stage.set_feedback("fa_random", random_phase)

        self.feedback_method = method
        self.feedback_random_base_seed = base_seed
        self.feedback_connector_seeds = tuple(connector_seeds)

    def feedback_snapshot(self) -> torch.Tensor:
        if self.feedback_method == "bp_current":
            # The stage runtime buffers are irrelevant/stale in exact BP mode;
            # report the dynamic physical forward phases instead.
            return self.phase_snapshot()
        return torch.stack(
            [
                slot.stage.feedback_phase.detach().cpu().clone()
                for slot in self.slots
            ]
        )

    def feedback_manifest(self) -> dict[str, Any]:
        phases = self.feedback_snapshot()
        source = self.feedback_source_snapshot()
        connections: list[dict[str, Any]] = []
        for index, slot in enumerate(self.slots):
            phase = phases[index]
            source_phase = source[index]
            random_seed = (
                self.feedback_connector_seeds[index]
                if self.feedback_method == "fa_random"
                else None
            )
            connections.append(
                {
                    "connector_index_zero_based": index,
                    "connector_role": (
                        "adapter_to_stage_input"
                        if index == 0
                        else "inter_stage_output_to_next_stage_input"
                    ),
                    "source_node": (
                        "adapter_output" if index == 0 else f"stage_{index - 1}_output"
                    ),
                    "target_optical_operator": f"stage_{index}_optical_branch",
                    "axis": slot.stage.optical_axis,
                    "is_p11_anchor_stage": slot.is_anchor,
                    "p11_source_stage_zero_based": slot.source_stage_index,
                    "frozen": self.feedback_method != "bp_current",
                    "runtime_buffer_used": self.feedback_method != "bp_current",
                    "feedback_phase_sha256": _sha256_tensor(phase),
                    "source_phase_sha256": _sha256_tensor(source_phase),
                    "propagation_transfer_sha256": _sha256_tensor(
                        slot.stage.propagator.transfer_function
                    ),
                    "random_substream_seed": random_seed,
                    "random_substream_is_stage_specific": random_seed is not None,
                    "modulation_elementwise_magnitude": 1.0,
                    "connector_scale_control": (
                        "exact_same_connector"
                        if self.feedback_method == "fa_source"
                        else "exact_complex_linear_spectrum_and_frobenius_norm_by_unitary_right_factor"
                        if self.feedback_method == "fa_random"
                        else "dynamic_current_connector"
                    ),
                    "real_amplitude_jacobian_spectrum_claimed_equal": False,
                }
            )

        feedback_equals_current = torch.equal(phases, self.phase_snapshot())
        return {
            "format": P13_FEEDBACK_FORMAT,
            "method": self.feedback_method,
            "depth": self.num_stages,
            "connector_count": len(connections),
            "internal_interstage_connector_count": self.num_stages - 1,
            "adapter_input_connector_count": 1,
            "feedback_phase_sequence_sha256": _phase_sequence_sha256(phases),
            "source": self.feedback_source_manifest(),
            "random_base_seed": self.feedback_random_base_seed,
            "random_connector_seeds_are_unique": (
                len(set(self.feedback_connector_seeds))
                == len(self.feedback_connector_seeds)
                if self.feedback_method == "fa_random"
                else None
            ),
            "feedback_equals_current_forward_phase": feedback_equals_current,
            "source_match_is_exact_at_undrifted_capture": (
                self.feedback_method == "fa_source" and feedback_equals_current
            ),
            "runtime_feedback_phase_persistent": False,
            "runtime_feedback_phase_requires_grad": False,
            "resume_contract": (
                "state_dict restores the persistent full-depth source snapshot; "
                "runtime mode/buffers reset to bp_current and the controller must "
                "call configure_feedback after load"
            ),
            "fixed_gradient_scope": (
                "only each optical complex-field connector to the preceding "
                "stage/input amplitude"
            ),
            "current_local_phase_gradient": "exact_autograd",
            "detector_normalization_residual_electronics_gradient": "exact_autograd",
            "outer_depth_blend_gradient": "exact_autograd",
            "connections": connections,
        }

    def set_new_stage_alpha(self, value: float) -> None:
        for slot in self.new_slots():
            slot.set_alpha(value)

    def apply_depth_ramp(self, epoch: int) -> float:
        """Set all new alphas for a forced epsilon-to-one growth schedule.

        Epoch zero is the exact P11 identity point. Epoch one begins at the
        configured epsilon so every new phase receives a local gradient; the
        last configured ramp epoch reaches alpha one exactly.
        """

        current = int(epoch)
        if current <= 0:
            value = 0.0
        elif self.new_stage_ramp_epochs == 1:
            value = 1.0
        else:
            progress = min(
                max((current - 1) / (self.new_stage_ramp_epochs - 1), 0.0),
                1.0,
            )
            value = self.new_stage_alpha_epsilon + (
                1.0 - self.new_stage_alpha_epsilon
            ) * progress
        self.set_new_stage_alpha(value)
        return value

    def depth_alpha_report(self) -> dict[str, Any]:
        values = [slot.alpha_value for slot in self.new_slots()]
        return {
            "new_stage_count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
            "all_full_depth": all(value == 1.0 for value in values),
            "all_exact_bypass": all(value == 0.0 for value in values),
        }

    def _images_01(self, images: torch.Tensor) -> torch.Tensor:
        return (images.float() * self.clip_std + self.clip_mean).clamp(0.0, 1.0)

    def optical_input(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            qwen_tokens = self.stem(self._images_01(images))
        tokens = self.adapter(qwen_tokens.detach())
        field = F.pad(tokens, (0, 0, 0, self.canvas_size - tokens.shape[1]))
        field = rms_normalize(
            field.unsqueeze(1).expand(-1, self.optical_channels, -1, -1),
            1.0e-5,
        )
        return field, qwen_tokens

    def _slot_forward(
        self,
        slot: ProgressiveOpticalStageSlot,
        amplitude: torch.Tensor,
        *,
        ablation: Ablation,
    ) -> torch.Tensor:
        phase_override = (
            slot.stage.random_phase if ablation == "phase_random" else None
        )
        return slot(
            amplitude,
            phase_override=phase_override,
            optical_off=ablation == "optical_off",
            disable_electronic_skip=ablation == "electronic_skip_off",
        )

    def forward_field(
        self,
        amplitude: torch.Tensor,
        *,
        ablation: Ablation = "normal",
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if ablation not in {
            "normal",
            "optical_off",
            "phase_random",
            "electronic_skip_off",
        }:
            raise ValueError(f"Unsupported ablation: {ablation}")
        expected = (self.optical_channels, self.canvas_size, self.canvas_size)
        if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]},{expected[2]}]")
        outputs: list[torch.Tensor] = []
        for slot in self.slots:
            if (
                self.activation_checkpointing
                and self.training
                and amplitude.requires_grad
                and (slot.is_anchor or slot.alpha_value != 0.0)
            ):
                # Bind the slot now: non-reentrant checkpoint recomputes this
                # closure during backward, after the Python loop has advanced.
                def run(
                    value: torch.Tensor,
                    current_slot: ProgressiveOpticalStageSlot = slot,
                ) -> torch.Tensor:
                    return self._slot_forward(
                        current_slot,
                        value,
                        ablation=ablation,
                    )

                amplitude = checkpoint(
                    run,
                    amplitude,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                amplitude = self._slot_forward(slot, amplitude, ablation=ablation)
            if return_intermediates:
                outputs.append(amplitude)
        return amplitude, tuple(outputs)

    def forward_features(
        self,
        images: torch.Tensor,
        *,
        ablation: Ablation = "normal",
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        amplitude, _ = self.optical_input(images)
        return self.forward_field(
            amplitude,
            ablation=ablation,
            return_intermediates=return_intermediates,
        )

    def forward(self, images: torch.Tensor, *, ablation: Ablation = "normal") -> torch.Tensor:
        final, _ = self.forward_features(
            images,
            ablation=ablation,
            return_intermediates=False,
        )
        return self.readout(final, self.stem.token_count)

    def phase_parameters(self):
        for slot in self.slots:
            yield slot.stage.raw_phase

    def adapter_parameters(self):
        yield from self.adapter.parameters()

    def residual_parameters(self):
        phase_ids = {id(parameter) for parameter in self.phase_parameters()}
        for slot in self.slots:
            for parameter in slot.stage.parameters():
                if id(parameter) not in phase_ids:
                    yield parameter

    def head_parameters(self):
        yield from self.readout.parameters()

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Reusable P13 state without the temporary ImageNet readout head."""

        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("readout.")
        }

    def phase_snapshot(self) -> torch.Tensor:
        return torch.stack(
            [slot.stage.phase().detach().cpu() for slot in self.slots]
        )

    def optical_gates(self) -> list[float]:
        return [
            float(slot.stage.residual.main_weight().detach().cpu())
            for slot in self.slots
        ]

    def parameter_report(self) -> dict[str, Any]:
        optical = sum(parameter.numel() for parameter in self.phase_parameters())
        adapter = sum(parameter.numel() for parameter in self.adapter_parameters())
        anchor_electronic = sum(
            parameter.numel()
            for slot in self.anchor_slots()
            for parameter in slot.stage.parameters()
            if parameter is not slot.stage.raw_phase
        )
        new_electronic = sum(
            parameter.numel()
            for slot in self.new_slots()
            for parameter in slot.stage.parameters()
            if parameter is not slot.stage.raw_phase
        )
        residual = anchor_electronic + new_electronic
        head = sum(parameter.numel() for parameter in self.head_parameters())
        electronic_backbone = adapter + residual
        backbone = optical + electronic_backbone
        identity_skip_parameters = sum(
            parameter.numel()
            for slot in self.new_slots()
            for parameter in slot.stage.electronic_skip.parameters()
        )
        return {
            "architecture": "p13_progressive_p11_token_channel",
            "num_stages": self.num_stages,
            "optical_macro_blocks": self.num_stages // 2,
            "axis_schedule": [slot.stage.optical_axis for slot in self.slots],
            "p11_anchor_stage_indices_zero_based": list(self.anchor_indices),
            "p11_anchor_mapping_zero_based": [
                {
                    "source": int(slot.source_stage_index),
                    "target": slot.stage_index,
                }
                for slot in self.anchor_slots()
            ],
            "unique_width96_mixer_instances": len(self.anchor_slots()),
            "new_stage_count": len(self.new_slots()),
            "new_stage_identity_skip_parameters": identity_skip_parameters,
            "outer_depth_gate_trainable_parameters": 0,
            "optical_phase_parameters": optical,
            "adapter_electronic_parameters": adapter,
            "anchor_stage_electronic_parameters": anchor_electronic,
            "new_stage_electronic_parameters": new_electronic,
            "residual_electronic_parameters": residual,
            "electronic_backbone_parameters": electronic_backbone,
            "temporary_imagenet_head_parameters": head,
            "backbone_trainable_parameters_excluding_head": backbone,
            "optical_fraction_of_backbone_trainable": optical / max(backbone, 1),
            "frozen_qwen_stem_parameters": self.stem.parameter_report()[
                "frozen_parameters"
            ],
            "minimum_optical_gate": min(self.optical_gates()),
            "depth_alpha": self.depth_alpha_report(),
            "activation_checkpointing": self.activation_checkpointing,
            "feedback_contract": {
                "format": P13_FEEDBACK_FORMAT,
                "method": self.feedback_method,
                "full_depth_connector_count": self.num_stages,
                "source_phase_sequence_sha256": _phase_sequence_sha256(
                    self.feedback_source_phases
                ),
                "source_snapshot_persistent": True,
                "runtime_feedback_persistent": False,
                "local_phase_and_electronics_gradients": "exact_autograd",
            },
            "migration_manifest": self.migration_manifest,
        }
