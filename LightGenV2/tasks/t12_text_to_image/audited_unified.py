"""T12 v2: live Qwen-style language blocks and fixed-geometry audited optics.

Legacy checkpoints are initialization sources, never advertised as trained v2.
The optical path is reused from T01 (including its optical router installation).
224 expert tiles sit inside the 478 active ROI on the 518 propagation canvas.
Token pooling does NOT resize phase parameters. No catalogue classifier is used.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .modeling import ConditionedElectronicResidual, ScaleMatchedFusion
from .qwen_mini_small import QwenMiniConfig, QwenMiniTextEncoder

ARCHITECTURE = "t12_live_qwen_parallel_language_vision_dc20_v2"


def audited_settings():
    from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
    config = Path(__file__).resolve().parents[1] / "t01_object_retrieval/configs/moe_optical_router_scale_matched_dc20.yaml"
    settings = load_settings(config)
    if (settings.active_size, settings.canvas_size, settings.expert_size,
            settings.num_experts, settings.top_k) != (478, 518, 224, 4, 2):
        raise ValueError("T12 requires the audited 224/478/518 MoE4 top-2 contract")
    if settings.router_backend != "optical":
        raise ValueError("An electronic router must not silently replace the optical router")
    return settings


def optical_path(width: int, max_tokens: int):
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import MoE4LanguageTwoBlockOpticalPath
    from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.modeling import _router_for
    settings = audited_settings()
    if max_tokens > settings.input_adapter_dim:
        raise ValueError("Token rows cannot exceed the audited 224-row input field")
    path = MoE4LanguageTwoBlockOpticalPath(width, settings, max_tokens=max_tokens)
    # The generic path defaults to an electronic router: explicitly replace it.
    path.core.router = _router_for(settings, path.core.geometry)
    return path


def fusion():
    return ScaleMatchedFusion(.5, .4, .75, 1e-6)


class AuditedQwenTextEncoder(QwenMiniTextEncoder):
    """Keep the existing two Qwen-style blocks, fuse optical results in each.

    E1(x) || router/expert(x); E2(fused1) || global(reloaded fused1).
    These are width-pruned Qwen-style blocks, NOT original pretrained Qwen layers.
    """
    def __init__(self, config: QwenMiniConfig):
        if config.layers != 2:
            raise ValueError("The audited language contract has exactly two stages")
        super().__init__(config)
        self.optical = optical_path(config.width, config.max_length)
        self.fusion1, self.fusion2 = fusion(), fusion()

    def hidden(self, embeddings, mask):
        valid = mask.bool()
        if not bool(valid.any(1).all()):
            raise ValueError("Every prompt needs at least one valid token")
        # Pack valid tokens; supports left or right padding without shifting the
        # final-token pooling index. No text-feature cache bypasses this graph.
        length = int(valid.sum(1).max())
        packed = embeddings.new_zeros(len(embeddings), length, embeddings.shape[-1])
        packed_valid = torch.zeros(len(embeddings), length, dtype=torch.bool, device=mask.device)
        for i in range(len(embeddings)):
            n = int(valid[i].sum())
            packed[i, :n] = embeddings[i, valid[i]]
            packed_valid[i, :n] = True
        value = self.input_projection(packed)
        padding = ~packed_valid
        electronic1 = self.layers[0](value, packed_valid)
        with torch.autocast(value.device.type, enabled=False):
            optical1, routing, lengths = self.optical.run_expert_block(value.float(), padding)
        stage2 = self.fusion1(electronic1, optical1.to(electronic1.dtype))
        stage2 = stage2.masked_fill(padding[..., None], 0)
        electronic2 = self.layers[1](stage2, packed_valid)
        with torch.autocast(value.device.type, enabled=False):
            field = self.optical.encode_global_input(stage2.float(), padding, routing)
            optical2 = self.optical.run_global_block(field, lengths, padding, torch.float32)
        value = self.norm(self.fusion2(electronic2, optical2.to(electronic2.dtype)))
        self.last_routing = routing
        pooled = value[torch.arange(len(value), device=value.device), packed_valid.sum(1)-1]
        self.current_hidden = pooled
        return pooled


class AuditedSpatialBottleneck(nn.Module):
    """Two learned electronic residuals parallel to audited expert/global stages."""
    def __init__(self, width: int, condition_dim: int, grid: int = 14):
        super().__init__()
        if grid ** 2 > 224:
            raise ValueError("Spatial token count exceeds the physical input rows")
        self.grid = grid
        self.condition = nn.Linear(condition_dim, width)
        self.electronic1 = ConditionedElectronicResidual(width, condition_dim, grid)
        self.electronic2 = ConditionedElectronicResidual(width, condition_dim, grid)
        self.optical = optical_path(width, grid ** 2)
        self.fusion1, self.fusion2 = fusion(), fusion()

    def forward(self, value, condition):
        pooled = F.adaptive_avg_pool2d(value, (self.grid, self.grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        shared = tokens + self.condition(condition)[:, None]
        padding = torch.zeros(shared.shape[:2], dtype=torch.bool, device=value.device)
        electronic1 = self.electronic1(shared, condition)
        with torch.autocast(value.device.type, enabled=False):
            optical1, routing, lengths = self.optical.run_expert_block(shared.float(), padding)
        stage2 = self.fusion1(electronic1, optical1.to(electronic1.dtype))
        electronic2 = self.electronic2(stage2, condition)
        with torch.autocast(value.device.type, enabled=False):
            field = self.optical.encode_global_input(stage2.float(), padding, routing)
            optical2 = self.optical.run_global_block(field, lengths, padding, torch.float32)
        result = self.fusion2(electronic2, optical2.to(electronic2.dtype))
        self.last_routing = routing
        delta = (result-tokens).transpose(1, 2).reshape_as(pooled)
        # Preserve unpooled spatial detail through a feature residual, not a
        # foreground mask or post-hoc pasted image.
        return value + F.interpolate(delta, value.shape[-2:], mode="bilinear", align_corners=False)


class AuditedLatentMidBlock(nn.Module):
    has_cross_attention = True

    def __init__(self, legacy):
        super().__init__()
        # Retain trained interface weights; remove the compact FFT path entirely.
        for name in ("input_norm", "input_projection", "timestep_projection",
                     "condition_projection", "output_norm", "output_projection", "output_gate"):
            setattr(self, name, getattr(legacy, name))
        self.channels = legacy.channels
        width = self.input_projection.out_features
        self.hybrid = AuditedSpatialBottleneck(width, width)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, **kwargs):
        if temb is None or encoder_hidden_states is None:
            raise ValueError("Both timestep and live text conditions are required")
        tokens = hidden_states.flatten(2).transpose(1, 2)
        projected = self.input_projection(self.input_norm(tokens))
        projected = projected.transpose(1, 2).reshape(len(tokens), -1, *hidden_states.shape[-2:])
        condition = self.timestep_projection(temb) + self.condition_projection(encoder_hidden_states.mean(1))
        result = self.hybrid(projected, condition)
        delta = self.output_projection(self.output_norm(result.flatten(2).transpose(1, 2)))
        delta = delta.transpose(1, 2).reshape_as(hidden_states)
        return hidden_states + torch.sigmoid(self.output_gate) * delta


class AuditedUnifiedEditor(nn.Module):
    def __init__(self, *, kind, text, editor=None, unet=None, adapter=None, bridge=None, vae=None, training=None):
        super().__init__()
        self.kind, self.text = kind, text
        self.editor, self.unet, self.adapter, self.bridge, self.vae = editor, unet, adapter, bridge, vae
        self.training_config = training or {}

    def forward(self, reference, embeddings, mask, noise):
        if reference.shape[-2:] != (256, 256):
            raise ValueError("Both versions require 256x256 input/output")
        if self.kind == "small":
            return self.editor(reference, (embeddings, mask), noise)
        from .product_repair_model import one_step_edit
        pooled = self.bridge(self.text.hidden(embeddings, mask))
        condition = self.adapter.condition(pooled)
        with torch.no_grad():
            latent = self.vae.encode(reference).latent_dist.mode() * self.vae.config.scaling_factor
        latent_noise = F.interpolate(noise, latent.shape[-2:], mode="bilinear", align_corners=False)
        # The caller supplies a real four-channel Gaussian noise tensor for the
        # large model. Interpolation is unnecessary in the normal runner.
        if latent_noise.shape[1] != latent.shape[1]:
            raise ValueError("Large editor noise must have four channels")
        edited = one_step_edit(self.unet, latent_noise, latent, condition, latent.new_tensor(1.),
                              noise_scale=float(self.training_config.get("noise_scale", .05)),
                              residual_scale=float(self.training_config.get("residual_scale", .25)))
        self.current_latent = edited
        return self.vae.decode(edited / self.vae.config.scaling_factor, return_dict=False)[0]


def migrate(payload: dict, *, initial_unet=None, vae_checkpoint=None, adapter_checkpoint=None):
    """Return a NEW initialized model; never mutate/save over legacy weights."""
    from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor
    kind = "small" if "editor_config" in payload else "large"
    frontend = payload if kind == "small" else payload["text_frontend"]
    config = QwenMiniConfig(**frontend["qwen_mini_config" if kind == "small" else "config"])
    text = AuditedQwenTextEncoder(config)
    state = ({key.removeprefix("text."): value for key, value in payload["model"].items() if key.startswith("text.")}
             if kind == "small" else frontend["text"])
    incompatible = text.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not name.startswith(("optical.", "fusion1.", "fusion2."))
                                           for name in incompatible.missing_keys):
        raise ValueError(f"Unexpected language warm-start mismatch: {incompatible}")
    if kind == "small":
        values = dict(payload["editor_config"])
        values.update(image_size=256, widths=tuple(values["widths"]))
        editor = SmallFullFrameEditor(SmallEditorConfig(**values))
        editor.text = QwenMiniTextEncoder(config)
        editor.load_state_dict(payload["model"], strict=True)
        old = editor.bottleneck
        hybrid = AuditedSpatialBottleneck(values["widths"][-1], values["condition_dim"])
        hybrid.electronic1.load_state_dict(old.electronic.state_dict())
        hybrid.electronic2.load_state_dict(old.electronic.state_dict())
        editor.bottleneck, editor.text = hybrid, text
        return AuditedUnifiedEditor(kind=kind, text=text, editor=editor)
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from .electronic_turbo_infer import _load_adapter
    from .product_repair_model import RepairModelConfig
    from .progressive_student import build_narrow_optical_unet
    config_unet = UNet2DConditionModel.load_config(initial_unet, subfolder="unet", local_files_only=True)
    unet, legacy, _ = build_narrow_optical_unet(config_unet, payload["student_widths"], RepairModelConfig(**payload["model_config"]))
    unet.load_state_dict(payload["unet"])
    unet.mid_block = AuditedLatentMidBlock(legacy)
    adapter, _ = _load_adapter(Path(adapter_checkpoint), torch.device("cpu"))
    adapter.load_state_dict(payload["adapter"])
    bridge = nn.Linear(text.config.width, 2048)
    bridge.load_state_dict(frontend["bridge"])
    vae = AutoencoderKL.from_pretrained(vae_checkpoint, subfolder="vae", variant="fp16", local_files_only=True)
    vae.requires_grad_(False)
    return AuditedUnifiedEditor(kind=kind, text=text, unet=unet, adapter=adapter,
                               bridge=bridge, vae=vae, training=payload["training_config"])


def architecture_report(model):
    # named_parameters deduplicates the shared text module in the small editor.
    parameters = dict(model.named_parameters())
    phases = {name: list(p.shape) for name, p in parameters.items() if "raw_phase" in name or "raw_router_phase" in name}
    total = sum(p.numel() for p in parameters.values())
    fixed_condition = (sum(value.numel() for value in model.adapter.buffers())
                       if model.adapter is not None else 0)
    components = {}
    for name, parameter in parameters.items():
        component = name.split(".")[0]
        components[component] = components.get(component, 0)+parameter.numel()
    return {"architecture": ARCHITECTURE, "kind": model.kind, "counted_parameters": total,
            "components": components,
            "fixed_condition_buffer_values": fixed_condition,
            "parameters_plus_fixed_condition_values": total+fixed_condition,
            "parameter_limit": 15_000_000 if model.kind == "small" else 150_000_000,
            "within_limit": total+fixed_condition <= (15_000_000 if model.kind == "small" else 150_000_000),
            "pure_phase_parameters": sum(parameters[n].numel() for n in phases),
            "phase_tensors": phases, "physical_active_roi": [478, 478], "propagation_canvas": [518, 518],
            "expert_tiles": [4, 224, 224], "language_optics": True, "vision_optics": True,
            "electronic_optical_parallel": True, "alpha_minimum": .4,
            "design_router": False, "image_size": [256, 256],
            "language_kind": "two trained width-pruned Qwen-style blocks, not pretrained original Qwen layers",
            "text_config": asdict(model.text.config), "optical_paths": 2,
            "decoder_refinement": getattr(model, "decoder_refinement", False),
            "hardware_timing": "not measured; map router/expert/global traversals before applying six-layer latency",
            "status": "initialized architecture migration; requires fine-tuning and quality validation"}


def reduce_condition_rank(model, rank):
    """Keep leading PCA directions and matching trained coefficient rows."""
    if model.adapter is None:
        raise ValueError("Only the latent editor has a PCA condition adapter")
    adapter = model.adapter
    old = adapter.predictor[-1]
    if not 0 < rank <= old.out_features:
        raise ValueError("PCA rank must be positive and cannot expand a checkpoint")
    if rank == old.out_features:
        return
    reduced = nn.Linear(old.in_features, rank).to(old.weight.device, old.weight.dtype)
    with torch.no_grad():
        reduced.weight.copy_(old.weight[:rank]); reduced.bias.copy_(old.bias[:rank])
    adapter.predictor[-1] = reduced
    adapter.basis = adapter.basis[:rank].contiguous()
    adapter.coefficient_mean = adapter.coefficient_mean[:rank].contiguous()
    adapter.coefficient_std = adapter.coefficient_std[:rank].contiguous()


def prune_text_mlp(model, width, indices=None):
    """Structured SwiGLU neuron pruning; preserve all attention/optical widths."""
    from dataclasses import replace
    old_width = model.text.config.intermediate_width
    if not 0 < width <= old_width:
        raise ValueError("Cannot expand or empty the text MLP")
    selected = []
    for i, layer in enumerate(model.text.layers):
        if indices is None:
            importance = layer.down_proj.weight.norm(dim=0)*(layer.up_proj.weight.norm(dim=1)
                                                            +layer.gate_proj.weight.norm(dim=1))
            keep = importance.topk(width).indices.sort().values
        else:
            keep = torch.as_tensor(indices[i], device=layer.down_proj.weight.device)
            if len(keep) != width or len(keep.unique()) != width:
                raise ValueError("Invalid recorded MLP indices")
        for name in ("gate_proj", "up_proj"):
            old = getattr(layer, name)
            new = nn.Linear(old.in_features, width, bias=False).to(old.weight)
            with torch.no_grad():
                new.weight.copy_(old.weight[keep])
            setattr(layer, name, new)
        old = layer.down_proj
        new = nn.Linear(width, old.out_features, bias=False).to(old.weight)
        with torch.no_grad():
            new.weight.copy_(old.weight[:, keep])
        layer.down_proj = new
        selected.append(keep.cpu().tolist())
    model.text.config = replace(model.text.config, intermediate_width=width)
    model.text_mlp_indices = selected


class ConditionedDetailResidual(nn.Module):
    """Zero-initialized decoder feature refinement, not RGB postprocessing."""
    def __init__(self, channels, hidden, condition_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, 3, padding=1)
        self.norm = nn.GroupNorm(16, hidden)
        self.affine = nn.Linear(condition_dim, 2*hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self, value, condition):
        scale, shift = self.affine(condition).chunk(2, dim=1)
        hidden = self.norm(self.conv1(value))*(1+scale[:,:,None,None])+shift[:,:,None,None]
        return value + .1*self.conv2(F.silu(hidden))


class RefinedUpBlock(nn.Module):
    def __init__(self, base, channels, hidden, depth, condition_dim):
        super().__init__()
        self.base = base
        self.details = nn.ModuleList([ConditionedDetailResidual(channels, hidden, condition_dim)
                                      for _ in range(depth)])

    def forward(self, value, skip, condition):
        value = self.base(value, skip, condition)
        for block in self.details:
            value = block(value, condition)
        return value


def add_decoder_refinement(model):
    if model.kind != "small" or getattr(model, "decoder_refinement", False):
        raise ValueError("Detail expansion requires a small unexpanded editor")
    editor = model.editor
    widths = editor.config.widths
    editor.up3 = RefinedUpBlock(editor.up3, widths[2], 256, 5, editor.config.condition_dim)
    editor.up2 = RefinedUpBlock(editor.up2, widths[1], 160, 2, editor.config.condition_dim)
    model.decoder_refinement = True


def optical_diagnostics(model):
    spatial = model.editor.bottleneck if model.kind == "small" else model.unet.mid_block.hybrid
    result = {}
    for label, branch in (("language", model.text), ("vision", spatial)):
        routing = getattr(branch, "last_routing", {})
        statistics = {}
        for key in ("weights", "selected_mask", "probabilities"):
            value = routing.get(key)
            if isinstance(value, torch.Tensor):
                statistics[key+"_last_batch_mean"] = value.detach().float().mean(0).cpu().tolist()
        result[label] = {"alpha1": float(branch.fusion1.alpha.detach()),
                         "alpha2": float(branch.fusion2.alpha.detach()), "routing": statistics}
    return result
