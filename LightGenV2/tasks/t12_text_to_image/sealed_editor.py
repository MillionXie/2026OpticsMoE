"""Self-describing editor weights, without legacy checkpoint dependencies."""
from dataclasses import asdict
import torch
from torch import nn
from .audited_unified import (AuditedUnifiedEditor, AuditedQwenTextEncoder, AuditedSpatialBottleneck,
                             AuditedLatentMidBlock, add_decoder_refinement)
from .qwen_mini_small import QwenMiniConfig


def construction_metadata(model, source, args):
    metadata = {"kind": model.kind, "text_config": asdict(model.text.config)}
    if model.kind == "small":
        metadata["editor_config"] = asdict(model.editor.config)
    else:
        from diffusers import UNet2DConditionModel
        metadata.update(unet_config=dict(UNet2DConditionModel.load_config(args.initial_unet, subfolder="unet")),
                        student_widths=source["student_widths"], model_config=source["model_config"],
                        training_config=source["training_config"], vae_config=dict(model.vae.config))
        adapter = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
        metadata["adapter_config"] = dict(adapter["config"], pca_rank=model.adapter.basis.shape[0])
        metadata["adapter_text_dim"] = adapter["settings"]["text_dim"]
        metadata["condition_tokens"] = model.adapter.token_count
        metadata["condition_dim"] = model.adapter.condition_dim
    return metadata


def build_sealed(saved):
    cfg, state = saved["construction"], saved["model"]
    text = AuditedQwenTextEncoder(QwenMiniConfig(**cfg["text_config"]))
    if cfg["kind"] == "small":
        from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor
        values = dict(cfg["editor_config"])
        values["widths"] = tuple(values["widths"])
        editor = SmallFullFrameEditor(SmallEditorConfig(**values))
        editor.text = text
        editor.bottleneck = AuditedSpatialBottleneck(values["widths"][-1], values["condition_dim"])
        model = AuditedUnifiedEditor(kind="small", text=text, editor=editor)
    else:
        from diffusers import AutoencoderKL
        from .progressive_student import build_narrow_optical_unet
        from .product_repair_model import RepairModelConfig
        from .electronic_turbo import QwenTurboConditionAdapter, TurboAdapterConfig
        unet, legacy, _ = build_narrow_optical_unet(cfg["unet_config"], cfg["student_widths"],
                                                  RepairModelConfig(**cfg["model_config"]))
        unet.mid_block = AuditedLatentMidBlock(legacy)
        adapter = QwenTurboConditionAdapter(cfg["adapter_text_dim"], TurboAdapterConfig(**cfg["adapter_config"]),
                    state["adapter.teacher_mean"], state["adapter.basis"], state["adapter.coefficient_mean"],
                    state["adapter.coefficient_std"], cfg["condition_tokens"], cfg["condition_dim"])
        vae = AutoencoderKL.from_config(cfg["vae_config"]).requires_grad_(False)
        model = AuditedUnifiedEditor(kind="large", text=text, unet=unet, adapter=adapter,
                    bridge=nn.Linear(text.config.width,2048), vae=vae, training=cfg["training_config"])
    if saved.get("decoder_refinement", False):
        add_decoder_refinement(model)
    model.text_mlp_indices = saved.get("text_mlp_indices")
    model.construction = cfg
    model.load_state_dict(state, strict=True)
    return model
