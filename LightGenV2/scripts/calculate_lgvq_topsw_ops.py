"""Count LGVQ temporal baseline and optical-student MAC-derived operations.

The report is deliberately tied to the paper timing boundary and workload:

* Qwen3-VL: one 4-frame, 448x448 video, from Vision block 0 input through
  all Vision/Language blocks, the complete native vocabulary projection, and
  the five-quality-token score.  The result is multiplied by 16 because the
  paper row executes sixteen batch-1 calls.
* LightGenV2: one physical field containing sixteen videos x four frames.
  Electronic modules and their occurrence counts match the narrow A100 timing
  audit.  Six 478x478 coherent propagations are reported separately as dense
  complex-MVM equivalents.

The formal arithmetic convention is one real MAC = two OP.  A dense complex
MAC is also exposed both as one complex MAC and as eight real OP (four real
multiplications plus four real additions including accumulation).  Elementwise
activations, normalization, pooling, reductions, indexing, and softmax are not
included in the MAC-derived total; this is the usual model-FLOP convention and
keeps both models on the same basis.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def output_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"No tensor output in {type(value)!r}")


class ModuleMacCounter:
    """Count actual calls to Linear and convolution modules with runtime shapes."""

    def __init__(self) -> None:
        self.active = True
        self.rows: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def install(self, root: nn.Module, prefix: str = "") -> None:
        for name, module in root.named_modules():
            full_name = f"{prefix}.{name}" if prefix and name else (prefix or name)
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear(full_name)))
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self.handles.append(module.register_forward_hook(self._conv(full_name)))

    def _linear(self, name: str) -> Callable[..., None]:
        def hook(module: nn.Linear, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.active:
                return
            tensor = output_tensor(output)
            macs = int(tensor.numel()) * int(module.in_features)
            self.rows.append(
                {
                    "name": name,
                    "kind": "Linear",
                    "input_shape": list(inputs[0].shape),
                    "output_shape": list(tensor.shape),
                    "macs": macs,
                }
            )

        return hook

    def _conv(self, name: str) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.active:
                return
            tensor = output_tensor(output)
            kernel = 1
            for value in module.kernel_size:
                kernel *= int(value)
            per_output = kernel * int(module.in_channels) // int(module.groups)
            macs = int(tensor.numel()) * per_output
            self.rows.append(
                {
                    "name": name,
                    "kind": module.__class__.__name__,
                    "input_shape": list(inputs[0].shape),
                    "output_shape": list(tensor.shape),
                    "macs": macs,
                }
            )

        return hook

    def add_manual(
        self,
        *,
        name: str,
        kind: str,
        macs: int,
        input_shape: list[int] | None = None,
        output_shape: list[int] | None = None,
    ) -> None:
        self.rows.append(
            {
                "name": name,
                "kind": kind,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "macs": int(macs),
            }
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def group_qwen(name: str, kind: str) -> str:
    if kind.startswith("attention"):
        return "vision_attention_matmul" if "visual" in name else "language_attention_matmul"
    if name.startswith("qwen.model.visual.blocks"):
        return "vision_blocks_linear"
    if name.startswith("qwen.model.visual"):
        return "vision_post_blocks"
    if name.startswith("qwen.model.language_model.layers"):
        return "language_blocks_linear"
    if name == "qwen.lm_head":
        return "native_vocabulary_projection"
    if name.startswith("quality_head"):
        return "five_quality_token_readout"
    return "other_after_vision_block0"


@torch.inference_mode()
def count_qwen(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from LightGenV2.tasks.t06_video_quality_assessment import quality_token_common as core

    model_path = args.qwen_model.resolve()
    checkpoint_path = args.qwen_checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=448 * 448,
        max_pixels=448 * 448,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    quality_head = core.FiveNativeTokenRows(payload["state_dict"]["weight"]).to(device).eval()
    quality_scores = payload["level_scores"].float().to(device)
    rows = core.read_manifest(manifest_path)
    sample = next(row for row in rows if row["split"] == "test")
    prompt = core.render_prompt(processor, args.target)
    inputs, positions = core.prepare_inputs(
        sample, core.FRAME_FRACTIONS[4], processor, prompt, device, 448
    )

    counter = ModuleMacCounter()
    counter.active = False
    counter.install(model, "qwen")
    start_state: dict[str, Any] = {"started": False}

    def start_hook(_module: nn.Module, hook_inputs: tuple[Any, ...]) -> None:
        counter.active = True
        start_state["started"] = True
        start_state["vision_block0_shape"] = list(hook_inputs[0].shape)

    start_handle = model.model.visual.blocks[0].register_forward_pre_hook(start_hook)
    attention_handles: list[Any] = []
    vision_segments: list[dict[str, Any]] = []
    language_shapes: list[list[int]] = []

    def vision_attention_hook(
        name: str, heads: int, head_dim: int
    ) -> Callable[..., None]:
        def hook(_module: nn.Module, hook_inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if not counter.active:
                return
            hidden = hook_inputs[0] if hook_inputs else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError(f"{name}: missing hidden_states")
            cu = kwargs.get("cu_seqlens")
            if cu is None and len(hook_inputs) > 1:
                cu = hook_inputs[1]
            if cu is None:
                raise RuntimeError(f"{name}: missing cu_seqlens")
            lengths = (cu[1:] - cu[:-1]).detach().cpu().tolist()
            # QK^T and attention-probability times V.
            macs = 2 * heads * head_dim * sum(int(length) ** 2 for length in lengths)
            counter.add_manual(
                name=name,
                kind="attention_dense_segmented_QK_AV",
                macs=macs,
                input_shape=list(hidden.shape),
                output_shape=list(hidden.shape),
            )
            vision_segments.append(
                {"name": name, "lengths": lengths, "sum_length_squared": sum(int(x) ** 2 for x in lengths)}
            )

        return hook

    def language_attention_hook(
        name: str, heads: int, head_dim: int
    ) -> Callable[..., None]:
        def hook(_module: nn.Module, hook_inputs: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
            if not counter.active:
                return
            hidden = hook_inputs[0] if hook_inputs else _kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError(f"{name}: missing hidden_states")
            batch, length = int(hidden.shape[0]), int(hidden.shape[1])
            dense_macs = 2 * batch * heads * head_dim * length * length
            counter.add_manual(
                name=name,
                kind="attention_dense_QK_AV",
                macs=dense_macs,
                input_shape=list(hidden.shape),
                output_shape=list(hidden.shape),
            )
            language_shapes.append(list(hidden.shape))

        return hook

    for index, block in enumerate(model.model.visual.blocks):
        attention_handles.append(
            block.attn.register_forward_pre_hook(
                vision_attention_hook(
                    f"qwen.model.visual.blocks.{index}.attn.matmul",
                    int(block.attn.num_heads),
                    int(block.attn.dim // block.attn.num_heads),
                ),
                with_kwargs=True,
            )
        )
    for index, layer in enumerate(model.model.language_model.layers):
        attn = layer.self_attn
        attention_handles.append(
            attn.register_forward_pre_hook(
                language_attention_hook(
                    f"qwen.model.language_model.layers.{index}.self_attn.matmul",
                    int(attn.config.num_attention_heads),
                    int(attn.head_dim),
                ),
                with_kwargs=True,
            )
        )

    torch.cuda.synchronize(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
    mask = inputs["attention_mask"].bool()
    indices = torch.arange(mask.shape[1], device=device).expand_as(mask)
    last = indices.masked_fill(~mask, -1).amax(1)
    pooled = hidden[torch.arange(hidden.shape[0], device=device), last].float()
    _native_full_logits = model.lm_head(pooled.to(model.lm_head.weight.dtype))
    score = quality_head(pooled).softmax(-1) @ quality_scores
    torch.cuda.synchronize(device)
    counter.add_manual(
        name="quality_head.five_rows",
        kind="LinearEquivalent",
        macs=int(pooled.shape[0]) * int(pooled.shape[1]) * 5,
        input_shape=list(pooled.shape),
        output_shape=[int(pooled.shape[0]), 5],
    )
    counter.add_manual(
        name="quality_head.weighted_score",
        kind="DotProduct",
        macs=int(pooled.shape[0]) * 5,
        input_shape=[int(pooled.shape[0]), 5],
        output_shape=[int(pooled.shape[0])],
    )
    counter.active = False
    if not start_state["started"]:
        raise RuntimeError("Vision block-0 boundary hook did not fire")

    for handle in attention_handles:
        handle.remove()
    start_handle.remove()
    counter.close()

    grouped: dict[str, int] = collections.defaultdict(int)
    for row in counter.rows:
        grouped[group_qwen(row["name"], row["kind"])] += int(row["macs"])
    one_video_macs = sum(grouped.values())
    qwen_result = {
        "model": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "sample_id": sample["sample_id"],
        "selected_frame_positions": positions,
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "vision_block0_input_shape": start_state["vision_block0_shape"],
        "language_block_input_shapes_unique": sorted({tuple(x) for x in language_shapes}),
        "vision_attention_segment_patterns": vision_segments,
        "grouped_macs_per_video": dict(grouped),
        "macs_per_video": one_video_macs,
        "ops_per_video_1mac_eq_2op": 2 * one_video_macs,
        "macs_16_videos_batch1_sequential": 16 * one_video_macs,
        "ops_16_videos_1mac_eq_2op": 32 * one_video_macs,
        "score": float(score.item()),
        "native_logits_shape": list(_native_full_logits.shape),
        "module_rows": counter.rows,
    }
    del model, quality_head, inputs
    torch.cuda.empty_cache()
    return qwen_result


def count_component(
    name: str,
    module: nn.Module,
    function: Callable[[], Any],
    occurrence_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counter = ModuleMacCounter()
    counter.install(module, name)
    function()
    counter.close()
    one_call = sum(int(row["macs"]) for row in counter.rows)
    return (
        {
            "component": name,
            "occurrence_count": occurrence_count,
            "macs_per_occurrence": one_call,
            "macs_total": one_call * occurrence_count,
        },
        counter.rows,
    )


@torch.inference_mode()
def count_optical_student(device: torch.device) -> dict[str, Any]:
    from LightGenV2.scripts import profile_optical_electronics_5090d as legacy

    components: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []

    frame_readout = legacy.T06FrameReadout().to(device).eval()
    frame_patches = torch.rand(64, 56, 56, device=device)

    def frame_readout_call() -> torch.Tensor:
        normalized = legacy._normalize_patch(frame_patches)
        pooled = frame_readout.pool(normalized.unsqueeze(1)).squeeze(1)
        return frame_readout.output(F.softplus(frame_readout.norm(pooled)))

    video_readout = legacy.T06VideoReadout().to(device).eval()
    video_patches = torch.rand(16, 115, 115, device=device)

    def video_readout_call() -> torch.Tensor:
        normalized = legacy._normalize_patch(video_patches)
        pooled = F.adaptive_avg_pool2d(normalized.unsqueeze(1), (42, 96)).squeeze(1)
        return video_readout.output(F.softplus(video_readout.norm(pooled)))

    frame_hidden = torch.randn(16, 4, 49, 192, device=device)
    video_hidden = torch.randn(16, 42, 192, device=device)
    video_mask = torch.ones(16, 42, dtype=torch.bool, device=device)
    frame_residual = legacy.T06VisionResidual().to(device).eval()
    video_residual = legacy.T06LanguageResidual().to(device).eval()

    bridge = legacy.T06Bridge().to(device).eval()
    frame_summary = torch.randn(1, 16, 4, 384, device=device)
    serial_sequence = torch.randn(1, 16, 42, 192, device=device)

    def bridge_call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = bridge.frame_merger(frame_summary)
        serial = F.softplus(bridge.serial_field(serial_sequence))
        router_encoded = F.softplus(bridge.router_width(image))
        router = F.softplus(
            bridge.router_frames(router_encoded.transpose(-2, -1))
        ).transpose(-2, -1)
        return image, serial, router

    head = legacy.T06Head().to(device).eval()
    specifications = [
        ("frame_ccd_nonlinearity_readout_nn", frame_readout, frame_readout_call, 2),
        ("video_ccd_nonlinearity_readout_nn", video_readout, video_readout_call, 2),
        ("frame_parallel_residual_nn", frame_residual, lambda: frame_residual(frame_hidden), 2),
        ("video_parallel_residual_nn", video_residual, lambda: video_residual(video_hidden, video_mask), 2),
        ("bridge_nn_only", bridge, bridge_call, 1),
        ("task_head_nn", head, lambda: head(frame_hidden, video_hidden, video_mask), 1),
    ]
    for name, module, function, occurrences in specifications:
        component, rows = count_component(name, module, function, occurrences)
        components.append(component)
        module_rows.extend(rows)

    electronic_macs = sum(int(row["macs_total"]) for row in components)
    active_width = 478
    modes = active_width * active_width
    passes = 6
    complex_macs = passes * modes * modes
    # Complex multiply: 4 multiply + 2 add; complex accumulation: 2 add.
    optical_real_ops = 8 * complex_macs
    electronic_real_ops = 2 * electronic_macs
    return {
        "workload": "one physical field = 16 videos x 4 frames",
        "electronic_scope": (
            "exact narrow-A100 components: two frame/video CCD readouts, two "
            "frame/video residual calls, one bridge, one task head; router "
            "normalization/softmax has no MAC-derived contribution"
        ),
        "electronic_components": components,
        "electronic_module_rows": module_rows,
        "electronic_macs": electronic_macs,
        "electronic_ops_1mac_eq_2op": electronic_real_ops,
        "optical": {
            "active_field_width_px": active_width,
            "active_modes": modes,
            "coherent_full_field_propagations": passes,
            "dense_complex_macs": complex_macs,
            "real_ops_1_complex_mac_eq_8_real_ops": optical_real_ops,
            "alternative_ops_1_complex_mac_eq_2_ops": 2 * complex_macs,
        },
        "total_real_ops_recommended": optical_real_ops + electronic_real_ops,
        "total_real_ops_alternative_complex_mac_eq_2op": 2 * complex_macs + electronic_real_ops,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=("temporal", "spatial"), default="temporal")
    parser.add_argument("--ours-time-ms", type=float, default=8.660)
    parser.add_argument("--ours-energy-j", type=float, default=1.162)
    parser.add_argument("--qwen-time-ms-16", type=float, default=1200.053)
    parser.add_argument("--qwen-energy-j-16", type=float, default=103.784)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the exact Qwen runtime-shape audit")
    device = torch.device(args.device)
    qwen = count_qwen(args, device)
    ours = count_optical_student(device)

    ours_ops = int(ours["total_real_ops_recommended"])
    qwen_ops = int(qwen["ops_16_videos_1mac_eq_2op"])
    ours_alt_ops = int(ours["total_real_ops_alternative_complex_mac_eq_2op"])
    summary = {
        "schema_version": 1,
        "status": "complete",
        "arithmetic_convention": {
            "real_mac": "1 real MAC = 2 OP",
            "complex_mac_recommended": (
                "1 dense complex MAC = 8 real OP: four real multiplies, two "
                "adds inside the complex product, two accumulation adds"
            ),
            "mac_derived_scope": (
                "Linear/Conv and attention QK/AV; excludes elementwise activation, "
                "normalization, pooling, reduction, indexing and softmax"
            ),
        },
        "paper_red_box": {
            "ours": {"time_ms_16_videos": args.ours_time_ms, "energy_j_16_videos": args.ours_energy_j},
            "qwen3vl": {"time_ms_16_videos": args.qwen_time_ms_16, "energy_j_16_videos": args.qwen_energy_j_16},
        },
        "ours": ours,
        "qwen3vl": qwen,
        "topsw": {
            "ours_effective_recommended": (ours_ops / 1.0e12) / args.ours_energy_j,
            "ours_effective_alternative_complex_mac_eq_2op": (ours_alt_ops / 1.0e12) / args.ours_energy_j,
            "qwen3vl_actual_mac_derived": (qwen_ops / 1.0e12) / args.qwen_energy_j_16,
        },
        "throughput_tops": {
            "ours_effective_recommended": (ours_ops / 1.0e12) / (args.ours_time_ms / 1000.0),
            "qwen3vl_actual_mac_derived": (qwen_ops / 1.0e12) / (args.qwen_time_ms_16 / 1000.0),
        },
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "operation_count_and_topsw.json", summary)
    with (output / "qwen_module_macs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(qwen["module_rows"][0]))
        writer.writeheader()
        writer.writerows(qwen["module_rows"])
    with (output / "ours_electronic_module_macs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ours["electronic_module_rows"][0]))
        writer.writeheader()
        writer.writerows(ours["electronic_module_rows"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
