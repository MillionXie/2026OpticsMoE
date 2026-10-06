"""Profile the three changed frozen-Qwen baselines on an otherwise idle A100.

The formal latency boundary is the pre-hook of the first native Vision
Transformer block through the task's final GPU-resident readout.  Model loading,
file I/O, image decoding, processor/tokenizer work, H2D copies, and patch
embedding are outside that boundary.  Timing and board telemetry are collected
in separate passes so ``nvidia-smi`` polling cannot perturb the formal latency.

This script intentionally handles only the latest baselines whose readout or
retrieval protocol changed after the 2026-09-09 A100 audit:

* SALICON: frozen Qwen Vision + aligned 192-wide density readout;
* OpenMoji: frozen complete Qwen + identical shared image/text readout;
* ABO image-to-image: frozen Qwen, enrolled-SKU, native-aspect 64-D prefix.

It can also reprofile the unchanged LSP and ABO image-to-text baselines under
the same empty-card/process-audit and independent-power-pass protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F


A100_POWER_LIMIT_W = 250.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def compute_processes(gpu_index: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",", 2)]
        if len(pieces) == 3 and pieces[0].isdigit():
            rows.append(
                {
                    "pid": int(pieces[0]),
                    "process_name": pieces[1],
                    "used_memory_mib": float(pieces[2]),
                }
            )
    return rows


class TelemetrySampler:
    def __init__(self, gpu_index: int, interval_ms: int = 10) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_ms = int(interval_ms)
        self.phase: str | None = None
        self.lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) != 4:
                continue
            try:
                power, utilization, memory, clock = map(float, pieces)
            except ValueError:
                continue
            with self.lock:
                phase = self.phase
            if phase is not None:
                self.samples.append(
                    {
                        "host_time_unix": time.time(),
                        "host_monotonic_s": time.monotonic(),
                        "phase": phase,
                        "power_w": power,
                        "gpu_utilization_percent": utilization,
                        "memory_used_mib": memory,
                        "sm_clock_mhz": clock,
                    }
                )

    def set_phase(self, phase: str | None) -> None:
        with self.lock:
            self.phase = phase

    def stop(self) -> list[dict[str, Any]]:
        self.set_phase(None)
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)
        return list(self.samples)


class FirstBlockTimer:
    def __init__(self, first_block: torch.nn.Module) -> None:
        self.state: dict[str, Any] = {}
        self.handle = first_block.register_forward_pre_hook(self._start)

    def _start(self, _module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if self.state.get("calls", 0) == 0:
            self.state["start_event"].record()
            self.state["host_started_ns"] = time.perf_counter_ns()
            if args and torch.is_tensor(args[0]):
                self.state["input_shape"] = list(args[0].shape)
        self.state["calls"] = int(self.state.get("calls", 0)) + 1

    def reset(self) -> None:
        self.state = {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "calls": 0,
        }

    def finish(self) -> dict[str, Any]:
        if int(self.state.get("calls", 0)) < 1:
            raise RuntimeError("First Transformer block was not called")
        self.state["end_event"].record()
        self.state["end_event"].synchronize()
        return {
            "cuda_event_ms": float(
                self.state["start_event"].elapsed_time(self.state["end_event"])
            ),
            "synchronized_wall_ms": (
                time.perf_counter_ns() - int(self.state["host_started_ns"])
            )
            / 1.0e6,
            "first_block_calls": int(self.state["calls"]),
            "first_block_input_shape": self.state.get("input_shape"),
        }

    def close(self) -> None:
        self.handle.remove()


@dataclass
class TaskRuntime:
    task_id: str
    task_name: str
    metric_name: str
    metric_value: float
    performance_samples: int
    input_contract: dict[str, Any]
    checkpoint: Path | None
    performance_report: Path
    model_path: Path
    first_block: torch.nn.Module
    forwards: list[Callable[[], dict[str, Any]]]
    close: Callable[[], None]


def _resolve_visual_output(value: Any) -> torch.Tensor:
    tensor = value[0] if isinstance(value, tuple) else value
    if not torch.is_tensor(tensor):
        raise RuntimeError("Vision hook did not receive a Tensor")
    return tensor


def load_salicon(args: argparse.Namespace, device: torch.device) -> TaskRuntime:
    from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import (
        FrozenQwenVisionTeacher,
        preprocess_vision,
    )
    from LightGenV2.tasks.t03_saliency.aligned_baseline import AlignedReadout
    from LightGenV2.tasks.t03_saliency.modeling import load_vision_backbone
    from LightGenV2.tasks.t03_saliency.settings import load_settings

    config = args.repo / "LightGenV2/tasks/t03_saliency/configs/qwen_aligned_head_50.yaml"
    checkpoint = args.repo / "LightGenV2/tasks/t03_saliency/runs/simulation/qwen_aligned_head_50_20260912_seed42/best_checkpoint.pt"
    performance_report = checkpoint.parent / "selected_checkpoint_test_evaluation.json"
    settings = load_settings(config)
    settings.model_id = str(args.embedding_model)
    settings.cache_dir = None
    settings.local_files_only = True
    settings.download = False
    loaded = load_vision_backbone(settings, device)
    head = AlignedReadout(settings.vision_hidden_size).to(device).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(payload["head"], strict=True)
    if sum(parameter.numel() for parameter in head.parameters()) != 282596:
        raise RuntimeError("SALICON aligned-head parameter contract changed")
    model = FrozenQwenVisionTeacher(loaded, head).eval()
    report = json.loads(performance_report.read_text(encoding="utf-8"))

    images = sorted(args.salicon_data.rglob("*.jpg"))
    validation = [path for path in images if "val" in path.as_posix().lower()]
    images = validation or images
    if len(images) < args.timing_samples:
        raise RuntimeError(f"Only {len(images)} SALICON images found")
    forwards: list[Callable[[], dict[str, Any]]] = []
    for image_path in images[: args.timing_samples]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = preprocess_vision(loaded.processor, [image], device)

        def forward(
            inputs: dict[str, torch.Tensor] = inputs,
            sample_id: str = image_path.name,
        ) -> dict[str, Any]:
            logits, spatial = model(inputs["pixel_values"], inputs["image_grid_thw"])
            density = logits.flatten(1).softmax(-1)
            return {
                "sample_id": sample_id,
                "output_shape": list(density.shape),
                "spatial_shape": list(spatial.shape),
            }

        forwards.append(forward)
    return TaskRuntime(
        task_id="T03",
        task_name="SALICON saliency",
        metric_name="CC",
        metric_value=float(report["metrics"]["cc"]),
        performance_samples=int(report["metrics"]["samples"]),
        input_contract={
            "images": "RGB SALICON validation images",
            "processor_pixels": [settings.processor_min_pixels, settings.processor_max_pixels],
            "output": "softmax-normalized 224x224 density map",
            "head": "Linear 1024->192 + LayerNorm + 85,412-parameter density decoder",
            "head_parameters": 282596,
        },
        checkpoint=checkpoint,
        performance_report=performance_report,
        model_path=args.embedding_model,
        first_block=loaded.visual.blocks[0],
        forwards=forwards,
        close=model.close,
    )


def load_lsp(args: argparse.Namespace, device: torch.device) -> TaskRuntime:
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
        LSPPoseDataset,
        _read_source,
        split_standard_protocol,
    )
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import (
        build_teacher,
        load_vision_backbone,
        preprocess_vision,
    )
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import (
        load_settings,
    )

    config = args.repo / "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt2.yaml"
    checkpoint = args.repo / "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_optical_moe16_opt2/checkpoints/teacher_best_train_loss.pt"
    performance_report = (
        args.existing_a100_root
        / "LightGenV2/tasks/t02_keypoint_detection/runs/simulation/qwen_frozen_a100_20260907/baseline_report.json"
    )
    settings = load_settings(config)
    settings.data_root = args.lsp_data
    settings.model_id = str(args.embedding_model)
    settings.cache_dir = None
    settings.local_files_only = True
    settings.download = False
    loaded = load_vision_backbone(settings, device)
    model = build_teacher(loaded, settings)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.head.load_state_dict(payload["head"], strict=True)
    model.eval()
    lsp = _read_source(settings.data_root / "lsp_dataset", "lsp", settings.visibility_policy)
    _train, test = split_standard_protocol(lsp, [])
    if len(test) != 1000:
        raise RuntimeError(f"Expected 1000 LSP test samples, got {len(test)}")
    dataset = LSPPoseDataset(test, settings, training=False)
    report = json.loads(performance_report.read_text(encoding="utf-8"))
    forwards: list[Callable[[], dict[str, Any]]] = []
    for index in range(args.timing_samples):
        item = dataset[index]
        inputs = preprocess_vision(loaded.processor, [item["image"]], device)

        def forward(
            inputs: dict[str, torch.Tensor] = inputs,
            sample_id: str = item["sample_id"],
        ) -> dict[str, Any]:
            heatmaps, spatial = model(inputs["pixel_values"], inputs["image_grid_thw"])
            coordinates = heatmaps.flatten(2).argmax(-1)
            return {
                "sample_id": sample_id,
                "spatial_shape": list(spatial.shape),
                "heatmap_shape": list(heatmaps.shape),
                "coordinate_index_shape": list(coordinates.shape),
            }

        forwards.append(forward)
    return TaskRuntime(
        task_id="T02",
        task_name="LSP keypoint detection",
        metric_name="PCK@0.2",
        metric_value=float(report["performance"]["pck_at_0.2_torso"]),
        performance_samples=int(report["performance"]["samples"]),
        input_contract={
            "image": "one RGB 224x224 person image",
            "output": "14x56x56 heatmaps and 14 peak indices",
            "head": report["readout"],
            "head_parameters": int(report["readout_trainable_parameters"]),
        },
        checkpoint=checkpoint,
        performance_report=performance_report,
        model_path=args.embedding_model,
        first_block=loaded.visual.blocks[0],
        forwards=forwards,
        close=model.close,
    )


def load_abo_image_text(args: argparse.Namespace, device: torch.device) -> TaskRuntime:
    from types import SimpleNamespace
    from LightGenV2.tasks.t08_abo_image_text_retrieval import baseline_5090d as baseline
    from LightGenV2.tasks.t01_object_retrieval.modeling import load_backbone
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
        move_inputs,
        teacher_embeddings,
    )

    performance_report = (
        args.existing_a100_root
        / "LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/qwen_frozen_a100_easy100_controlled_20260907/baseline_report.json"
    )
    report = json.loads(performance_report.read_text(encoding="utf-8"))
    queries, titles = baseline._load_contract(args.abo_easy100_data, None)
    settings = SimpleNamespace(
        model_id=str(args.embedding_model),
        cache_dir=None,
        local_files_only=True,
        processor_min_pixels=baseline.IMAGE_PIXELS,
        processor_max_pixels=baseline.IMAGE_PIXELS,
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    loaded = load_backbone(settings, device)
    title_embeddings = torch.stack(
        [baseline._embed_title(loaded, title.text) for title in titles]
    ).float()
    timing_queries = baseline._balanced_timing_queries(queries, args.timing_samples)
    forwards: list[Callable[[], dict[str, Any]]] = []
    for query in timing_queries:
        with Image.open(query.image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        inputs = move_inputs(
            baseline._inputs(
                loaded.processor,
                image=image,
                instruction=baseline.QUERY_INSTRUCTION,
            ),
            device,
        )

        def forward(
            inputs: dict[str, torch.Tensor] = inputs,
            sample_id: str = query.sample_id,
        ) -> dict[str, Any]:
            embedding = teacher_embeddings(
                loaded.model, inputs, baseline.EMBEDDING_DIM
            )[0]
            scores = embedding.float() @ title_embeddings.T
            ranking = scores.argsort(descending=True)
            return {
                "sample_id": sample_id,
                "embedding_shape": list(embedding.shape),
                "candidate_count": int(ranking.numel()),
            }

        forwards.append(forward)
    return TaskRuntime(
        task_id="T08",
        task_name="ABO image-to-text retrieval",
        metric_name="R@1",
        metric_value=float(report["performance"]["recall_at_1"]),
        performance_samples=int(report["test_samples"]),
        input_contract={
            "image": "one RGB 224x224 catalog image",
            "descriptor": "normalized 2048-D final valid-token embedding",
            "gallery": "100 precomputed title embeddings",
            "output": "100 cosine scores and full ranking",
            "trainable_parameters": 0,
        },
        checkpoint=None,
        performance_report=performance_report,
        model_path=args.embedding_model,
        first_block=loaded.model.model.visual.blocks[0],
        forwards=forwards,
        close=lambda: None,
    )


def _instruction_span(token_ids: list[int], instruction_ids: list[int]) -> slice:
    matches = [
        index
        for index in range(len(token_ids) - len(instruction_ids) + 1)
        if token_ids[index : index + len(instruction_ids)] == instruction_ids
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one instruction-token span, found {len(matches)}")
    return slice(matches[0], matches[0] + len(instruction_ids))


def load_openmoji(args: argparse.Namespace, device: torch.device) -> TaskRuntime:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from LightGenV2.tasks.t04_semantic_interaction.qwen_shared import QwenSharedReadout
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings

    config = args.repo / "LightGenV2/tasks/t04_semantic_interaction/configs/qwen_shared.yaml"
    checkpoint = args.repo / "LightGenV2/tasks/t04_semantic_interaction/runs/simulation/qwen_shared_s73/best_checkpoint.pt"
    performance_report = checkpoint.parent / "selected_checkpoint_test_evaluation.json"
    settings = load_settings(config)
    settings.qwen_checkpoint = args.instruct_model
    processor = AutoProcessor.from_pretrained(
        str(args.instruct_model),
        min_pixels=224**2,
        max_pixels=224**2,
        local_files_only=True,
    )
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.instruct_model),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    head = QwenSharedReadout(settings).to(device).eval().requires_grad_(False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("head"))
    if state is None:
        raise RuntimeError("OpenMoji checkpoint has neither model nor head state")
    head.load_state_dict(state, strict=True)
    head_parameters = sum(parameter.numel() for parameter in head.parameters())
    if head_parameters != 972952:
        raise RuntimeError(f"OpenMoji shared-head contract changed: {head_parameters}")
    report = json.loads(performance_report.read_text(encoding="utf-8"))
    metric_container = report.get("metrics", report)
    if "overall" in metric_container:
        metric_container = metric_container["overall"]
    metric_value = float(metric_container["changed_cell_accuracy"])

    rows = [
        json.loads(line)
        for line in settings.test_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 OpenMoji test rows, got {len(rows)}")
    captured: dict[str, torch.Tensor | None] = {"value": None}
    capture_handle = model.model.visual.blocks[-1].register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__(
            "value", _resolve_visual_output(output)
        )
    )
    forwards: list[Callable[[], dict[str, Any]]] = []
    for row in rows[: args.timing_samples]:
        image_path = settings.data_dir / row["relative_dir"] / row["files"]["source"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["instruction"]},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        instruction_ids = processor.tokenizer.encode(
            row["instruction"], add_special_tokens=False
        )
        span = _instruction_span(inputs["input_ids"][0].tolist(), instruction_ids)
        gpu_inputs = {key: value.to(device) for key, value in inputs.items()}

        def forward(
            gpu_inputs: dict[str, torch.Tensor] = gpu_inputs,
            span: slice = span,
            sample_id: str = row["sample_id"],
        ) -> dict[str, Any]:
            captured["value"] = None
            output = model.model(
                **gpu_inputs, use_cache=False, return_dict=True
            )
            visual = captured["value"]
            if visual is None:
                raise RuntimeError("OpenMoji final Vision block was not captured")
            language = output.last_hidden_state[0, span]
            result = head(visual.unsqueeze(0), [language])
            category = result["category_logits"].argmax(1)
            edit = result["edit_logits"].sigmoid().ge(0.5)
            return {
                "sample_id": sample_id,
                "vision_shape": list(visual.shape),
                "language_shape": list(language.shape),
                "category_shape": list(category.shape),
                "edit_shape": list(edit.shape),
            }

        forwards.append(forward)

    def close() -> None:
        capture_handle.remove()

    return TaskRuntime(
        task_id="T04",
        task_name="OpenMoji semantic interaction",
        metric_name="changed-cell accuracy",
        metric_value=metric_value,
        performance_samples=1000,
        input_contract={
            "image": "one RGB 224x224 scene",
            "text": "the original editing instruction inside the native chat wrapper",
            "vision_feature": "last pre-merger native Vision hidden [196,1024]",
            "language_feature": "last native Language hidden at instruction positions [L,2048]",
            "output": "6x6 category argmax and edit sigmoid threshold",
            "trainable_readout_parameters_total": head_parameters,
            "parameter_breakdown": {
                "image_adapter": 197184,
                "text_adapter": 393792,
                "shared_grid_readout": 381976,
            },
        },
        checkpoint=checkpoint,
        performance_report=performance_report,
        model_path=args.instruct_model,
        first_block=model.model.visual.blocks[0],
        forwards=forwards,
        close=close,
    )


def _load_abo_module(package_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "abo_qwen_reproduction", package_dir / "baseline.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load packaged ABO reproduction evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_abo_image_image(args: argparse.Namespace, device: torch.device) -> TaskRuntime:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    package_dir = args.abo_package
    abo = _load_abo_module(package_dir)
    manifest = package_dir / "evidence/abo_similarity10_manifest.csv"
    enrolled = package_dir / "evidence/enrolled_protocol.json"
    feature_path = package_dir / "evidence/enrolled/features.pt"
    performance_report = package_dir / "evidence/enrolled/final_report.json"
    rows = abo.load_rows(manifest, "enrolled_sku", enrolled, None)
    gallery_rows = [row for row in rows if row["split"] == "gallery"]
    query_rows = [row for row in rows if row["split"] == "query"]
    if len(gallery_rows) != 1600 or len(query_rows) != 800:
        raise RuntimeError("ABO enrolled protocol must contain 1600 gallery/800 query images")
    features = torch.load(feature_path, map_location="cpu", weights_only=True)
    ids = list(features["ids"])
    vectors = features.get("vectors", features.get("vectors_by_dimension", {}).get("64"))
    if vectors is None or tuple(vectors.shape) != (2400, 64):
        raise RuntimeError("ABO enrolled feature cache is not [2400,64]")
    by_id = {sample_id: index for index, sample_id in enumerate(ids)}
    gallery = F.normalize(
        torch.stack([vectors[by_id[row["sample_id"]]] for row in gallery_rows]).float(),
        dim=-1,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        str(args.embedding_model),
        local_files_only=True,
        min_pixels=50176,
        max_pixels=50176,
    )
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.embedding_model),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    report = json.loads(performance_report.read_text(encoding="utf-8"))
    metrics = report.get("metrics_by_dimension", {}).get("64", report.get("metrics", report))
    if "normal" in metrics:
        metrics = metrics["normal"]
    metric_value = float(metrics.get("hit_at_1", metrics.get("recall_at_1")))
    forwards: list[Callable[[], dict[str, Any]]] = []
    for row in query_rows[: args.timing_samples]:
        image_path = args.abo_data / row["image_path"]
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": abo.PROMPT}],
            },
            {"role": "user", "content": [{"type": "image", "image": image}]},
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        )
        gpu_inputs = {
            key: inputs[key].to(device)
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }

        def forward(
            gpu_inputs: dict[str, torch.Tensor] = gpu_inputs,
            sample_id: str = row["sample_id"],
        ) -> dict[str, Any]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(
                    **gpu_inputs, use_cache=False, return_dict=True
                ).last_hidden_state
                pooled = abo.pool_last(hidden, gpu_inputs["attention_mask"])
                descriptor = F.normalize(pooled[:, :64].float(), dim=-1)
                scores = descriptor @ gallery.T
                ranking = scores.argsort(dim=1, descending=True, stable=True)
            return {
                "sample_id": sample_id,
                "sequence_length": int(gpu_inputs["input_ids"].shape[1]),
                "descriptor_shape": list(descriptor.shape),
                "candidate_count": int(ranking.shape[1]),
            }

        forwards.append(forward)
    return TaskRuntime(
        task_id="T07",
        task_name="ABO image-to-image enrolled-SKU retrieval",
        metric_name="R@1",
        metric_value=metric_value,
        performance_samples=800,
        input_contract={
            "image": "native aspect; processor min_pixels=max_pixels=50176",
            "descriptor": "normalized first 64 dimensions of final valid-token hidden",
            "gallery": "1600 precomputed enrolled-SKU image descriptors",
            "output": "1600 cosine scores and stable full ranking",
            "trainable_parameters": 0,
        },
        checkpoint=None,
        performance_report=performance_report,
        model_path=args.embedding_model,
        first_block=model.model.visual.blocks[0],
        forwards=forwards,
        close=lambda: None,
    )


LOADERS = {
    "lsp": load_lsp,
    "abo_image_text": load_abo_image_text,
    "salicon": load_salicon,
    "openmoji": load_openmoji,
    "abo_image_image": load_abo_image_image,
}


def audit_no_competitor(gpu_index: int) -> list[dict[str, Any]]:
    rows = compute_processes(gpu_index)
    foreign = [row for row in rows if int(row["pid"]) != os.getpid()]
    if foreign:
        raise RuntimeError(f"A100 has competing compute processes: {foreign}")
    return rows


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected an A100 as cuda:0, got {torch.cuda.get_device_name(0)}")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if args.timing_samples < 20:
        raise ValueError("Formal timing requires at least 20 samples")
    args.output.mkdir(parents=True)
    process_audit_before_load = audit_no_competitor(args.physical_gpu_index)
    task = LOADERS[args.task](args, torch.device("cuda:0"))
    torch.cuda.synchronize()
    process_audit_before_timing = audit_no_competitor(args.physical_gpu_index)
    timer = FirstBlockTimer(task.first_block)
    raw_timing: list[dict[str, Any]] = []
    try:
        # No explicit warm-up: one model load followed by varied real samples.
        # The first sample is retained, matching the requested dataset-run protocol.
        for index, forward in enumerate(task.forwards):
            torch.cuda.synchronize()
            timer.reset()
            output_contract = forward()
            timing = timer.finish()
            raw_timing.append(
                {
                    "sample_index": index,
                    **timing,
                    "output_contract": json.dumps(output_contract, ensure_ascii=False),
                }
            )
            if index == 0 or (index + 1) % 25 == 0:
                print(
                    f"[{task.task_id} timing] {index + 1}/{len(task.forwards)} "
                    f"wall={timing['synchronized_wall_ms']:.3f} ms",
                    flush=True,
                )
    finally:
        timer.close()
    process_audit_after_timing = audit_no_competitor(args.physical_gpu_index)
    write_csv(args.output / "timing_per_sample.csv", raw_timing)

    sampler = TelemetrySampler(args.physical_gpu_index, args.telemetry_interval_ms)
    sampler.start()
    sampler.set_phase("idle_loaded_model")
    time.sleep(args.idle_seconds)
    sampler.set_phase(None)
    torch.cuda.synchronize()
    sampler.set_phase("active_continuous_inference")
    active_started = time.monotonic()
    active_calls = 0
    try:
        while time.monotonic() - active_started < args.active_seconds:
            task.forwards[active_calls % len(task.forwards)]()
            active_calls += 1
        torch.cuda.synchronize()
    finally:
        active_elapsed = time.monotonic() - active_started
        sampler.set_phase(None)
        telemetry = sampler.stop()
        task.close()
    process_audit_after_power = audit_no_competitor(args.physical_gpu_index)
    write_csv(args.output / "power_utilization_samples.csv", telemetry)

    idle = [row for row in telemetry if row["phase"] == "idle_loaded_model"]
    active = [row for row in telemetry if row["phase"] == "active_continuous_inference"]
    if not idle or not active:
        raise RuntimeError("Power pass did not capture both idle and active samples")
    wall_values = [float(row["synchronized_wall_ms"]) for row in raw_timing]
    cuda_values = [float(row["cuda_event_ms"]) for row in raw_timing]
    idle_power = summarize([float(row["power_w"]) for row in idle])
    active_power = summarize([float(row["power_w"]) for row in active])
    active_util = summarize([float(row["gpu_utilization_percent"]) for row in active])
    active_memory = summarize([float(row["memory_used_mib"]) for row in active])
    active_clock = summarize([float(row["sm_clock_mhz"]) for row in active])
    mean_wall_ms = statistics.fmean(wall_values)
    report = {
        "schema_version": 1,
        "status": "complete",
        "task_id": task.task_id,
        "task": task.task_name,
        "performance": {
            "metric": task.metric_name,
            "value": task.metric_value,
            "samples": task.performance_samples,
            "source_report": str(task.performance_report),
            "source_report_sha256": sha256_file(task.performance_report),
        },
        "timing": {
            "formal_clock": "Synchronized Wall",
            "formal_boundary": (
                "pre-hook at first native Vision Transformer block through all "
                "required native Vision/Language blocks and final task readout, "
                "ending after CUDA synchronization"
            ),
            "excluded": [
                "model/processor loading",
                "file I/O and image/video decode",
                "processor/tokenizer",
                "CPU-to-GPU transfer",
                "Vision patch embedding before native block 0",
            ],
            "protocol": "one model load; zero explicit warm-up; first varied real sample retained",
            "samples": len(raw_timing),
            "synchronized_wall_ms": summarize(wall_values),
            "cuda_event_ms_audit_only": summarize(cuda_values),
        },
        "power": {
            "protocol": "independent continuous-inference pass after formal timing",
            "sampler": "nvidia-smi board telemetry",
            "sampling_interval_ms": args.telemetry_interval_ms,
            "idle_seconds_requested": args.idle_seconds,
            "active_seconds_requested": args.active_seconds,
            "active_seconds_measured": active_elapsed,
            "active_forward_calls": active_calls,
            "idle_sample_count": len(idle),
            "active_sample_count": len(active),
            "idle_power_w": idle_power,
            "active_power_w": active_power,
            "active_gpu_utilization_percent": active_util,
            "active_memory_used_mib": active_memory,
            "active_sm_clock_mhz": active_clock,
            "measured_board_energy_j_per_inference": active_power["mean"] * mean_wall_ms / 1000.0,
            "idle_subtracted_energy_j_per_inference": max(
                0.0, active_power["mean"] - idle_power["mean"]
            )
            * mean_wall_ms
            / 1000.0,
            "rated_250w_upper_energy_j_per_inference": A100_POWER_LIMIT_W * mean_wall_ms / 1000.0,
        },
        "input_contract": task.input_contract,
        "checkpoint": (
            None
            if task.checkpoint is None
            else {
                "path": str(task.checkpoint),
                "sha256": sha256_file(task.checkpoint),
            }
        ),
        "model": str(task.model_path),
        "model_safetensors_sha256": sha256_file(task.model_path / "model.safetensors"),
        "process_audit": {
            "physical_gpu_index": args.physical_gpu_index,
            "before_load": process_audit_before_load,
            "before_timing": process_audit_before_timing,
            "after_timing": process_audit_after_timing,
            "after_power": process_audit_after_power,
            "rule": "no compute PID other than this profiler was allowed",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dtype": "bfloat16",
            "batch_size": 1,
            "rated_power_limit_w": A100_POWER_LIMIT_W,
        },
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    write_json(args.output / "report.json", report)
    write_json(args.output / "process_audit.json", report["process_audit"])
    (args.output / "command.txt").write_text(
        " ".join([os.environ.get("PYTHON", "python"), "-m", __spec__.name, *os.sys.argv[1:]])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(LOADERS))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--instruct-model", type=Path, required=True)
    parser.add_argument("--salicon-data", type=Path, required=True)
    parser.add_argument("--abo-data", type=Path, required=True)
    parser.add_argument("--abo-easy100-data", type=Path, required=True)
    parser.add_argument("--lsp-data", type=Path, required=True)
    parser.add_argument("--abo-package", type=Path, required=True)
    parser.add_argument("--existing-a100-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, default=6)
    parser.add_argument("--timing-samples", type=int, default=200)
    parser.add_argument("--telemetry-interval-ms", type=int, default=10)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--active-seconds", type=float, default=15.0)
    args = parser.parse_args()
    for name in (
        "repo", "embedding_model", "instruct_model", "salicon_data", "abo_data",
        "abo_easy100_data", "lsp_data", "abo_package", "existing_a100_root", "output",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
