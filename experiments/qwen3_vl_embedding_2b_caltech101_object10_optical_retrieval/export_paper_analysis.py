"""Export frozen source data for Caltech-101 manuscript figures."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache_teacher_embeddings import TeacherEmbeddingStore
from .features import (
    move_inputs,
    preprocess_images,
    student_embeddings,
    validate_token_budgets,
)
from .io_utils import write_csv, write_json
from .modeling import build_optical_student, load_backbone
from .prepare_caltech101_retrieval_subset import (
    Caltech101RetrievalDataset,
    Caltech101Sample,
    collate_caltech101,
    prepare_caltech101_subset,
)
from .settings import Settings, load_settings
from .train_optical_retrieval import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _encode_and_trace(
    loaded: Any,
    replacement: Any,
    readout: Any,
    samples: Sequence[Caltech101Sample],
    settings: Settings,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    dataset = Caltech101RetrievalDataset(samples, settings.image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_caltech101,
    )
    loaded.model.eval()
    replacement.use_student()
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    readout.eval()
    embedding_chunks: list[torch.Tensor] = []
    detector_chunks: list[torch.Tensor] = []
    routing_rows: list[dict[str, Any]] = []
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    for batch in loader:
        inputs = preprocess_images(loaded.processor, batch["images"], settings.instruction)
        validate_token_budgets(inputs, settings)
        inputs = move_inputs(inputs, loaded.device)
        with torch.autocast(
            device_type=loaded.device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            embeddings, detector = student_embeddings(
                loaded.model, replacement, readout, inputs
            )
        embedding_chunks.append(embeddings.detach().float().cpu())
        detector_chunks.append(detector.detach().float().cpu())
        traces = {
            "vision": replacement.vision_surrogate.core.last_routing,
            "language": replacement.language_surrogate.core.last_routing,
        }
        for stack, routing in traces.items():
            probability = routing["probabilities"].detach().float().cpu()
            weight = routing["weights"].detach().float().cpu()
            selected = routing["selected_mask"].detach().cpu()
            if probability.shape[0] != len(batch["samples"]):
                raise RuntimeError(
                    f"{stack} router batch {probability.shape[0]} does not match "
                    f"sample batch {len(batch['samples'])}"
                )
            entropy = -(
                probability.clamp_min(1.0e-12).log() * probability
            ).sum(dim=-1) / math.log(float(settings.num_experts))
            for row_index, sample in enumerate(batch["samples"]):
                selected_pair = "+".join(
                    str(index + 1)
                    for index in torch.nonzero(selected[row_index], as_tuple=False)
                    .flatten()
                    .tolist()
                )
                for expert in range(settings.num_experts):
                    routing_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "split": sample.split,
                            "class_index": sample.class_index,
                            "class_name": sample.class_name,
                            "stack": stack,
                            "expert": expert + 1,
                            "selected": int(selected[row_index, expert]),
                            "routing_probability": float(probability[row_index, expert]),
                            "routing_weight": float(weight[row_index, expert]),
                            "normalized_entropy": float(entropy[row_index]),
                            "selected_pair": selected_pair,
                        }
                    )
    embeddings = torch.cat(embedding_chunks, dim=0)
    detectors = torch.cat(detector_chunks, dim=0)
    expected_embedding = (len(samples), settings.embedding_dim)
    expected_detector = (len(samples), settings.detector_output_size)
    if embeddings.shape != expected_embedding:
        raise RuntimeError(f"Student embedding shape {tuple(embeddings.shape)} != {expected_embedding}")
    if detectors.shape != expected_detector:
        raise RuntimeError(f"Detector shape {tuple(detectors.shape)} != {expected_detector}")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(detectors).all():
        raise RuntimeError("Paper-analysis embeddings/detectors contain NaN or Inf")
    return embeddings, detectors, routing_rows


def _phase_source(replacement: Any) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for stack, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        core = surrogate.core
        planes: list[tuple[str, Any]] = []
        for stage_index, stage in enumerate(core.expert_layers, 1):
            for expert_index, expert in enumerate(stage.experts, 1):
                planes.append((f"expert_stage{stage_index}_expert{expert_index}", expert))
        planes.append(("global", core.global_phase.phase))
        for name, layer in planes:
            phase = layer.phase().detach().float().cpu()
            array = phase.numpy().astype(np.float32)
            key = f"{stack}_{name}"
            arrays[key] = array
            phasor = torch.exp(1j * phase)
            resultant = float(phasor.mean().abs())
            rows.append(
                {
                    "stack": stack,
                    "plane": name,
                    "height": phase.shape[0],
                    "width": phase.shape[1],
                    "phase_mean_rad": float(phase.mean()),
                    "phase_std_rad": float(phase.std(unbiased=False)),
                    "phase_min_rad": float(phase.min()),
                    "phase_max_rad": float(phase.max()),
                    "phase_delta_from_pi_rms_rad": float(
                        (phase - math.pi).square().mean().sqrt()
                    ),
                    "circular_resultant_rho": resultant,
                    "phasor_dc_power": resultant * resultant,
                }
            )
    return arrays, rows


def _routing_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stack"]), str(row["class_name"]), int(row["expert"]))].append(row)
    output: list[dict[str, Any]] = []
    for (stack, class_name, expert), values in sorted(groups.items()):
        selected = np.asarray([int(row["selected"]) for row in values], dtype=np.float64)
        probability = np.asarray(
            [float(row["routing_probability"]) for row in values], dtype=np.float64
        )
        weight = np.asarray([float(row["routing_weight"]) for row in values], dtype=np.float64)
        entropy = np.asarray(
            [float(row["normalized_entropy"]) for row in values], dtype=np.float64
        )
        chosen = weight[selected.astype(bool)]
        output.append(
            {
                "stack": stack,
                "class_name": class_name,
                "expert": expert,
                "sample_count": len(values),
                "selection_frequency": float(selected.mean()),
                "mean_probability": float(probability.mean()),
                "mean_sparse_weight": float(weight.mean()),
                "mean_weight_when_selected": float(chosen.mean()) if len(chosen) else 0.0,
                "mean_normalized_entropy": float(entropy.mean()),
            }
        )
    return output


def export_analysis(
    config: Path,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    settings = load_settings(config)
    bundle = prepare_caltech101_subset(settings, persist=True)
    teacher_store = TeacherEmbeddingStore(settings.teacher_cache_path, bundle, settings)
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    loaded = load_backbone(settings, device)
    replacement, readout = build_optical_student(loaded, settings)
    try:
        checkpoint_payload = load_checkpoint(checkpoint, replacement, readout)
        samples = bundle.gallery_samples + bundle.test_samples
        teacher = teacher_store.lookup(samples).float()
        student, detector, routing_rows = _encode_and_trace(
            loaded, replacement, readout, samples, settings
        )
        if teacher.shape != student.shape:
            raise RuntimeError(
                f"Teacher/student source shape mismatch: {teacher.shape} != {student.shape}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        embedding_rows: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            row: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "class_index": sample.class_index,
                "class_name": sample.class_name,
                "image_path": str(sample.image_path),
                "teacher_student_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        teacher[index : index + 1], student[index : index + 1]
                    )[0]
                ),
            }
            row.update(
                {f"teacher_{column:02d}": float(teacher[index, column]) for column in range(settings.embedding_dim)}
            )
            row.update(
                {f"student_{column:02d}": float(student[index, column]) for column in range(settings.embedding_dim)}
            )
            embedding_rows.append(row)
        write_csv(output_dir / "embedding_source.csv", embedding_rows, list(embedding_rows[0]))
        write_csv(output_dir / "routing_assignments.csv", routing_rows, list(routing_rows[0]))
        summary_rows = _routing_summary(routing_rows)
        write_csv(output_dir / "routing_summary.csv", summary_rows, list(summary_rows[0]))
        phase_arrays, phase_rows = _phase_source(replacement)
        write_csv(output_dir / "phase_statistics.csv", phase_rows, list(phase_rows[0]))
        np.savez_compressed(
            output_dir / "analysis_arrays.npz",
            teacher_embeddings=teacher.numpy().astype(np.float32),
            student_embeddings=student.numpy().astype(np.float32),
            detector_features=detector.numpy().astype(np.float32),
            class_index=np.asarray([sample.class_index for sample in samples], dtype=np.int64),
            split=np.asarray([sample.split for sample in samples]),
            sample_id=np.asarray([sample.sample_id for sample in samples]),
            **phase_arrays,
        )
        metadata = {
            "config": str(config.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "manifest_sha256": bundle.manifest_digest,
            "sample_count": len(samples),
            "gallery_count": len(bundle.gallery_samples),
            "query_count": len(bundle.test_samples),
            "class_names": list(bundle.class_names),
            "teacher_embedding_shape": list(teacher.shape),
            "student_embedding_shape": list(student.shape),
            "detector_feature_shape": list(detector.shape),
            "routing_rows": len(routing_rows),
            "phase_planes": sorted(phase_arrays),
            "selection_policy": (
                "explicit checkpoint supplied by the caller; see checkpoint metadata "
                "and training config for its selection policy"
            ),
        }
        write_json(output_dir / "analysis_metadata.json", metadata)
        return metadata
    finally:
        replacement.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metadata = export_analysis(
        Path(args.config).expanduser().resolve(),
        Path(args.checkpoint).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
