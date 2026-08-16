from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs,
    preprocess_images,
    student_embeddings,
    validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    seed_everything,
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    export_centered_bmp,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    episodic_prototype_retrieval_loss,
    load_checkpoint,
    supervised_contrastive_loss,
)

from .modeling import build_hybrid_student, load_backbone
from .optical_blocks import LanguageSecondLayerOpticalCore
from .settings import load_settings


def _key(sample: Any) -> str:
    digest = hashlib.sha1(sample.sample_id.encode("utf-8")).hexdigest()[:10]
    safe_class = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in sample.sku_name
    )
    return f"{sample.split}__{sample.sku_index:02d}__{safe_class}__{digest}"


def _load_hybrid_checkpoint(path: Path, replacement: Any, readout: Any) -> None:
    load_checkpoint(path, replacement, readout)
    replacement.set_phase_dropout_active(False)
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    readout.eval()


@torch.no_grad()
def export_session(settings: Any, checkpoint: Path, session_dir: Path) -> None:
    device = torch.device(
        settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    bundle = prepare_caltech101_subset(settings, persist=True)
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_hybrid_student(loaded, settings)
    _load_hybrid_checkpoint(checkpoint, replacement, readout)
    session_dir.mkdir(parents=True, exist_ok=True)
    core = replacement.language_surrogate.core
    phase = core.optical_branch.phase.phase().detach().cpu()
    phase_report = export_centered_bmp(
        phase,
        session_dir / "phase_mask" / "language_block2_phase.bmp",
        value_type="phase",
        scale_factor=2,
        slm_width=1920,
        slm_height=1200,
        flip_vertical=True,
    )
    torch.save(phase, session_dir / "phase_mask" / "language_block2_phase_rad.pt")
    for name in ("block2_input", "electronic_block2_output", "simulation_ccd"):
        (session_dir / name).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    try:
        for index, sample in enumerate(bundle.all_samples(), 1):
            with Image.open(sample.image_path) as source:
                image = ImageOps.fit(
                    source.convert("RGB"),
                    (settings.image_size, settings.image_size),
                    method=Image.Resampling.BICUBIC,
                )
            inputs = preprocess_images(loaded.processor, [image], settings.instruction)
            validate_token_budgets(inputs, settings)
            inputs = move_inputs(inputs, loaded.device)
            student_embeddings(loaded.model, replacement, readout, inputs)
            key = _key(sample)
            amplitude = core.optical_branch.last_amplitude[0].cpu()
            simulated_ccd = core.optical_branch.last_raw_ccd[0].cpu()
            block2_input = core.last_block2_input_groups[0].cpu()
            electronic = core.last_electronic_block2_groups[0].cpu()
            torch.save(
                block2_input.to(torch.float16),
                session_dir / "block2_input" / f"{key}.pt",
            )
            torch.save(
                electronic.to(torch.float16),
                session_dir / "electronic_block2_output" / f"{key}.pt",
            )
            torch.save(
                simulated_ccd.to(torch.float32),
                session_dir / "simulation_ccd" / f"{key}.pt",
            )
            amplitude_report = export_centered_bmp(
                amplitude,
                session_dir / "amplitude_to_play" / f"{key}.bmp",
                value_type="amplitude",
                scale_factor=2,
                slm_width=1920,
                slm_height=1080,
                amplitude_encoding_mode="positive_percentile",
                amplitude_percentile=99.5,
                amplitude_gamma=1.0,
            )
            rows.append(
                {
                    "order": index - 1,
                    "key": key,
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "sku_index": sample.sku_index,
                    "sku_name": sample.sku_name,
                    "token_length": len(block2_input),
                    "image_path": str(sample.image_path),
                    "amplitude_bmp": amplitude_report["path"],
                    "ccd_basename": key,
                }
            )
            if index % 20 == 0 or index == len(bundle.all_samples()):
                print(f"[hardware_export] {index}/{len(bundle.all_samples())}", flush=True)
    finally:
        replacement.close()
    write_csv(session_dir / "manifest.csv", rows, list(rows[0]))
    write_json(
        session_dir / "deployment.json",
        {
            "checkpoint": str(checkpoint),
            "manifest_digest": bundle.manifest_digest,
            "samples": len(rows),
            "phase": phase_report,
            "logical_amplitude_shape": [224, 224],
            "physical_active_shape": [448, 448],
            "ccd_rule": "strict 2x2 block mean from 448x448 to 224x224",
            "ccd_semantics": "captured files are intensity and must never be squared again",
            "normalization": (
                "background quantile subtraction; divide by per-frame mean; "
                "relative clipping; log1p; row LayerNorm"
            ),
        },
    )
    (session_dir / "ccd_captured").mkdir(exist_ok=True)


def _read_manifest(session_dir: Path) -> list[dict[str, str]]:
    path = session_dir / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing hardware manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_image_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".npy":
        value = torch.from_numpy(np.load(path))
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim == 3:
            array = array[..., :3].mean(axis=-1)
        value = torch.from_numpy(np.asarray(array).copy())
    return value.squeeze().float()


def load_ccd(
    session_dir: Path,
    key: str,
    *,
    use_simulation: bool,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> torch.Tensor:
    if use_simulation:
        path = session_dir / "simulation_ccd" / f"{key}.pt"
    else:
        root = session_dir / "ccd_captured"
        candidates = [root / f"{key}{suffix}" for suffix in (".pt", ".npy", ".tif", ".tiff", ".png")]
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one measured CCD for {key} below {root}, found {len(matches)}"
            )
        path = matches[0]
    value = _load_image_tensor(path)
    if flip_vertical:
        value = torch.flip(value, (-2,))
    if flip_horizontal:
        value = torch.flip(value, (-1,))
    if tuple(value.shape) == (448, 448):
        value = value.reshape(224, 2, 224, 2).mean(dim=(1, 3))
    if tuple(value.shape) != (224, 224):
        raise RuntimeError(
            f"CCD {path} must be 224x224 or physical 448x448, got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise RuntimeError(f"CCD {path} contains invalid intensity")
    return value


def _load_downstream(settings: Any, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    language_state = payload["language_optical"]
    input_weight = language_state["core.input_adapter.weight"]
    core = LanguageSecondLayerOpticalCore(
        int(input_weight.shape[1]), settings.max_language_tokens, settings
    ).to(device)
    core_state = {
        name.removeprefix("core."): value
        for name, value in language_state.items()
        if name.startswith("core.")
    }
    core.load_state_dict(core_state, strict=True)
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(device)
    readout.load_state_dict(payload["retrieval_readout"], strict=True)
    core.requires_grad_(False)
    core.optical_branch.ccd_normalizer.requires_grad_(True)
    core.optical_branch.decoder.requires_grad_(True)
    core.output_norm.requires_grad_(True)
    core.optical_fusion_logit.requires_grad_(True)
    readout.requires_grad_(True)
    return payload, core, readout


def _embedding(
    row: dict[str, str],
    session_dir: Path,
    core: LanguageSecondLayerOpticalCore,
    readout: ElectronicRetrievalReadout,
    device: torch.device,
    *,
    use_simulation: bool,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> torch.Tensor:
    key = row["key"]
    electronic = torch.load(
        session_dir / "electronic_block2_output" / f"{key}.pt",
        map_location="cpu",
        weights_only=True,
    ).to(device)
    ccd = load_ccd(
        session_dir,
        key,
        use_simulation=use_simulation,
        flip_vertical=flip_vertical,
        flip_horizontal=flip_horizontal,
    ).to(device)
    features = core.detector_features_from_cached(electronic, ccd)
    return readout(features.unsqueeze(0))[0]


@torch.no_grad()
def _evaluate(
    rows: list[dict[str, str]],
    session_dir: Path,
    core: Any,
    readout: Any,
    device: torch.device,
    **ccd_options: Any,
) -> float:
    gallery = [row for row in rows if row["split"] == "gallery"]
    test = [row for row in rows if row["split"] == "test"]
    gallery_embeddings = torch.stack(
        [_embedding(row, session_dir, core, readout, device, **ccd_options) for row in gallery]
    )
    gallery_labels = torch.tensor([int(row["sku_index"]) for row in gallery], device=device)
    classes = torch.unique(gallery_labels, sorted=True)
    prototypes = torch.stack(
        [F.normalize(gallery_embeddings[gallery_labels == label].mean(0), dim=0) for label in classes]
    )
    correct = 0
    for row in test:
        embedding = _embedding(row, session_dir, core, readout, device, **ccd_options)
        predicted = int(classes[(embedding @ prototypes.T).argmax()])
        correct += predicted == int(row["sku_index"])
    return correct / max(1, len(test))


def finetune_session(
    settings: Any,
    checkpoint: Path,
    session_dir: Path,
    *,
    use_simulation: bool,
    flip_vertical: bool,
    flip_horizontal: bool,
    epochs: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _read_manifest(session_dir)
    payload, core, readout = _load_downstream(settings, checkpoint, device)
    train_rows = [row for row in rows if row["split"] == "train"]
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        grouped[int(row["sku_index"])].append(row)
    if len(grouped) != 10 or any(len(items) < 2 for items in grouped.values()):
        raise RuntimeError("Hardware fine-tuning requires ten classes with >=2 train captures")
    parameters = [
        parameter
        for module in (core, readout)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-4, weight_decay=0.01)
    ccd_options = {
        "use_simulation": use_simulation,
        "flip_vertical": flip_vertical,
        "flip_horizontal": flip_horizontal,
    }
    best_loss = float("inf")
    generator = torch.Generator().manual_seed(settings.random_seed)
    for epoch in range(1, epochs + 1):
        core.eval()
        core.optical_branch.ccd_normalizer.train()
        core.optical_branch.decoder.train()
        core.output_norm.train()
        readout.train()
        steps = max(1, len(train_rows) // 30)
        epoch_loss = 0.0
        for _ in range(steps):
            batch_rows = []
            for label in sorted(grouped):
                indexes = torch.randperm(len(grouped[label]), generator=generator)[:3]
                batch_rows.extend(grouped[label][int(index)] for index in indexes)
            embeddings = torch.stack(
                [_embedding(row, session_dir, core, readout, device, **ccd_options) for row in batch_rows]
            )
            labels = torch.tensor([int(row["sku_index"]) for row in batch_rows], device=device)
            contrastive = supervised_contrastive_loss(
                embeddings, labels, settings.temperature
            )
            prototype, _, _ = episodic_prototype_retrieval_loss(
                embeddings, labels, settings.gallery_temperature
            )
            loss = contrastive + prototype
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_loss += float(loss.detach())
        average = epoch_loss / steps
        top1 = _evaluate(rows, session_dir, core, readout, device, **ccd_options)
        print(
            f"[hardware_finetune] epoch={epoch:03d}/{epochs:03d} "
            f"loss={average:.5f} test_top1={top1:.4f} "
            f"fusion={float(core.optical_fusion.detach()):.4f}",
            flush=True,
        )
        if average < best_loss:
            best_loss = average
            updated_language = dict(payload["language_optical"])
            updated_language.update(
                {f"core.{name}": value.cpu() for name, value in core.state_dict().items()}
            )
            updated = dict(payload)
            updated["language_optical"] = updated_language
            updated["retrieval_readout"] = {
                name: value.cpu() for name, value in readout.state_dict().items()
            }
            updated["hardware_finetune"] = {
                "source_checkpoint": str(checkpoint),
                "use_simulation": use_simulation,
                "epoch": epoch,
                "train_loss": average,
                "observed_test_top1": top1,
                "trainable_scope": "CCD normalizer, decoder, fusion, output norm, readout",
            }
            torch.save(updated, session_dir / "hardware_finetuned_checkpoint.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Language block-2 optical hardware bridge")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--phase", choices=("export", "finetune"), required=True)
    parser.add_argument("--use-simulation", action="store_true")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    settings = load_settings(args.config)
    seed_everything(settings.random_seed)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    session_dir = Path(args.session_dir).expanduser().resolve()
    if args.phase == "export":
        export_session(settings, checkpoint, session_dir)
    else:
        finetune_session(
            settings,
            checkpoint,
            session_dir,
            use_simulation=args.use_simulation,
            flip_vertical=args.flip_vertical,
            flip_horizontal=args.flip_horizontal,
            epochs=args.epochs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
