"""Export high-confidence one-shot amplitude/global-phase BMP pairs.

The exported amplitude is the zero-phase OEO reload immediately before the
shared global phase.  No propagation exists between those two tensors in the
model.  Only the global phase to detector propagation remains (20 cm in the
target experiment).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import create_loaders
from model import OpticalMoEClassifier
from utils import BASE_DIR, choose_device, save_json, set_seed

OPTICALMOE_ROOT = BASE_DIR.parent
if str(OPTICALMOE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTICALMOE_ROOT))
from slm_bmp import encode_amplitude_uint8, encode_phase_uint8, export_plane_bmp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export correctly classified one-shot BMP inputs from the final OEO amplitude plane."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_exact_checkpoint(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in payload:
        raise RuntimeError("Checkpoint has no embedded config; exact architecture reconstruction is impossible.")
    config = payload["config"]
    class_indices = config.get("dataset", {}).get("class_indices", [0, 1, 2, 3])
    model = OpticalMoEClassifier(config, len(class_indices))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return payload, config, model


@torch.no_grad()
def _rank_correct_samples(model, dataset, batch_size, num_workers, device, samples_per_class, multiplier):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    candidates = defaultdict(list)
    offset = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(1)
        for batch_index in range(len(labels)):
            label = int(labels[batch_index])
            pred = int(preds[batch_index])
            if pred != label:
                continue
            values = logits[batch_index].detach().float().cpu()
            true_score = float(values[label])
            other_score = float(torch.cat((values[:label], values[label + 1 :])).max())
            candidates[label].append(
                {
                    "dataset_index": offset + batch_index,
                    "true_score": true_score,
                    "margin": true_score - other_score,
                    "float_logits": values.tolist(),
                }
            )
        offset += len(labels)
    keep = max(samples_per_class, samples_per_class * multiplier)
    return {
        label: sorted(rows, key=lambda row: (row["margin"], row["true_score"]), reverse=True)[:keep]
        for label, rows in candidates.items()
    }


def _active_amplitude(items, active):
    # ``at_global_fc`` is already zero phase in the OEO model.  Use abs() to
    # make the physical amplitude contract explicit and crop the 450x450 plane.
    value = items["at_global_fc"][0].detach().abs().float()
    value = value[active.y0 : active.y1, active.x0 : active.x1]
    peak = float(value.max())
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError(f"Invalid pre-global amplitude peak: {peak}")
    # A global positive scale does not change normalized detector energies.
    return value / peak, peak


@torch.no_grad()
def _quantized_replay(model, amplitude_active, phase_active, device):
    amplitude_q = encode_amplitude_uint8(amplitude_active).float().to(device) / 255.0
    phase_u8 = encode_phase_uint8(phase_active).to(device)
    phase_q = phase_u8.float() * ((2.0 * math.pi) / 255.0)
    aperture = model.layout.active_aperture
    field = torch.zeros(
        1, model.layout.canvas_size, model.layout.canvas_size, dtype=torch.complex64, device=device
    )
    modulated = amplitude_q.to(torch.complex64) * torch.exp(1j * phase_q).to(torch.complex64)
    field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = modulated
    detector_field = model.to_detector(field)
    logits = model.detector(detector_field)[0].detach().float().cpu()
    intensity = detector_field.abs().square()[0].detach().float().cpu()
    return logits, intensity


def _save_preview(image, path):
    value = image.detach().cpu().float()
    while value.ndim > 2:
        value = value[0]
    value = value.clamp(0, 1)
    Image.fromarray(torch.round(value * 255).to(torch.uint8).numpy(), mode="L").save(path)


@torch.no_grad()
def export_oneshot_package(model, dataset, ranked, class_names, output_dir, checkpoint_path, payload, config, device, samples_per_class):
    output_dir = Path(output_dir)
    samples_root = output_dir / "samples"
    diagnostics_root = output_dir / "diagnostics"
    samples_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    active = model.layout.active_aperture
    phase_active = model.global_fc.get_phase().detach().cpu()

    shared_phase_path = output_dir / "00_shared_global_phase_active450_scale2_1920x1200.bmp"
    phase_file = export_plane_bmp(
        phase_active, shared_phase_path, "phase", scale_factor=2, slm_width=1920, slm_height=1200
    )

    rows = []
    sequence_number = 0
    for label, class_name in enumerate(class_names):
        accepted = 0
        for candidate in ranked.get(label, []):
            if accepted >= samples_per_class:
                break
            dataset_index = int(candidate["dataset_index"])
            image, true_label = dataset[dataset_index]
            image_batch = image.unsqueeze(0).to(device)
            float_logits, items = model(image_batch, return_intermediates=True, capture_layer_fields=True)
            float_logits = float_logits[0].detach().float().cpu()
            amplitude_active, amplitude_peak = _active_amplitude(items, active)
            replay_logits, replay_intensity = _quantized_replay(model, amplitude_active, phase_active, device)
            replay_pred = int(replay_logits.argmax())
            if replay_pred != int(true_label):
                continue

            source_index = int(dataset.indices[dataset_index]) if hasattr(dataset, "indices") else dataset_index
            safe_margin = f"{candidate['margin']:.6f}".replace("-", "m").replace(".", "p")
            stem = (
                f"{sequence_number:04d}_{class_name}_subset{dataset_index:05d}_"
                f"cifar{source_index:05d}_margin{safe_margin}"
            )
            class_dir = samples_root / f"class_{label}_{class_name}"
            class_dir.mkdir(parents=True, exist_ok=True)
            amplitude_path = class_dir / f"{stem}_amplitude_before_global_1920x1080.bmp"
            amplitude_file = export_plane_bmp(
                amplitude_active,
                amplitude_path,
                "amplitude",
                scale_factor=2,
                slm_width=1920,
                slm_height=1080,
            )

            preview_dir = diagnostics_root / stem
            preview_dir.mkdir(parents=True, exist_ok=True)
            _save_preview(image, preview_dir / "input_grayscale_120.png")
            _save_preview(amplitude_active, preview_dir / "amplitude_before_global_active450.png")
            detector_max = float(replay_intensity.max())
            detector_preview = replay_intensity / detector_max if detector_max > 0 else replay_intensity
            _save_preview(detector_preview, preview_dir / "quantized_replay_detector_intensity_480.png")

            routing_indices = [int(v) for v in items["routing_selected_indices"][0].detach().cpu()]
            routing_weights = [float(v) for v in items["routing_weights"][0].detach().cpu()]
            row = {
                "sequence": sequence_number,
                "split_subset_index": dataset_index,
                "cifar10_source_index": source_index,
                "true_label": int(true_label),
                "true_name": class_name,
                "float_pred_label": int(float_logits.argmax()),
                "float_pred_name": class_names[int(float_logits.argmax())],
                "quantized_replay_pred_label": replay_pred,
                "quantized_replay_pred_name": class_names[replay_pred],
                "float_true_score": float(float_logits[int(true_label)]),
                "selection_margin": float(candidate["margin"]),
                "amplitude_peak_before_unit_normalization": amplitude_peak,
                "amplitude_bmp": str(amplitude_path.relative_to(output_dir)),
                "global_phase_bmp": str(shared_phase_path.relative_to(output_dir)),
                "routing_topk": json.dumps(routing_indices),
                "routing_weights": json.dumps(routing_weights),
                "float_detector_energies": json.dumps([float(v) for v in float_logits]),
                "quantized_replay_detector_energies": json.dumps([float(v) for v in replay_logits]),
                "diagnostics_dir": str(preview_dir.relative_to(output_dir)),
            }
            rows.append(row)
            save_json(
                {
                    **row,
                    "amplitude_export": amplitude_file,
                    "global_phase_export": phase_file,
                    "amplitude_encoding_note": "Per-sample positive scalar normalization by peak, then uint8 round(amplitude*255).",
                    "quantized_replay_correct": True,
                },
                preview_dir / "metadata.json",
            )
            sequence_number += 1
            accepted += 1

    _write_csv(output_dir / "manifest.csv", rows)
    detector_bounds = []
    for index, mask in enumerate(model.detector.masks.detach().cpu()):
        coords = mask.nonzero()
        detector_bounds.append(
            {
                "class_index": index,
                "class_name": class_names[index],
                "y0": int(coords[:, 0].min()),
                "y1_exclusive": int(coords[:, 0].max()) + 1,
                "x0": int(coords[:, 1].min()),
                "x1_exclusive": int(coords[:, 1].max()) + 1,
            }
        )
    counts = {name: sum(row["true_name"] == name for row in rows) for name in class_names}
    source_center = model.layout.canvas_size / 2.0
    physical_detector_regions = []
    for bounds in detector_bounds:
        center_y = (bounds["y0"] + bounds["y1_exclusive"]) / 2.0
        center_x = (bounds["x0"] + bounds["x1_exclusive"]) / 2.0
        offset_y_um = (center_y - source_center) * 16.0
        offset_x_um = (center_x - source_center) * 16.0
        entry = {
            **bounds,
            "center_offset_from_optical_axis_um_yx": [offset_y_um, offset_x_um],
            "region_size_um_hw": [800.0, 800.0],
        }
        for width, height in ((1920, 1080), (1920, 1200)):
            target_center_y = height / 2.0 + offset_y_um / 8.0
            target_center_x = width / 2.0 + offset_x_um / 8.0
            half_size = 400.0 / 8.0
            entry[f"bounds_on_centered_{width}x{height}_8um_y0_y1_x0_x1"] = [
                int(round(target_center_y - half_size)),
                int(round(target_center_y + half_size)),
                int(round(target_center_x - half_size)),
                int(round(target_center_x + half_size)),
            ]
        physical_detector_regions.append(entry)
    manifest = {
        "purpose": "One-shot physical validation of final OEO amplitude + co-planar global phase + 20 cm detector.",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "split": "dataset recorded in manifest.csv",
        "class_names": class_names,
        "requested_samples_per_class": samples_per_class,
        "exported_samples_per_class": counts,
        "exported_total": len(rows),
        "source_plane": "at_global_fc: fifth expert phase -> 20 cm -> square detection -> LayerNorm -> ReLU -> zero-phase amplitude reload",
        "coplanar_contract": "The amplitude is multiplied immediately by global_fc phase; no simulated propagation exists between them.",
        "remaining_path": "global phase modulation -> 20 cm angular-spectrum propagation -> square-law four-region detector",
        "amplitude_device": {"size_wh": [1920, 1080], "pixel_size_um": 8.0, "active_shape_hw": [900, 900], "center_padding_lrtb": [510, 510, 90, 90]},
        "phase_device": {"size_wh": [1920, 1200], "pixel_size_um": 8.0, "active_shape_hw": [900, 900], "center_padding_lrtb": [510, 510, 150, 150]},
        "source_simulation": {"active_shape_hw": [450, 450], "pixel_size_um": 16.0, "wavelength_nm": 532.0},
        "global_phase_file": phase_file,
        "global_phase_to_detector_distance_m": float(config["optics"]["distances_m"]["global_fc_to_detector"]),
        "detector_bounds_on_480_simulation_grid": detector_bounds,
        "detector_regions_physical_and_8um_mappings": physical_detector_regions,
        "selection": "Correct float prediction, ranked per class by detector-energy margin, and retained only when uint8 amplitude/phase replay remains correct.",
    }
    save_json(manifest, output_dir / "manifest.json")
    (output_dir / "README.md").write_text(
        "# One-shot physical BMP package\n\n"
        "Each file under `samples/` is the amplitude immediately before the global phase. "
        "Load it on the 1920x1080, 8 um amplitude device. Load "
        "`00_shared_global_phase_active450_scale2_1920x1200.bmp` on the 1920x1200, 8 um phase device. "
        "Their 900x900 active areas are independently centered, so their physical centers coincide.\n\n"
        "The simulation has no propagation between these two planes. After co-planar modulation, propagate 20 cm "
        "to the four detector regions listed in `manifest.json`; both physical offsets and centered 8 um sensor "
        "coordinates are included. `manifest.csv` records label, confidence, routing, "
        "float detector energies, and 8-bit quantized replay detector energies for every sample.\n",
        encoding="utf-8",
    )
    return manifest


def main():
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    if args.candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be positive")
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "one_shot_last_oeo_global_phase_20cm"
    device = choose_device(args.device)
    payload, config, model = _load_exact_checkpoint(checkpoint_path, device)
    seed = int(config.get("seed", 7))
    set_seed(seed)
    config.setdefault("dataset", {})["batch_size"] = int(args.batch_size)
    config["dataset"]["num_workers"] = int(args.num_workers)
    config["dataset"]["persistent_workers"] = args.num_workers > 0
    train_loader, test_loader, class_names = create_loaders(config, seed, smoke_test=False)
    dataset = train_loader.dataset if args.split == "train" else test_loader.dataset
    ranked = _rank_correct_samples(
        model,
        dataset,
        args.batch_size,
        args.num_workers,
        device,
        args.samples_per_class,
        args.candidate_multiplier,
    )
    manifest = export_oneshot_package(
        model,
        dataset,
        ranked,
        class_names,
        output_dir,
        checkpoint_path,
        payload,
        config,
        device,
        args.samples_per_class,
    )
    manifest["split"] = args.split
    save_json(manifest, output_dir / "manifest.json")
    print(f"exported={manifest['exported_total']} per_class={manifest['exported_samples_per_class']}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
