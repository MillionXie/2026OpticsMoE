"""Export four fixed T08 test inputs and simulated router CCDs for optical bring-up.

Run from the exact server worktree with CUDA. This does not control hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs, preprocess_images, student_embeddings, validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import GroceryRetrievalDataset
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import load_checkpoint
from LightGenV2.tasks.t01_object_retrieval.modeling import build_student, load_backbone
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import load_contract


EXPECTED = "cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae"
INSTRUCTION = "Represent the user's input."


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha(a.checkpoint) != EXPECTED:
        raise RuntimeError("Wrong checkpoint")
    settings = load_settings(a.config)
    contract = load_contract(a.data_root)
    # The test CSV is product-grouped; adjacent rows would all be one SKU.
    samples = tuple(contract.test[i] for i in (0, 24, 1200, 2376))
    if len(samples) != 4:
        raise RuntimeError("Expected four pilot samples")
    loaded = load_backbone(settings, torch.device("cuda"))
    replacement, readout = build_student(loaded, settings)
    try:
        load_checkpoint(a.checkpoint, replacement, readout)
        replacement.use_student()
        replacement.set_phase_dropout_active(False)
        replacement.vision_surrogate.eval()
        replacement.language_surrogate.eval()
        readout.eval()
        dataset = GroceryRetrievalDataset(samples, settings.image_size, augment=False)
        images = [dataset[i]["image"] for i in range(4)]
        inputs = preprocess_images(loaded.processor, images, INSTRUCTION)
        validate_token_budgets(inputs, settings)
        inputs = move_inputs(inputs, loaded.device)
        branch = replacement.vision_surrogate.core.optical_branch
        branch.core.capture_intermediate_fields = True
        branch.core.capture_sample_count = 4
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=settings.amp_enabled):
            student_embeddings(loaded.model, replacement, readout, inputs)
        amplitudes = branch.core.last_input_fields.detach().float().numpy()
        simulated = branch.core.router.last_detector_intensity.detach().float().cpu().numpy()
        if amplitudes.shape != (4, 224, 224) or simulated.shape != (4, 478, 478):
            raise RuntimeError(f"Unexpected router tensors: {amplitudes.shape}, {simulated.shape}")
        a.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for i, (sample, amplitude, ccd) in enumerate(zip(samples, amplitudes, simulated)):
            scale = float(np.percentile(amplitude[amplitude > 0], 99.5))
            gray = np.rint(np.clip(amplitude / max(scale, 1e-8), 0, 1) * 255).astype(np.uint8)
            native = np.zeros((1080, 1920), dtype=np.uint8)
            # 224 logical pixels at 17 um -> 476 native pixels at 8 um.
            size = round(224 * 17 / 8)
            positions = (np.arange(size) + .5 - size / 2) * 8
            idx = np.floor(positions / 17 + 224 / 2).astype(int).clip(0, 223)
            square = gray[np.ix_(idx, idx)]
            y, x = (1080 - size) // 2, (1920 - size) // 2
            native[y:y + size, x:x + size] = square
            path = a.output / f"amplitude_{i:02d}.bmp"
            Image.fromarray(native).save(path)
            np.save(a.output / f"sim_router_{i:02d}.npy", ccd)
            rows.append({"index": i, "sample_id": sample.sample_id, "product_id": sample.sku_name,
                         "amplitude_bmp": path.name, "amplitude_sha256": sha(path),
                         "source_image": str(sample.image_path), "amplitude_scale_p995": scale,
                         "sim_ccd_sum": float(ccd.sum())})
        (a.output / "manifest.json").write_text(json.dumps({
            "schema": 1, "checkpoint_sha256": EXPECTED, "instruction": INSTRUCTION,
            "propagation_distance_m": .1, "wavelength_nm": 532,
            "logical_pitch_um": 17, "native_pitch_um": 8, "rows": rows,
        }, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(a.output), "rows": rows}, indent=2), flush=True)
    finally:
        replacement.close()


if __name__ == "__main__":
    main()
