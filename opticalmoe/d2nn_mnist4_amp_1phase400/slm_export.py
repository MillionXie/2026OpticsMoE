from pathlib import Path

import torch

from utils import BASE_DIR, save_json

OPTICALMOE_ROOT = BASE_DIR.parent
import sys

if str(OPTICALMOE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTICALMOE_ROOT))

from slm_bmp import export_plane_bmp


@torch.no_grad()
def export_best_checkpoint_slm_package(model, loader, checkpoint_path, output_dir, config, device, class_names):
    cfg = config.get("slm_export", {})
    if not bool(cfg.get("enabled", True)):
        return None
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    selected = None
    fallback = None
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        for index in range(len(images)):
            score = float(logits[index, labels[index]].item())
            candidate = (score, images[index : index + 1].detach(), int(labels[index]), int(preds[index]))
            if fallback is None or score > fallback[0]:
                fallback = candidate
            if preds[index] == labels[index] and (selected is None or score > selected[0]):
                selected = candidate
    selected = selected or fallback
    if selected is None:
        raise RuntimeError("Cannot export SLM BMPs from an empty test loader.")
    score, image, true_label, pred_label = selected
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    scale = int(cfg.get("scale_factor", 2))
    width = int(cfg.get("slm_width", 1920))
    height = int(cfg.get("slm_height", 1200))
    canvas_input = model.prepare_canvas_input(image).abs()[0]
    files = [
        export_plane_bmp(canvas_input, export_dir / "input_amplitude.bmp", "amplitude", scale, width, height)
    ]
    for index, phase in enumerate(model.phase_stack_wrapped(), start=1):
        files.append(export_plane_bmp(phase, export_dir / f"phase_layer_{index:02d}.bmp", "phase", scale, width, height))
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "true_label": true_label,
        "true_name": class_names[true_label],
        "pred_label": pred_label,
        "pred_name": class_names[pred_label],
        "selection_score": score,
        "source_pixel_size_um": float(cfg.get("source_pixel_size_um", 16.0)),
        "slm_pixel_size_um": float(cfg.get("slm_pixel_size_um", 8.0)),
        "files": files,
    }
    save_json(metadata, export_dir / "manifest.json")
    return metadata
