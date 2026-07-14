"""Type-aware export for the heterogeneous expert bank.

Fourier masks are explicitly labelled as frequency-domain masks.  Fiber
experts are exported as mode-bank parameters, never as D2NN phase planes.
"""

import sys
from pathlib import Path

import torch

from utils import BASE_DIR, save_json

OPTICALMOE_ROOT = BASE_DIR.parent
if str(OPTICALMOE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTICALMOE_ROOT))
from slm_bmp import export_plane_bmp


@torch.no_grad()
def export_best_slm_package(model, loader, checkpoint_path, output_dir, config, device, class_names):
    cfg = config.get("slm_export", {})
    if not bool(cfg.get("enabled", True)):
        return None
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    selected = fallback = None
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        predictions = logits.argmax(1)
        for index in range(len(images)):
            score = float(logits[index, labels[index]])
            candidate = (score, images[index : index + 1].detach(), int(labels[index]), int(predictions[index]))
            if fallback is None or score > fallback[0]:
                fallback = candidate
            if predictions[index] == labels[index] and (selected is None or score > selected[0]):
                selected = candidate
    selected = selected or fallback
    if selected is None:
        raise RuntimeError("Cannot export from an empty loader")
    score, image, true_label, predicted_label = selected
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scale = int(cfg.get("scale_factor", 2))
    width = int(cfg.get("slm_width", 1920))
    height = int(cfg.get("slm_height", 1200))
    active = model.layout.active_aperture
    _, items = model(image, return_intermediates=True, capture_expert_outputs=True)
    input_active = model.prepare_canvas_input(image).abs()[0, active.y0 : active.y1, active.x0 : active.x1]
    prompt_amplitude = items["prompt_amplitude"][0, active.y0 : active.y1, active.x0 : active.x1]
    prompt_phase = items["prompt_phase"][0, active.y0 : active.y1, active.x0 : active.x1]
    files = {
        "common_planes": [
            export_plane_bmp(input_active, output / "input_amplitude_active450.bmp", "amplitude", scale, width, height),
            export_plane_bmp(prompt_amplitude, output / "prompt_amplitude_active450.bmp", "amplitude", scale, width, height),
            export_plane_bmp(prompt_phase, output / "prompt_phase_active450.bmp", "phase", scale, width, height),
            export_plane_bmp(items["global_fc_phase"], output / "global_fc_phase_active450.bmp", "phase", scale, width, height),
        ],
        "d2nn_phase_mosaics": [],
        "fourier_frequency_domain_masks": [],
        "fourier_tail_spatial_masks": [],
        "fiber_encoder_decoder_spatial_masks": [],
        "fiber_mode_parameter_files": [],
    }
    # One physical active-area mosaic per D2NN depth.  Only D2NN cells are
    # populated; the manifest explicitly marks all other cells as absent.
    d2nn_experts = [expert for expert in model.expert_bank.experts if expert.expert_type == "d2nn"]
    maximum_d2nn_depth = max((expert.num_layers for expert in d2nn_experts), default=0)
    for layer_index in range(maximum_d2nn_depth):
        mosaic = torch.zeros(model.layout.active_size, model.layout.active_size)
        populated = []
        for expert_index, (aperture, expert) in enumerate(zip(model.layout.expert_apertures, model.expert_bank.experts)):
            if expert.expert_type != "d2nn" or layer_index >= expert.num_layers:
                continue
            y0 = aperture.y0 - active.y0
            x0 = aperture.x0 - active.x0
            mosaic[y0 : y0 + model.layout.expert_size, x0 : x0 + model.layout.expert_size] = expert.phase_stack()[layer_index].cpu()
            populated.append(expert_index)
        filename = output / f"d2nn_phase_layer_{layer_index + 1:02d}_mosaic_active450.bmp"
        exported = export_plane_bmp(mosaic, filename, "phase", scale, width, height)
        files["d2nn_phase_mosaics"].append({"file": exported, "populated_expert_indices": populated})
    expert_manifest = []
    raw_parameters = {}
    for expert_index, expert in enumerate(model.expert_bank.experts):
        entry = {"expert_index": expert_index, **expert.parameter_summary()}
        if expert.expert_type == "d2nn":
            entry.update(
                {
                    "deployment": "phase-only local D2NN cells in d2nn_phase_layer_* mosaics",
                    "parameter_files": [value["file"] for value in files["d2nn_phase_mosaics"] if expert_index in value["populated_expert_indices"]],
                }
            )
            raw_parameters[f"expert_{expert_index:02d}_d2nn_phase_stack"] = expert.phase_stack().cpu()
        elif expert.expert_type == "fourier":
            frequency_files = []
            spatial_files = []
            for block_index, phase in enumerate(expert.frequency_phase_stack(), 1):
                filename = output / f"fourier_expert_{expert_index:02d}_frequency_phase_{block_index:02d}.bmp"
                exported = export_plane_bmp(phase.cpu(), filename, "phase", scale, width, height)
                frequency_files.append(exported)
                files["fourier_frequency_domain_masks"].append(exported)
            for layer_index, phase in enumerate(expert.spatial_phase_stack(), 1):
                filename = output / f"fourier_expert_{expert_index:02d}_tail_spatial_phase_{layer_index:02d}.bmp"
                exported = export_plane_bmp(phase.cpu(), filename, "phase", scale, width, height)
                spatial_files.append(exported)
                files["fourier_tail_spatial_masks"].append(exported)
            entry.update(
                {
                    "deployment": "three finite-aperture Fourier-plane phase masks followed by two explicitly labelled spatial mixing masks",
                    "frequency_domain_parameter_files": frequency_files,
                    "tail_spatial_parameter_files": spatial_files,
                    "parameter_files": frequency_files + spatial_files,
                    "fft_normalization": "ortho",
                    "shift_convention": "ifftshift -> fft2 -> fftshift; inverse uses ifftshift -> ifft2 -> fftshift",
                    "non_foldability": "zero-padded finite Fourier aperture, spatial center crop, and finite-aperture propagation separate consecutive masks",
                }
            )
            raw_parameters[f"expert_{expert_index:02d}_fourier_frequency_phase_stack"] = expert.frequency_phase_stack().cpu()
            raw_parameters[f"expert_{expert_index:02d}_fourier_tail_spatial_phase_stack"] = expert.spatial_phase_stack().cpu()
        else:
            spatial_files = []
            for layer_index, phase in enumerate(expert.phase_stack(), 1):
                region = "encoder" if layer_index <= expert.num_pre_layers else "decoder"
                local_index = layer_index if region == "encoder" else layer_index - expert.num_pre_layers
                filename = output / f"fiber_expert_{expert_index:02d}_{region}_spatial_phase_{local_index:02d}.bmp"
                exported = export_plane_bmp(phase.cpu(), filename, "phase", scale, width, height)
                spatial_files.append(exported)
                files["fiber_encoder_decoder_spatial_masks"].append(exported)
            parameter_file = output / f"fiber_expert_{expert_index:02d}_mode_parameters.pt"
            fiber_payload = {
                "mode_grid": expert.mode_grid,
                "mode_centers_yx": expert.mode_centers_yx.cpu(),
                "mode_sigma_pixels": expert.mode_sigma_pixels,
                "mode_phase_rad": expert.mode_phase().cpu(),
                "mode_amplitude": expert.mode_amplitude().cpu(),
                "amplitude_bounds": [expert.amplitude_min, expert.amplitude_max],
                "encoder_decoder_spatial_phase_stack": expert.phase_stack().cpu(),
            }
            torch.save(fiber_payload, parameter_file)
            files["fiber_mode_parameter_files"].append(str(parameter_file))
            entry.update(
                {
                    "deployment": "two encoder spatial phase masks, coherent Gaussian fiber-mode array, then two decoder spatial phase masks",
                    "mode_parameter_files": [str(parameter_file)],
                    "encoder_decoder_spatial_parameter_files": spatial_files,
                    "parameter_files": spatial_files + [str(parameter_file)],
                }
            )
            raw_parameters[f"expert_{expert_index:02d}_fiber"] = fiber_payload
        expert_manifest.append(entry)
    raw_parameters.update(
        {
            "input_active450": input_active.cpu(),
            "prompt_amplitude": prompt_amplitude.cpu(),
            "prompt_phase": prompt_phase.cpu(),
            "routing_weights": items["routing_weights"][0].cpu(),
            "routing_selected_indices": items["routing_selected_indices"][0].cpu(),
            "global_fc_phase": items["global_fc_phase"].cpu(),
        }
    )
    torch.save(raw_parameters, output / "raw_heterogeneous_optical_parameters.pt")
    metadata = {
        "format": "deep_heterogeneous_optical_moe9_staged_oeo_type_aware_export_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "true_label": true_label,
        "true_name": class_names[true_label],
        "pred_label": predicted_label,
        "pred_name": class_names[predicted_label],
        "selection_score": score,
        "expert_type_map_row_major": list(model.expert_bank.expert_types),
        "nonlinear_schedules": [list(expert.nonlinear_schedule) for expert in model.expert_bank.experts],
        "stage_oeo": model.nonlinearity_parameter_report(),
        "fiber_stage2_bypasses_oeo": True,
        "post_global_fc_oeo": False,
        "oeo_deployment_note": "Each enabled OEO stage uses parameter-free per-sample stage-global intensity LayerNorm, ReLU, and zero-phase amplitude re-encoding. These electronic operations are not SLM phase masks.",
        "warning": "Fourier frequency masks and Fiber mode parameters are kept type-specific; their labelled spatial encoder/decoder masks are not represented as ordinary D2NN experts.",
        "experts": expert_manifest,
        "files": files,
        "raw_parameter_archive": str(output / "raw_heterogeneous_optical_parameters.pt"),
        "routing_top_k": int(items["routing_selected_indices"].shape[1]),
        "routing_selected_indices": [int(value) for value in items["routing_selected_indices"][0].cpu()],
        "routing_weights": [float(value) for value in items["routing_weights"][0].cpu()],
        "source_pixel_size_um": float(cfg.get("source_pixel_size_um", 16.0)),
        "slm_pixel_size_um": float(cfg.get("slm_pixel_size_um", 8.0)),
        "slm_size_wh": [width, height],
    }
    save_json(metadata, output / "manifest.json")
    return metadata
