from typing import Dict

import torch

from common.optics.electronic_models import LeNet5Classifier
from common.optics.expert_layout import ExpertLayout
from common.optics.optical_models import ASGlobalRouterMoEClassifier, GeneralD2NNClassifier
from common.training.phase_dropout import phase_dropout_settings


def _layout_from_config(config: Dict) -> ExpertLayout:
    model_cfg = config.get("model", {})
    return ExpertLayout(
        num_experts=int(model_cfg.get("num_experts", 9)),
        canvas_size=int(model_cfg.get("canvas_size", 1000)),
        input_size=int(model_cfg.get("input_size", 134)),
        expert_size=int(model_cfg.get("expert_size", 134)),
        expert_pitch=int(model_cfg.get("expert_pitch", 200)),
        padding=int(model_cfg.get("padding", 200)),
        prompt_aperture_size=int(model_cfg.get("prompt_aperture_size", 600)),
    )


def build_model(config: Dict, num_classes: int):
    model_cfg = config.get("model", {})
    optics_cfg = config.get("optics", {})
    prompt_cfg = config.get("prompt", {})
    detector_cfg = config.get("detector", {})
    readout_cfg = config.get("readout", {})
    dropout_cfg = phase_dropout_settings(config)
    model_type = str(model_cfg.get("type", "learnable_route_moe")).lower()

    if model_type == "lenet5":
        input_size = int(config.get("dataset", {}).get("input_size", model_cfg.get("input_size", 134)))
        return LeNet5Classifier(num_classes=num_classes, input_size=input_size)

    if model_type == "general_d2nn":
        return GeneralD2NNClassifier(
            num_classes=num_classes,
            canvas_size=int(model_cfg.get("canvas_size", 1000)),
            input_size=int(model_cfg.get("input_size", 134)),
            d2nn_phase_grid_size=int(model_cfg.get("d2nn_phase_grid_size", 402)),
            num_layers=int(model_cfg.get("d2nn_num_layers", model_cfg.get("num_layers", 5))),
            wavelength_m=float(optics_cfg.get("wavelength_m", 532e-9)),
            pixel_size_m=float(optics_cfg.get("pixel_size_m", 8e-6)),
            distances_m=optics_cfg.get("distances_m", {}),
            phase_param=optics_cfg.get("phase_param", "unconstrained"),
            phase_init=optics_cfg.get("expert_phase_init", "identity"),
            init_std=float(optics_cfg.get("expert_init_std", 0.02)),
            global_fc_phase_mode=optics_cfg.get("global_fc_phase_mode", "center_window"),
            global_fc_phase_size=optics_cfg.get("global_fc_phase_size"),
            global_fc_padding_mode=optics_cfg.get("global_fc_padding_mode", "transparent"),
            detector_size=int(detector_cfg.get("detector_size", 32)),
            detector_layout=detector_cfg.get("layout", "grid"),
            normalize_detector_energy=bool(readout_cfg.get("normalize_detector_energy", True)),
            readout_type=readout_cfg.get("type", "mlp"),
            logit_scale=float(readout_cfg.get("logit_scale", 10.0)),
            readout_hidden_dim=int(readout_cfg.get("hidden_dim", 64)),
            readout_activation=readout_cfg.get("activation", "gelu"),
            readout_input_norm=readout_cfg.get("input_norm", "layernorm"),
            readout_norm_affine=bool(readout_cfg.get("norm_affine", True)),
            readout_hidden_layers=int(readout_cfg.get("hidden_layers", 1)),
            readout_dropout=float(readout_cfg.get("dropout", 0.1)),
            phase_dropout_mode=dropout_cfg["expert_mode"],
            phase_dropout_p=dropout_cfg["expert_p"],
            global_fc_phase_dropout_mode=dropout_cfg["global_fc_mode"],
            global_fc_phase_dropout_p=dropout_cfg["global_fc_p"],
            phase_dropout_block_size=dropout_cfg["block_size"],
            phase_dropout_batch_shared=dropout_cfg["batch_shared"],
            evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
        )

    if model_type not in {"fixed_route_moe", "learnable_route_moe"}:
        raise ValueError(f"Unsupported model.type: {model_type}")

    layout = _layout_from_config(config)
    fixed = model_type == "fixed_route_moe"
    return ASGlobalRouterMoEClassifier(
        num_classes=num_classes,
        layout=layout,
        wavelength_m=float(optics_cfg.get("wavelength_m", 532e-9)),
        pixel_size_m=float(optics_cfg.get("pixel_size_m", 8e-6)),
        num_layers=int(model_cfg.get("num_layers", optics_cfg.get("num_layers", 5))),
        distances_m=optics_cfg.get("distances_m", {}),
        focal_length_m=float(optics_cfg.get("focal_length_m", 0.10)),
        aperture_mode=optics_cfg.get("aperture_mode", "hard"),
        phase_param=optics_cfg.get("phase_param", "unconstrained"),
        expert_phase_init=optics_cfg.get("expert_phase_init", "identity"),
        expert_init_std=float(optics_cfg.get("expert_init_std", 0.02)),
        global_fc_phase_init=optics_cfg.get("global_fc_phase_init", "identity"),
        global_fc_init_std=float(optics_cfg.get("global_fc_init_std", 0.02)),
        global_fc_phase_mode=optics_cfg.get("global_fc_phase_mode", "center_window"),
        global_fc_phase_size=optics_cfg.get("global_fc_phase_size", layout.active_window_size),
        global_fc_padding_mode=optics_cfg.get("global_fc_padding_mode", "transparent"),
        prompt_mode=prompt_cfg.get("mode", "complex_order_router"),
        prompt_amplitude_init_logits=float(prompt_cfg.get("amplitude_init_logits", 2.0)),
        train_prompt_amplitudes=bool(prompt_cfg.get("train_amplitudes", not fixed)) and not fixed,
        train_prompt_phase_biases=bool(prompt_cfg.get("train_phase_biases", not fixed)) and not fixed,
        grating_scale=float(prompt_cfg.get("grating_scale", 1.0)),
        grating_sign_x=float(prompt_cfg.get("grating_sign_x", 1.0)),
        grating_sign_y=float(prompt_cfg.get("grating_sign_y", 1.0)),
        prompt_normalize=prompt_cfg.get("normalize", "sum_amplitude"),
        detector_size=int(detector_cfg.get("detector_size", 32)),
        detector_layout=detector_cfg.get("layout", "grid"),
        normalize_detector_energy=bool(readout_cfg.get("normalize_detector_energy", True)),
        readout_type=readout_cfg.get("type", "mlp"),
        logit_scale=float(readout_cfg.get("logit_scale", 10.0)),
        readout_hidden_dim=int(readout_cfg.get("hidden_dim", 64)),
        readout_activation=readout_cfg.get("activation", "gelu"),
        readout_input_norm=readout_cfg.get("input_norm", "layernorm"),
        readout_norm_affine=bool(readout_cfg.get("norm_affine", True)),
        readout_hidden_layers=int(readout_cfg.get("hidden_layers", 1)),
        readout_dropout=float(readout_cfg.get("dropout", 0.1)),
        expert_phase_dropout_mode=dropout_cfg["expert_mode"],
        expert_phase_dropout_p=dropout_cfg["expert_p"],
        global_fc_phase_dropout_mode=dropout_cfg["global_fc_mode"],
        global_fc_phase_dropout_p=dropout_cfg["global_fc_p"],
        phase_dropout_block_size=dropout_cfg["block_size"],
        phase_dropout_batch_shared=dropout_cfg["batch_shared"],
        evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
    )


def build_optimizer(model: torch.nn.Module, config: Dict):
    cfg = config.get("optimizer", {})
    opt_type = str(cfg.get("type", "adamw")).lower()
    lr = float(cfg.get("lr", 0.001))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=float(cfg.get("momentum", 0.9)))
    raise ValueError(f"Unsupported optimizer.type: {opt_type}")
