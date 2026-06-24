from pathlib import Path
from typing import Dict

from ..utils.config import save_json, save_yaml
from ..utils.filesystem import write_text
from ..utils.git_info import collect_environment, collect_git_info


def save_run_manifest(run_dir: Path, config: Dict, command: str, repo_root: Path) -> Dict:
    git_info = collect_git_info(repo_root)
    env = collect_environment()
    save_yaml(config, run_dir / "config.yaml")
    save_json(config, run_dir / "config_resolved.json")
    save_json(git_info, run_dir / "git_info.json")
    save_json(env, run_dir / "environment.json")
    write_text(run_dir / "command.txt", command)
    return {"git": git_info, "environment": env}


def architecture_report(model, config: Dict, run_dir: Path) -> Dict:
    model_cfg = config.get("model", {})
    readout_cfg = config.get("readout", {})
    phase_dropout = config.get("regularization", {}).get("phase_dropout", {})
    model_type = model_cfg.get("type")
    report = {
        "model_type": model_type,
        "readout_type": readout_cfg.get("type"),
        "readout_dropout_is_electronic": True,
        "phase_dropout_config": phase_dropout,
        "optical_parameter_count": int(model.optical_parameter_count()),
        "prompt_parameter_count": int(model.prompt_parameter_count()),
        "electronic_parameter_count": int(model.electronic_parameter_count()),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
    }

    if model_type in {"learnable_route_moe", "fixed_route_moe"}:
        layout = getattr(model, "layout", None)
        report.update(
            {
                "num_experts": getattr(layout, "num_experts", model_cfg.get("num_experts")),
                "expert_size": getattr(layout, "expert_size", model_cfg.get("expert_size")),
                "expert_pitch": getattr(layout, "expert_pitch", model_cfg.get("expert_pitch")),
                "canvas_size": getattr(layout, "canvas_size", model_cfg.get("canvas_size")),
                "input_size": getattr(layout, "input_size", model_cfg.get("input_size")),
                "prompt_aperture_size": getattr(layout, "prompt_aperture_size", model_cfg.get("prompt_aperture_size")),
                "prompt_type": model_cfg.get("prompt_type"),
                "routing_type": model_cfg.get("routing_type"),
                "prompt_train_amplitudes": bool(config.get("prompt", {}).get("train_amplitudes", model_type == "learnable_route_moe")),
                "prompt_train_phase_biases": bool(config.get("prompt", {}).get("train_phase_biases", model_type == "learnable_route_moe")),
            }
        )
    elif model_type == "general_d2nn":
        canvas_shape = getattr(model, "canvas_shape", None)
        canvas_size = canvas_shape[0] if canvas_shape else model_cfg.get("canvas_size")
        d2nn_layers = getattr(model, "num_layers", model_cfg.get("d2nn_num_layers", model_cfg.get("num_layers")))
        report.update(
            {
                "input_size": getattr(model, "input_size", model_cfg.get("input_size")),
                "canvas_size": canvas_size,
                "d2nn_phase_grid_size": getattr(model, "d2nn_phase_grid_size", model_cfg.get("d2nn_phase_grid_size")),
                "d2nn_num_layers": d2nn_layers,
                "d2nn_local_phase_params": int(model.d2nn_local_phase_parameter_count()),
                "d2nn_global_fc_params": int(model.d2nn_global_fc_parameter_count()),
                "global_fc_is_used": True,
                "global_fc_is_full_canvas": True,
                "d2nn_baseline_definition": f"{d2nn_layers} local D2NN phase masks + one full-canvas global phase mask",
                "target_param_count_note": "target_param_count is kept for backward compatibility and refers to local D2NN phase masks only.",
            }
        )
    elif model_type == "lenet5":
        report.update(
            {
                "input_size": getattr(model, "input_size", config.get("dataset", {}).get("input_size", model_cfg.get("input_size"))),
                "optical_parameter_count": 0,
                "prompt_parameter_count": 0,
                "electronic_parameter_count": int(model.electronic_parameter_count()),
                "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
                "readout_type": "electronic",
                "phase_dropout_config": "ignored",
            }
        )
    save_json(report, run_dir / "architecture_report.json")
    lines = [
        "# Architecture Report",
        "",
        f"- model_type: {report['model_type']}",
        f"- readout_type: {report['readout_type']}",
        "- readout.dropout is electronic dropout only.",
        "- regularization.phase_dropout is optical phase-layer dropout.",
        f"- optical_parameter_count: {report['optical_parameter_count']}",
        f"- prompt_parameter_count: {report['prompt_parameter_count']}",
        f"- electronic_parameter_count: {report['electronic_parameter_count']}",
    ]
    if model_type in {"learnable_route_moe", "fixed_route_moe"}:
        lines.extend(
            [
                f"- num_experts: {report.get('num_experts')}",
                f"- expert_size: {report.get('expert_size')}",
                f"- expert_pitch: {report.get('expert_pitch')}",
                f"- canvas_size: {report.get('canvas_size')}",
            ]
        )
    elif model_type == "general_d2nn":
        lines.extend(
            [
                "",
                "## General D2NN Accounting",
                "",
                f"- General D2NN baseline includes {report['d2nn_num_layers']} parameter-matched center-window phase masks and one full-canvas global phase mask.",
                "- The configured target_param_count refers only to the 5 local D2NN phase masks unless otherwise specified.",
                "- The actual optical parameter count also includes the full-canvas global mask.",
                f"- d2nn_phase_grid_size: {report['d2nn_phase_grid_size']}",
                f"- d2nn_num_layers: {report['d2nn_num_layers']}",
                f"- d2nn_local_phase_params: {report['d2nn_local_phase_params']}",
                f"- d2nn_global_fc_params: {report['d2nn_global_fc_params']}",
                f"- global_fc_is_used: {report['global_fc_is_used']}",
                f"- global_fc_is_full_canvas: {report['global_fc_is_full_canvas']}",
            ]
        )
    elif model_type == "lenet5":
        lines.extend(["", "## LeNet-5", "", "- LeNet-5 is an electronic baseline and phase dropout is ignored."])
    write_text(run_dir / "architecture_report.md", "\n".join(lines) + "\n")
    return report
