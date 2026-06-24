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
    report = {
        "model_type": config.get("model", {}).get("type"),
        "num_experts": config.get("model", {}).get("num_experts"),
        "readout_type": config.get("readout", {}).get("type"),
        "readout_dropout_is_electronic": True,
        "phase_dropout_config": config.get("regularization", {}).get("phase_dropout", {}),
        "optical_parameter_count": int(model.optical_parameter_count()),
        "prompt_parameter_count": int(model.prompt_parameter_count()),
        "electronic_parameter_count": int(model.electronic_parameter_count()),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    save_json(report, run_dir / "architecture_report.json")
    lines = [
        "# Architecture Report",
        "",
        f"- model_type: {report['model_type']}",
        f"- num_experts: {report['num_experts']}",
        f"- readout_type: {report['readout_type']}",
        "- readout.dropout is electronic dropout only.",
        "- regularization.phase_dropout is optical phase-layer dropout.",
        f"- optical_parameter_count: {report['optical_parameter_count']}",
        f"- prompt_parameter_count: {report['prompt_parameter_count']}",
        f"- electronic_parameter_count: {report['electronic_parameter_count']}",
    ]
    write_text(run_dir / "architecture_report.md", "\n".join(lines) + "\n")
    return report

