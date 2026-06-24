from typing import Dict


def phase_dropout_settings(config: Dict) -> Dict:
    cfg = config.get("regularization", {}).get("phase_dropout", {}) or {}
    enabled = bool(cfg.get("enabled", False))
    apply_to_experts = bool(cfg.get("apply_to_experts", True))
    apply_to_global_fc = bool(cfg.get("apply_to_global_fc", False))
    mode = cfg.get("mode", "none") if enabled else "none"
    return {
        "enabled": enabled,
        "mode": mode,
        "expert_mode": mode if enabled and apply_to_experts else "none",
        "global_fc_mode": mode if enabled and apply_to_global_fc else "none",
        "expert_p": float(cfg.get("expert_p", 0.0)) if enabled and apply_to_experts else 0.0,
        "global_fc_p": float(cfg.get("global_fc_p", 0.0)) if enabled and apply_to_global_fc else 0.0,
        "block_size": int(cfg.get("block_size", 8)),
        "batch_shared": bool(cfg.get("batch_shared", True)),
        "apply_to_experts": apply_to_experts,
        "apply_to_global_fc": apply_to_global_fc,
        "start_epoch": int(cfg.get("start_epoch", 0)),
    }


def phase_dropout_active_for_epoch(settings: Dict, epoch: int) -> bool:
    return bool(
        settings["enabled"]
        and settings["mode"] != "none"
        and (settings["expert_p"] > 0.0 or settings["global_fc_p"] > 0.0)
        and int(epoch) >= int(settings["start_epoch"])
    )

