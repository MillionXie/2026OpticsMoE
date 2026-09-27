"""Same-group, same-physics weight warm start with a fresh low-LR optimizer."""
def configure(raw, *, epochs=30, factor=0.1):
    import copy
    if epochs <= 0 or not 0 < factor < 1:
        raise ValueError("Fine-tuning requires positive epochs and LR factor in (0,1)")
    raw = copy.deepcopy(raw)
    raw["training"]["epochs"] = epochs
    for key in ("learning_rate", "phase_learning_rate", "router_phase_learning_rate"):
        raw["training"][key] *= factor
    return raw


def validate_parent(saved, group, protocol):
    if saved.get("study_group") != group or "state_dict" not in saved:
        raise ValueError("Fine-tune parent must be this group's real checkpoint")
    previous = saved.get("study_protocol", {})
    keys = ("schema_version", "selection_policy", "expected_train_count", "expected_test_count",
            "device_pitch_um", "deployment_eta", "dc_train_range", "bounded_amplitude",
            "phase_dropout_p", "router_noise_std", "ccd_model")
    if any(previous.get(key) != protocol.get(key) for key in keys):
        raise ValueError("Fine-tune parent physical/data/augmentation protocol mismatch")
