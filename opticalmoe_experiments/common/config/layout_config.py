from copy import deepcopy
from typing import Dict, Mapping

from common.optics.expert_layout import ExpertLayout


GEOMETRY_PROFILES = {
    "fast120_520": {
        "canvas_size": 520,
        "input_size": 120,
        "expert_size": 120,
        "expert_pitch": 150,
        "padding": 35,
        "prompt_aperture_size": 450,
    },
    "fair134_1000": {
        "canvas_size": 1000,
        "input_size": 134,
        "expert_size": 134,
        "expert_pitch": 200,
        "padding": 200,
        "prompt_aperture_size": 600,
    },
}


def _first(mapping: Mapping, *keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def layout_from_config(config: Dict, default_profile: str = "fast120_520") -> ExpertLayout:
    """Resolve layout fields with layout > legacy model > profile precedence."""
    layout_cfg = dict(config.get("layout", {}) or {})
    model_cfg = dict(config.get("model", {}) or {})
    profile_name = str(
        _first(layout_cfg, "geometry_profile", default=_first(model_cfg, "geometry_profile", default=default_profile))
    )
    if profile_name not in GEOMETRY_PROFILES:
        raise ValueError(
            f"Unknown geometry_profile={profile_name!r}; expected one of {sorted(GEOMETRY_PROFILES)}."
        )
    resolved = deepcopy(GEOMETRY_PROFILES[profile_name])

    legacy_values = {
        "canvas_size": _first(model_cfg, "canvas_size", "canvas_height"),
        "input_size": _first(model_cfg, "input_size"),
        "expert_size": _first(model_cfg, "expert_size"),
        "expert_pitch": _first(model_cfg, "expert_pitch"),
        "padding": _first(model_cfg, "padding"),
        "prompt_aperture_size": _first(model_cfg, "prompt_aperture_size"),
    }
    resolved.update({key: int(value) for key, value in legacy_values.items() if value is not None})

    canvas_height = _first(layout_cfg, "canvas_height", "canvas_size")
    canvas_width = _first(layout_cfg, "canvas_width", default=canvas_height)
    if canvas_height is not None:
        resolved["canvas_size"] = int(canvas_height)
    if canvas_width is not None and int(canvas_width) != int(resolved["canvas_size"]):
        raise ValueError(
            f"ExpertLayout currently requires a square canvas, got height={resolved['canvas_size']} width={canvas_width}."
        )
    for key in ("input_size", "expert_size", "expert_pitch", "padding", "prompt_aperture_size"):
        if layout_cfg.get(key) is not None:
            resolved[key] = int(layout_cfg[key])

    explicit_profile = "geometry_profile" in layout_cfg or "geometry_profile" in model_cfg
    if not explicit_profile:
        legacy = GEOMETRY_PROFILES["fair134_1000"]
        if all(int(resolved[key]) == int(value) for key, value in legacy.items()):
            profile_name = "fair134_1000"

    layout = ExpertLayout(
        num_experts=int(_first(layout_cfg, "num_experts", default=_first(model_cfg, "num_experts", default=9))),
        geometry_profile=profile_name,
        **resolved,
    )
    layout.validate()
    return layout

