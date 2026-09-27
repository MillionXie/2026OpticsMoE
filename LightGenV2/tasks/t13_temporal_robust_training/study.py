"""Four cumulative conditions; no torch dependency for planning/preflight."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "r0_post": {"dc": False, "ccd": False, "mapping": "post_training"},
    "r1_ccd_post": {"dc": False, "ccd": True, "mapping": "post_training"},
    "r2_ccd_dc_post": {"dc": True, "ccd": True, "mapping": "post_training"},
    "r3_ccd_dc_intrain": {"dc": True, "ccd": True, "mapping": "in_training"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_protocol() -> dict:
    return json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))


def make_config(group: str, *, purpose: str = "train", seed: int = 163,
                output: Path | None = None, paths: dict | None = None,
                device_pitch: float | None = None, eval_eta: float | None = None) -> dict:
    if group not in GROUPS or purpose not in {"train", "deployment", "nominal"}:
        raise ValueError("Unknown group or purpose")
    protocol = load_protocol()
    condition = GROUPS[group]
    raw = copy.deepcopy(yaml.safe_load((ROOT / "reference" / "teacher_v2.yaml").read_text(encoding="utf-8")))
    paths = paths or {}
    for key, value in raw["data"].items():
        value = paths.get(key, value)
        raw["data"][key] = str((ROOT / value).resolve()) if value else None
    raw["random_seed"] = seed
    raw["output_dir"] = str((output or ROOT / "runs" / "simulation" / f"{group}_s{seed}").resolve())
    # Every group starts from the same random student, not a robust-trained checkpoint.
    raw["training"]["initialization_checkpoint"] = None
    raw["training"]["epochs"] = protocol["epochs"]
    raw["training"]["phase_snapshot_interval_epochs"] = 0
    # Hold unrelated augmentation fixed (off) to avoid bundling extra factors.
    for key in ("input_shift_pixels", "phase_shift_pixels", "ccd_shift_pixels"):
        raw["robustness"][key] = 0
    raw["robustness"]["phase_dropout_p"] = 0.0
    raw["router"]["noise_std"] = 0.0
    optics = raw["optics"]
    pitch = protocol["device_pitch_um"] if device_pitch is None else device_pitch
    if pitch <= 0:
        raise ValueError("Device pitch must be positive")
    optics["modulator_pixel_pitch_um"] = (
        pitch if purpose == "deployment" or condition["mapping"] == "in_training"
        else optics["pixel_pitch_um"]
    )
    if purpose == "nominal":
        optics["modulator_pixel_pitch_um"] = optics["pixel_pitch_um"]
    eta = protocol["deployment_eta"] if eval_eta is None else eval_eta
    if not 0 <= eta < 1:
        raise ValueError("Leakage eta must lie in [0,1)")
    optics["unmodulated_power_fraction_min"] = protocol["dc_train_range"][0] if condition["dc"] else 0.0
    optics["unmodulated_power_fraction_max"] = protocol["dc_train_range"][1] if condition["dc"] else 0.0
    optics["unmodulated_power_fraction_eval"] = eta if purpose == "deployment" else (eta if condition["dc"] else 0.0)
    if purpose == "nominal":
        optics["unmodulated_power_fraction_eval"] = 0.0
    if purpose != "train":
        # Fixed evaluation condition, independent of the training DC interval.
        optics["unmodulated_power_fraction_min"] = optics["unmodulated_power_fraction_eval"]
        optics["unmodulated_power_fraction_max"] = optics["unmodulated_power_fraction_eval"]
    raw["robustness"]["ccd_noise"]["enabled"] = condition["ccd"]
    raw["robustness"]["ccd_noise"]["operator_override"] = protocol["ccd_profile"]
    return raw


def asset_preflight(raw: dict) -> dict:
    required = [raw["data"][key] for key in ("manifest", "vision_cache", "language_cache")]
    if raw["loss"].get("soft_target_weight", 0) > 0:
        required.append(raw["data"]["training_soft_targets"])
    missing = [str(path) for path in required if not path or not Path(path).is_file()]
    return {"status": "blocked" if missing else "assets_present_not_identity_verified",
            "missing": missing, "assets": {str(path): sha256(Path(path)) for path in required if path and Path(path).is_file()}}


def split_for_selection(payload: dict, *, fraction: float = 0.2, seed: int = 20260927) -> tuple[dict, dict]:
    """Use deterministic TRAIN holdout as legacy evaluator's internal 'test'.

    Original test becomes 'sealed'; cached sample tensors/indices do not change.
    Hash ranking groups corresponding generated variants by relative basename.
    """
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must lie in (0,1)")
    ids = list(map(str, payload["sample_ids"]))
    if len(ids) != len(set(ids)) or len(ids) != len(payload["splits"]):
        raise ValueError("Duplicate IDs or split length mismatch")
    train = [i for i, split in enumerate(payload["splits"]) if split == "train"]
    grouped: dict[str, list[int]] = {}
    for i in train:
        key = ids[i].replace("\\", "/").split("/", 1)[-1]
        grouped.setdefault(key, []).append(i)
    keys = sorted(grouped, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    count = max(1, round(len(keys) * fraction))
    if len(keys) < 2 or count >= len(keys):
        raise ValueError("Not enough train source groups for validation")
    selected = {i for key in keys[:count] for i in grouped[key]}
    result = dict(payload)
    result["splits"] = ["test" if i in selected else ("train" if split == "train" else "sealed")
                        for i, split in enumerate(payload["splits"])]
    manifest = {"seed": seed, "fraction": fraction, "grouping": "relative video basename after first directory",
                "train_ids": [ids[i] for i, split in enumerate(result["splits"]) if split == "train"],
                "validation_ids": [ids[i] for i in sorted(selected)],
                "original_test_ids": [ids[i] for i, split in enumerate(payload["splits"]) if split == "test"],
                "legacy_internal_test_means": "TRAIN-derived validation; not original test",
                "original_test_independence": "historically used for teacher checkpoint selection; not newly untouched"}
    return result, manifest
