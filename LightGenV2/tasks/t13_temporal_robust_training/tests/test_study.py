from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study import GROUPS, make_config, split_for_selection


def test_four_cumulative_groups_change_one_axis():
    conditions = list(GROUPS.values())
    assert len(conditions) == 4
    for left, right in zip(conditions, conditions[1:]):
        assert sum(left[key] != right[key] for key in left) == 1


def test_train_mapping_and_common_deployment():
    for group, condition in GROUPS.items():
        train = make_config(group)
        deploy = make_config(group, purpose="deployment")
        assert train["optics"]["modulator_pixel_pitch_um"] == (8 if condition["mapping"] == "in_training" else 17)
        assert deploy["optics"]["modulator_pixel_pitch_um"] == 8
        assert deploy["optics"]["unmodulated_power_fraction_eval"] == 0.3
        assert train["training"]["initialization_checkpoint"] is None
        assert train["robustness"]["ccd_noise"]["enabled"] == condition["ccd"]
        assert train["optics"]["unmodulated_power_fraction_max"] == (0.3 if condition["dc"] else 0)


def test_validation_never_uses_original_test():
    payload = {"sample_ids": ["a/one.mp4", "b/one.mp4", "a/two.mp4", "b/three.mp4", "a/sealed.mp4"],
               "splits": ["train", "train", "train", "train", "test"]}
    result, manifest = split_for_selection(payload, fraction=0.5)
    assert result["splits"][-1] == "sealed"
    assert result["splits"][0] == result["splits"][1]
    assert set(manifest["validation_ids"]).isdisjoint(manifest["original_test_ids"])
    assert set(manifest["train_ids"]).isdisjoint(manifest["validation_ids"])
    assert split_for_selection(payload, fraction=0.5) == (result, manifest)


def test_unrelated_augmentation_is_identical():
    for group in GROUPS:
        raw = make_config(group)
        assert raw["robustness"]["phase_dropout_p"] == 0
        assert raw["router"]["noise_std"] == 0
        assert all(raw["robustness"][key] == 0 for key in ("input_shift_pixels", "phase_shift_pixels", "ccd_shift_pixels"))


def test_theory_is_ideal_r0_without_extra_training():
    raw = make_config("r0_post", purpose="nominal")
    assert raw["optics"]["modulator_pixel_pitch_um"] == 17
    assert raw["optics"]["unmodulated_power_fraction_eval"] == 0
    assert not raw["robustness"]["ccd_noise"]["enabled"]


def test_immutable_source_manifest():
    from verify_source import verify
    assert verify()["status"] == "passed"


def test_plot_refuses_simulation_as_hardware():
    import importlib.util
    spec = importlib.util.spec_from_file_location("plot_results", ROOT / "tools" / "plot_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"id": f"G{i}", "scope": "simulation" if i == 1 else "hardware",
             "metrics": {"srcc": 0.5}, "checkpoint_sha256": "same", "session": "fixture"} for i in range(1,6)]
    assert len(module.validate_conditions(rows)) == 5
    import pytest
    rows[1]["scope"] = "simulation"
    with pytest.raises(ValueError):
        module.validate_conditions(rows)
