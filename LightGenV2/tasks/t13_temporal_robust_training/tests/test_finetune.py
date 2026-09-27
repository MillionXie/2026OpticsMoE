from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import make_config, load_protocol
from finetune import configure, validate_parent


def test_lr_reduced_without_other_changes():
    raw = make_config("r0_post")
    result = configure(raw)
    assert result["training"]["epochs"] == 30 and raw["training"]["epochs"] == 100
    for key in ("learning_rate", "phase_learning_rate", "router_phase_learning_rate"):
        assert result["training"][key] == pytest.approx(raw["training"][key]*.1)
    for key in ("robustness", "router", "optics", "data", "loss", "model"):
        assert raw[key] == result[key]
    with pytest.raises(ValueError):
        configure(raw, factor=1)


def test_parent_group_and_contract_are_enforced():
    protocol = load_protocol()
    saved = {"study_group": "r0_post", "study_protocol": dict(protocol), "state_dict": {}}
    validate_parent(saved, "r0_post", protocol)
    with pytest.raises(ValueError, match="group"):
        validate_parent(saved, "r1_ccd_post", protocol)
    saved["study_protocol"]["deployment_eta"] = .2
    with pytest.raises(ValueError, match="protocol"):
        validate_parent(saved, "r0_post", protocol)
