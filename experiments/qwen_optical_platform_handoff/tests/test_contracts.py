from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.qwen_optical_platform_handoff.contracts import (
    ContractError,
    load_and_validate_contract,
    validate_contract,
)


TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


@pytest.mark.parametrize("path", sorted(TEMPLATES.glob("*.contract.json")))
def test_templates_are_valid(path: Path) -> None:
    assert load_and_validate_contract(path)["schema_version"] == 1


def test_test_split_cannot_select_checkpoint() -> None:
    raw = json.loads((TEMPLATES / "simulation_retrieval.contract.json").read_text())
    broken = copy.deepcopy(raw)
    broken["training"]["selection_split"] = broken["data"]["test_split"]
    with pytest.raises(ContractError, match="selection_split"):
        validate_contract(broken)


def test_hardware_forbids_per_frame_minmax() -> None:
    raw = json.loads((TEMPLATES / "hardware_four_stage.contract.json").read_text())
    raw["hardware"]["detector"]["per_frame_minmax_normalization"] = True
    with pytest.raises(ContractError, match="min-max"):
        validate_contract(raw)
