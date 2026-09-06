from __future__ import annotations

from pathlib import Path

import pytest

from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import (
    DOCUMENT_INSTRUCTION,
    EMBEDDING_DIM,
    QUERY_INSTRUCTION,
    _resolve_from_config,
)
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings


TASK_DIR = Path(__file__).resolve().parents[1]


def test_abo_optical_contract_matches_current_t01_hardware_graph() -> None:
    settings = load_settings(TASK_DIR / "configs" / "optical_router_moe_dc20.yaml")
    assert settings.router_backend == "optical"
    assert settings.top_k == 2
    assert settings.expert_size == 224
    assert settings.language_optical_distance_m == pytest.approx(0.10)
    assert settings.language_optical_pixel_pitch_um == pytest.approx(17.0)
    assert settings.fusion_mode == "scale_matched_convex"
    assert settings.phase_dc_enabled is True
    assert settings.language_optical_zero_order_enabled is True
    assert settings.embedding_dim == EMBEDDING_DIM == 64


def test_abo_prompts_match_frozen_baseline() -> None:
    assert "product title" in QUERY_INSTRUCTION
    assert DOCUMENT_INSTRUCTION == "Represent the user's input."


def test_relative_paths_resolve_from_config(tmp_path) -> None:
    config = tmp_path / "configs" / "x.yaml"
    config.parent.mkdir()
    assert _resolve_from_config(config, "../data") == (tmp_path / "data").resolve()
