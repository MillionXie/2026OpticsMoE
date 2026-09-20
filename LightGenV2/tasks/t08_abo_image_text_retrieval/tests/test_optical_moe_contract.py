from __future__ import annotations

from pathlib import Path

import pytest
import torch

from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import (
    DOCUMENT_INSTRUCTION,
    EMBEDDING_DIM,
    QUERY_INSTRUCTION,
    TEXT_TO_IMAGE_QUERY_INSTRUCTION,
    _prompt_contract,
    _resolve_from_config,
    _text_to_image_metrics,
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


def test_true_text_to_image_assigns_query_and_document_roles() -> None:
    prompts = _prompt_contract({
        "abo_image_text": {"retrieval_direction": "text_to_image"}
    })
    assert prompts.title_instruction == TEXT_TO_IMAGE_QUERY_INSTRUCTION
    assert prompts.image_instruction == DOCUMENT_INSTRUCTION


def test_15cm_text_to_image_contract_is_a_distinct_geometry() -> None:
    settings = load_settings(
        TASK_DIR / "configs" / "optical_text_to_image_64_15cm_true.yaml"
    )
    assert settings.language_optical_distance_m == pytest.approx(0.15)
    assert settings.expert_interlayer_distance_m == pytest.approx(0.15)


def test_relative_paths_resolve_from_config(tmp_path) -> None:
    config = tmp_path / "configs" / "x.yaml"
    config.parent.mkdir()
    assert _resolve_from_config(config, "../data") == (tmp_path / "data").resolve()


def test_text_to_image_metrics_treat_all24_sku_images_as_positive() -> None:
    # The production contract has100 titles and exactly24 held-out images/SKU.
    titles = torch.eye(100)
    images = torch.eye(100).repeat_interleave(24, dim=0)
    labels = torch.arange(100).repeat_interleave(24).tolist()
    report, rows = _text_to_image_metrics(titles, images, labels)
    assert report["hit_at_1"] == 1.0
    assert report["recall_at_1"] == pytest.approx(1 / 24)
    assert report["map"] == 1.0
    assert len(rows) == 100
