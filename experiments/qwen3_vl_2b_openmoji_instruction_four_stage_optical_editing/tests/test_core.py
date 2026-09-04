from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.assets import (
    ICON_SPECS,
    render_grid,
    validate_assets,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import (
    MetricAccumulator,
    compose_prediction,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.modeling import (
    SemanticGridDecoder,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.scenes import (
    TASKS,
    generate_example,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.settings import (
    load_settings,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_assets_are_pinned_and_complete() -> None:
    settings = load_settings(CONFIG)
    metadata = validate_assets(settings)
    assert metadata["version"] == "17.0.0"
    assert len(ICON_SPECS) == 16


def test_every_task_is_deterministic_and_changes_the_grid() -> None:
    settings = load_settings(CONFIG)
    for task in TASKS:
        first = generate_example(task, 12345, settings)
        second = generate_example(task, 12345, settings)
        assert first["instruction"] == second["instruction"]
        assert np.array_equal(first["target_grid"], second["target_grid"])
        assert first["edit_grid"].any()
        assert first["source_grid"].max() <= 16
        assert first["target_grid"].max() <= 16


def test_many_move_scenes_always_have_a_valid_target() -> None:
    settings = load_settings(CONFIG)
    for seed in range(1_000):
        example = generate_example("move", seed, settings)
        assert example["edit_grid"].sum() == 2


def test_renderer_and_decoder_shapes() -> None:
    settings = load_settings(CONFIG)
    example = generate_example("add", 5, settings)
    image = render_grid(example["source_grid"], settings)
    assert image.size == (224, 224)
    decoder = SemanticGridDecoder(192, 6, 16)
    category, edit = decoder(torch.randn(2, 192, 14, 14))
    assert category.shape == (2, 17, 6, 6)
    assert edit.shape == (2, 6, 6)


def test_composition_copies_unedited_cells() -> None:
    source = torch.zeros(1, 6, 6, dtype=torch.long)
    source[0, 1, 1] = 3
    logits = torch.zeros(1, 17, 6, 6)
    logits[:, 7] = 1.0
    edit_logits = torch.full((1, 6, 6), -10.0)
    edit_logits[0, 2, 2] = 10.0
    output, _, edit = compose_prediction(logits, edit_logits, source)
    assert output[0, 1, 1] == 3
    assert output[0, 2, 2] == 7
    assert edit.sum() == 1


def test_perfect_scene_has_perfect_object_f1() -> None:
    source = torch.zeros(1, 6, 6, dtype=torch.long)
    target = source.clone()
    target[0, 2, 3] = 5
    category_logits = torch.full((1, 17, 6, 6), -10.0)
    category_logits[:, 0] = 10.0
    category_logits[0, 0, 2, 3] = -10.0
    category_logits[0, 5, 2, 3] = 10.0
    edit_logits = torch.full((1, 6, 6), -10.0)
    edit_logits[0, 2, 3] = 10.0
    batch = {
        "source_grid": source,
        "target_grid": target,
        "edit_grid": source.ne(target).float(),
        "preserve_grid": source.eq(target).float(),
        "task_index": torch.tensor([0]),
        "task": ["add"],
        "sample_id": ["perfect"],
        "instruction": ["Add the dog."],
    }
    outputs = {
        "category_logits": category_logits,
        "edit_logits": edit_logits,
        "task_logits": torch.tensor([[10.0, 0.0, 0.0, 0.0]]),
    }
    accumulator = MetricAccumulator()
    accumulator.update(outputs, batch)
    metrics = accumulator.compute()["overall"]
    assert metrics["object_f1"] == 1.0
    assert metrics["scene_exact_match"] == 1.0


def test_paths_remain_inside_the_experiment() -> None:
    settings = load_settings(CONFIG)
    assert settings.output_dir.parent.name == "runs"
    assert settings.data_dir.parent.name == "data"
    assert settings.output_dir.parent.parent.name == (
        "qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing"
    )
