from __future__ import annotations

from pathlib import Path

import torch

from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.metrics import (
    compose_prediction,
)
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.modeling import (
    StructuredCanvasDecoder,
    restore_block_major,
)
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.scenes import (
    TASKS,
    edge_classes,
    generate_example,
)
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.settings import (
    load_settings,
)
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.training import EMA


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_all_tasks_are_deterministic_and_well_formed() -> None:
    settings = load_settings(CONFIG)
    for task in TASKS:
        first = generate_example(task, 1234, settings)
        second = generate_example(task, 1234, settings)
        assert first["instruction"] == second["instruction"]
        assert torch.equal(
            torch.from_numpy(first["target_classes"]),
            torch.from_numpy(second["target_classes"]),
        )
        assert first["source_classes"].shape == (224, 224)
        assert first["target_classes"].max() < 8
        if task == "edge":
            assert first["edit_mask"].all()


def test_edge_map_is_black_on_white_and_nonempty() -> None:
    settings = load_settings(CONFIG)
    example = generate_example("edge", 99, settings)
    edge = edge_classes(example["source_classes"])
    assert set(torch.unique(torch.from_numpy(edge)).tolist()).issubset({0, 7})
    assert (edge == 7).any()


def test_attribute_generation_has_a_valid_edit_across_many_seeds() -> None:
    settings = load_settings(CONFIG)
    for seed in range(2_000):
        example = generate_example("attribute", seed, settings)
        assert example["program"]["operation"] in {"recolor", "reshape"}
        assert torch.from_numpy(
            example["source_classes"] != example["target_classes"]
        ).any()


def test_block_major_restore_is_exact() -> None:
    regular = torch.arange(14 * 14 * 3).view(1, 14, 14, 3)
    block = (
        regular.view(1, 7, 2, 7, 2, 3)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(1, 196, 3)
    )
    restored = restore_block_major(block)
    assert torch.equal(restored.permute(0, 2, 3, 1), regular)


def test_decoder_shapes_and_composition() -> None:
    decoder = StructuredCanvasDecoder()
    palette, edit = decoder(torch.randn(2, 192, 14, 14), torch.randn(2, 192))
    assert palette.shape == (2, 8, 224, 224)
    assert edit.shape == (2, 224, 224)
    source = torch.zeros(2, 224, 224, dtype=torch.long)
    composed, generated, mask = compose_prediction(palette, edit, source)
    assert composed.shape == source.shape
    assert generated.shape == source.shape
    assert mask.dtype == torch.bool


def test_ema_can_be_swapped_in_for_epoch_test_and_restored() -> None:
    model = torch.nn.Linear(3, 2)
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    ema = EMA(model, 0.9)
    for value in ema.shadow.values():
        value.zero_()
    backup = ema.copy_to(model)
    assert all(torch.count_nonzero(value) == 0 for value in model.parameters())
    EMA.restore(model, backup)
    assert all(
        torch.equal(value, original[name]) for name, value in model.named_parameters()
    )


def test_runs_are_scoped_inside_the_experiment() -> None:
    settings = load_settings(CONFIG)
    assert settings.output_dir.parent.name == "runs"
    assert settings.output_dir.parent.parent.name == (
        "qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing"
    )
