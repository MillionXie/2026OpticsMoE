from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assets import ICON_SPECS, INDEX_TO_SPEC, load_icons, render_changed_cells, render_grid, validate_assets
from .settings import Settings


TASKS = ("add", "replace", "move", "remove")
TASK_TO_INDEX = {name: index for index, name in enumerate(TASKS)}
RELATIONS = {
    "left_of": (0, -1),
    "right_of": (0, 1),
    "above": (-1, 0),
    "below": (1, 0),
}
RELATION_TEXT = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
}
PROMPT_TEMPLATES = {
    "add": (
        "Add the {new} {relation} the {reference}.",
        "Place a {new} {relation} the {reference}.",
        "Put the {new} directly {relation} the {reference}.",
    ),
    "replace": (
        "Replace the {old} with a {new}.",
        "Change the {old} into a {new}.",
        "Swap the {old} for a {new}.",
    ),
    "move": (
        "Move the {target} {relation} the {reference}.",
        "Place the {target} {relation} the {reference}.",
        "Relocate the {target} so it is {relation} the {reference}.",
    ),
    "remove": (
        "Remove the {target}.",
        "Delete the {target} from the scene.",
        "Erase the {target}.",
    ),
}


@dataclass(frozen=True, slots=True)
class SceneObject:
    object_id: int
    category: int
    row: int
    col: int

    def to_dict(self) -> dict[str, Any]:
        spec = INDEX_TO_SPEC[self.category]
        return {
            "object_id": self.object_id,
            "category_index": self.category,
            "category": spec.name,
            "openmoji_codepoint": spec.codepoint,
            "grid": [self.row, self.col],
        }


def prompt_key(instruction: str) -> str:
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()


def _template(rng: random.Random, task: str, settings: Settings) -> str:
    count = min(settings.prompt_templates_per_operation, len(PROMPT_TEMPLATES[task]))
    return rng.choice(PROMPT_TEMPLATES[task][:count])


def _random_scene(rng: random.Random, count: int, settings: Settings) -> list[SceneObject]:
    cells = rng.sample(
        [(row, col) for row in range(settings.grid_size) for col in range(settings.grid_size)],
        count,
    )
    categories = rng.sample([spec.index for spec in ICON_SPECS], count)
    return [
        SceneObject(index, category, row, col)
        for index, (category, (row, col)) in enumerate(zip(categories, cells))
    ]


def objects_to_grid(objects: list[SceneObject], settings: Settings) -> np.ndarray:
    grid = np.zeros((settings.grid_size, settings.grid_size), dtype=np.uint8)
    for obj in objects:
        if grid[obj.row, obj.col] != 0:
            raise RuntimeError("Two OpenMoji objects occupy the same grid cell")
        grid[obj.row, obj.col] = obj.category
    return grid


def generate_example(task: str, seed: int, settings: Settings) -> dict[str, Any]:
    if task not in TASK_TO_INDEX:
        raise ValueError(f"Unknown task {task}")
    rng = random.Random(seed)
    low, high = {"add": (1, 4), "replace": (1, 4), "move": (2, 4), "remove": (2, 4)}[task]
    source = _random_scene(rng, rng.randint(low, high), settings)
    target = list(source)

    if task == "add":
        occupied = {(obj.row, obj.col) for obj in source}
        positions = []
        for reference in source:
            for relation, (dr, dc) in RELATIONS.items():
                row, col = reference.row + dr, reference.col + dc
                if 0 <= row < settings.grid_size and 0 <= col < settings.grid_size and (row, col) not in occupied:
                    positions.append((reference, relation, row, col))
        if not positions:
            return generate_example(task, seed + 1_000_003, settings)
        reference, relation, row, col = rng.choice(positions)
        present = {obj.category for obj in source}
        new_category = rng.choice([spec.index for spec in ICON_SPECS if spec.index not in present])
        new_object = SceneObject(len(source), new_category, row, col)
        target.append(new_object)
        instruction = _template(rng, task, settings).format(
            new=INDEX_TO_SPEC[new_category].name,
            relation=RELATION_TEXT[relation],
            reference=INDEX_TO_SPEC[reference.category].name,
        )
        program = {
            "operation": task,
            "new_object": new_object.to_dict(),
            "reference_object_id": reference.object_id,
            "relation": relation,
        }
    elif task == "replace":
        selected = rng.choice(source)
        present = {obj.category for obj in source}
        new_category = rng.choice([spec.index for spec in ICON_SPECS if spec.index not in present])
        replacement = SceneObject(
            selected.object_id, new_category, selected.row, selected.col
        )
        target[selected.object_id] = replacement
        instruction = _template(rng, task, settings).format(
            old=INDEX_TO_SPEC[selected.category].name,
            new=INDEX_TO_SPEC[new_category].name,
        )
        program = {
            "operation": task,
            "target_object_id": selected.object_id,
            "new_category": INDEX_TO_SPEC[new_category].name,
        }
    elif task == "move":
        occupied = {(obj.row, obj.col) for obj in source}
        positions = []
        for selected in source:
            for reference in source:
                if selected.object_id == reference.object_id:
                    continue
                for relation, (dr, dc) in RELATIONS.items():
                    row, col = reference.row + dr, reference.col + dc
                    if (
                        0 <= row < settings.grid_size
                        and 0 <= col < settings.grid_size
                        and (row, col) not in occupied
                    ):
                        positions.append((selected, reference, relation, row, col))
        if not positions:
            return generate_example(task, seed + 1_000_003, settings)
        selected, reference, relation, row, col = rng.choice(positions)
        moved = SceneObject(selected.object_id, selected.category, row, col)
        target[selected.object_id] = moved
        instruction = _template(rng, task, settings).format(
            target=INDEX_TO_SPEC[selected.category].name,
            relation=RELATION_TEXT[relation],
            reference=INDEX_TO_SPEC[reference.category].name,
        )
        program = {
            "operation": task,
            "target_object_id": selected.object_id,
            "reference_object_id": reference.object_id,
            "relation": relation,
            "new_grid": [row, col],
        }
    else:
        selected = rng.choice(source)
        target = [obj for obj in source if obj.object_id != selected.object_id]
        instruction = _template(rng, task, settings).format(
            target=INDEX_TO_SPEC[selected.category].name
        )
        program = {"operation": task, "target_object_id": selected.object_id}

    source_grid = objects_to_grid(source, settings)
    target_grid = objects_to_grid(target, settings)
    edit_grid = (source_grid != target_grid).astype(np.uint8)
    return {
        "task": task,
        "task_index": TASK_TO_INDEX[task],
        "seed": seed,
        "instruction": instruction,
        "prompt_key": prompt_key(instruction),
        "program": program,
        "source_scene": [obj.to_dict() for obj in source],
        "target_scene": [obj.to_dict() for obj in target],
        "source_grid": source_grid,
        "target_grid": target_grid,
        "edit_grid": edit_grid,
    }


def _save_sample(
    example: dict[str, Any], sample_dir: Path, settings: Settings, icons: dict[int, Any]
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    render_grid(example["source_grid"], settings, icons).save(sample_dir / "source.png", optimize=True)
    render_grid(example["target_grid"], settings, icons).save(sample_dir / "target.png", optimize=True)
    render_changed_cells(example["edit_grid"], settings).save(sample_dir / "edit_mask.png", optimize=True)
    metadata = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in example.items()
    }
    metadata["files"] = {
        "source": "source.png",
        "target": "target.png",
        "edit_mask": "edit_mask.png",
    }
    (sample_dir / "scene.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def generate_split(split: str, count: int, settings: Settings, icons: dict[int, Any]) -> dict[str, Any]:
    lines: list[str] = []
    task_counts = {task: 0 for task in TASKS}
    prompts: set[str] = set()
    split_offset = 0 if split == "train" else 20_000_000
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        seed = settings.seed * 1_000_003 + split_offset + index
        example = generate_example(task, seed, settings)
        sample_id = f"{split}_{index:06d}"
        relative_dir = Path(split) / sample_id
        metadata = _save_sample(example, settings.data_dir / relative_dir, settings, icons)
        record = {
            "sample_id": sample_id,
            "split": split,
            "relative_dir": relative_dir.as_posix(),
            **metadata,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        task_counts[task] += 1
        prompts.add(example["instruction"])
    manifest = settings.data_dir / f"{split}.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "samples": count,
        "task_counts": task_counts,
        "unique_prompts": len(prompts),
        "manifest": str(manifest),
    }


def prepare_dataset(settings: Settings) -> dict[str, Any]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    assets = validate_assets(settings)
    icons = load_icons(settings)
    train = generate_split("train", settings.train_samples, settings, icons)
    test = generate_split("test", settings.test_samples, settings, icons)
    summary = {
        "type": "openmoji_semantic_grid_instruction_editing_v1",
        "image_size": settings.image_size,
        "grid_size": settings.grid_size,
        "icon_size": settings.icon_size,
        "tasks": list(TASKS),
        "assets": assets,
        "train": train,
        "test": test,
        "validation_split": False,
        "unseen_split": False,
        "split_rule": "same icon/task/template distribution; disjoint deterministic seeds",
    }
    (settings.data_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


__all__ = [
    "TASKS",
    "TASK_TO_INDEX",
    "generate_example",
    "objects_to_grid",
    "prepare_dataset",
    "prompt_key",
]
