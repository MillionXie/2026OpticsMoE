from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .settings import Settings


TASKS = ("attribute", "add", "remove", "edge")
TASK_TO_INDEX = {name: index for index, name in enumerate(TASKS)}
SHAPES = ("circle", "square", "triangle")
COLOR_NAMES = ("background", "red", "green", "blue", "yellow", "cyan", "magenta", "black")
PALETTE = np.asarray(
    [
        (255, 255, 255),
        (230, 45, 55),
        (45, 180, 75),
        (45, 95, 220),
        (240, 205, 45),
        (35, 200, 205),
        (205, 55, 205),
        (0, 0, 0),
    ],
    dtype=np.uint8,
)
EDIT_COLORS = tuple(range(1, 7))
RELATIONS = {
    "left_of": (0, -1),
    "right_of": (0, 1),
    "above": (-1, 0),
    "below": (1, 0),
}


PROMPT_TEMPLATES = {
    "recolor": (
        "Change the {old_color} {shape} to {new_color}.",
        "Make the {old_color} {shape} {new_color}.",
        "Recolor the {old_color} {shape} as {new_color}.",
    ),
    "reshape": (
        "Change the {color} {old_shape} into a {new_shape}.",
        "Turn the {color} {old_shape} into a {new_shape}.",
        "Replace the shape of the {color} {old_shape} with a {new_shape}.",
    ),
    "add": (
        "Add a {new_color} {new_shape} {relation} the {ref_color} {ref_shape}.",
        "Place a {new_color} {new_shape} {relation} the {ref_color} {ref_shape}.",
        "Draw a {new_color} {new_shape} directly {relation} the {ref_color} {ref_shape}.",
    ),
    "remove": (
        "Remove the {color} {shape}.",
        "Delete the {color} {shape} from the scene.",
        "Erase the {color} {shape}.",
    ),
    "edge": (
        "Extract the edges of all objects.",
        "Show only the outlines of every object.",
        "Convert the scene into a black and white object edge map.",
    ),
}

RELATION_TEXT = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
}


@dataclass(frozen=True, slots=True)
class SceneObject:
    object_id: int
    shape: str
    color: int
    row: int
    col: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "shape": self.shape,
            "color": COLOR_NAMES[self.color],
            "color_index": self.color,
            "grid": [self.row, self.col],
        }


def prompt_key(instruction: str) -> str:
    return hashlib.sha1(instruction.encode("utf-8")).hexdigest()


def _grid_centers(image_size: int, grid_size: int) -> list[int]:
    margin = image_size // (grid_size + 1)
    return [margin * (index + 1) for index in range(grid_size)]


def object_mask(obj: SceneObject, settings: Settings) -> np.ndarray:
    centers = _grid_centers(settings.image_size, settings.grid_size)
    center_x, center_y = centers[obj.col], centers[obj.row]
    half = settings.object_half_size
    image = Image.new("L", (settings.image_size, settings.image_size), 0)
    draw = ImageDraw.Draw(image)
    if obj.shape == "circle":
        draw.ellipse((center_x - half, center_y - half, center_x + half, center_y + half), fill=1)
    elif obj.shape == "square":
        draw.rectangle((center_x - half, center_y - half, center_x + half, center_y + half), fill=1)
    elif obj.shape == "triangle":
        draw.polygon(
            [
                (center_x, center_y - half),
                (center_x - half, center_y + half),
                (center_x + half, center_y + half),
            ],
            fill=1,
        )
    else:
        raise ValueError(f"Unknown shape {obj.shape}")
    return np.asarray(image, dtype=np.uint8).astype(bool)


def render_classes(objects: list[SceneObject], settings: Settings) -> np.ndarray:
    canvas = np.zeros((settings.image_size, settings.image_size), dtype=np.uint8)
    for obj in objects:
        mask = object_mask(obj, settings)
        if np.any(canvas[mask] != 0):
            raise RuntimeError("Synthetic objects overlap")
        canvas[mask] = obj.color
    return canvas


def classes_to_rgb(classes: np.ndarray) -> np.ndarray:
    if classes.min() < 0 or classes.max() >= len(PALETTE):
        raise ValueError("Class canvas contains an invalid palette index")
    return PALETTE[classes]


def edge_classes(source: np.ndarray) -> np.ndarray:
    foreground = source != 0
    boundary = np.zeros_like(foreground)
    boundary[1:, :] |= foreground[1:, :] & (source[1:, :] != source[:-1, :])
    boundary[:-1, :] |= foreground[:-1, :] & (source[:-1, :] != source[1:, :])
    boundary[:, 1:] |= foreground[:, 1:] & (source[:, 1:] != source[:, :-1])
    boundary[:, :-1] |= foreground[:, :-1] & (source[:, :-1] != source[:, 1:])
    result = np.zeros_like(source)
    result[boundary] = 7
    return result


def _unique_descriptor(shape: str, color: int, objects: list[SceneObject], ignore: int | None = None) -> bool:
    return all(
        obj.object_id == ignore or obj.shape != shape or obj.color != color
        for obj in objects
    )


def _random_scene(rng: random.Random, count: int, settings: Settings) -> list[SceneObject]:
    cells = rng.sample(
        [(row, col) for row in range(settings.grid_size) for col in range(settings.grid_size)],
        count,
    )
    descriptors = rng.sample([(shape, color) for shape in SHAPES for color in EDIT_COLORS], count)
    return [
        SceneObject(index, shape, color, row, col)
        for index, ((row, col), (shape, color)) in enumerate(zip(cells, descriptors))
    ]


def _template(rng: random.Random, operation: str, settings: Settings) -> str:
    templates = PROMPT_TEMPLATES[operation][
        : min(settings.prompt_templates_per_operation, len(PROMPT_TEMPLATES[operation]))
    ]
    return rng.choice(templates)


def generate_example(task: str, seed: int, settings: Settings) -> dict[str, Any]:
    if task not in TASK_TO_INDEX:
        raise ValueError(f"Unknown task {task}")
    rng = random.Random(seed)
    count_ranges = {
        "attribute": (1, 3),
        "add": (1, 3),
        "remove": (2, 4),
        "edge": (1, 4),
    }
    low, high = count_ranges[task]
    source_objects = _random_scene(rng, rng.randint(low, high), settings)
    target_objects = list(source_objects)

    if task == "attribute":
        target = rng.choice(source_objects)
        recolor_candidates = [
            color
            for color in EDIT_COLORS
            if color != target.color
            and _unique_descriptor(target.shape, color, source_objects, target.object_id)
        ]
        reshape_candidates = [
            shape
            for shape in SHAPES
            if shape != target.shape
            and _unique_descriptor(shape, target.color, source_objects, target.object_id)
        ]
        available_operations = []
        if recolor_candidates:
            available_operations.append("recolor")
        if reshape_candidates:
            available_operations.append("reshape")
        operation = rng.choice(available_operations)
        if operation == "recolor":
            new_color = rng.choice(recolor_candidates)
            target_objects[target.object_id] = SceneObject(
                target.object_id, target.shape, new_color, target.row, target.col
            )
            instruction = _template(rng, operation, settings).format(
                old_color=COLOR_NAMES[target.color],
                shape=target.shape,
                new_color=COLOR_NAMES[new_color],
            )
            program = {
                "operation": operation,
                "target_object_id": target.object_id,
                "new_color": COLOR_NAMES[new_color],
            }
        else:
            new_shape = rng.choice(reshape_candidates)
            target_objects[target.object_id] = SceneObject(
                target.object_id, new_shape, target.color, target.row, target.col
            )
            instruction = _template(rng, operation, settings).format(
                color=COLOR_NAMES[target.color],
                old_shape=target.shape,
                new_shape=new_shape,
            )
            program = {
                "operation": operation,
                "target_object_id": target.object_id,
                "new_shape": new_shape,
            }
    elif task == "add":
        occupied = {(obj.row, obj.col) for obj in source_objects}
        candidates: list[tuple[SceneObject, str, int, int]] = []
        for reference in source_objects:
            for relation, (dr, dc) in RELATIONS.items():
                row, col = reference.row + dr, reference.col + dc
                if 0 <= row < settings.grid_size and 0 <= col < settings.grid_size and (row, col) not in occupied:
                    candidates.append((reference, relation, row, col))
        if not candidates:
            return generate_example(task, seed + 1_000_003, settings)
        reference, relation, row, col = rng.choice(candidates)
        descriptors = [
            (shape, color)
            for shape in SHAPES
            for color in EDIT_COLORS
            if _unique_descriptor(shape, color, source_objects)
        ]
        new_shape, new_color = rng.choice(descriptors)
        new_object = SceneObject(len(source_objects), new_shape, new_color, row, col)
        target_objects.append(new_object)
        operation = "add"
        instruction = _template(rng, operation, settings).format(
            new_color=COLOR_NAMES[new_color],
            new_shape=new_shape,
            relation=RELATION_TEXT[relation],
            ref_color=COLOR_NAMES[reference.color],
            ref_shape=reference.shape,
        )
        program = {
            "operation": operation,
            "new_object": new_object.to_dict(),
            "reference_object_id": reference.object_id,
            "relation": relation,
            "grid_offset": list(RELATIONS[relation]),
        }
    elif task == "remove":
        target = rng.choice(source_objects)
        target_objects = [obj for obj in source_objects if obj.object_id != target.object_id]
        operation = "remove"
        instruction = _template(rng, operation, settings).format(
            color=COLOR_NAMES[target.color], shape=target.shape
        )
        program = {"operation": operation, "target_object_id": target.object_id}
    else:
        operation = "edge"
        instruction = _template(rng, operation, settings)
        program = {"operation": operation, "scope": "all_objects"}

    source_classes = render_classes(source_objects, settings)
    target_classes = (
        edge_classes(source_classes)
        if task == "edge"
        else render_classes(target_objects, settings)
    )
    edit_mask = (
        np.ones_like(source_classes, dtype=np.uint8)
        if task == "edge"
        else (source_classes != target_classes).astype(np.uint8)
    )
    preserve_mask = (1 - edit_mask).astype(np.uint8)
    return {
        "task": task,
        "task_index": TASK_TO_INDEX[task],
        "seed": seed,
        "instruction": instruction,
        "prompt_key": prompt_key(instruction),
        "program": program,
        "source_scene": [obj.to_dict() for obj in source_objects],
        "target_scene": [obj.to_dict() for obj in target_objects],
        "source_classes": source_classes,
        "target_classes": target_classes,
        "edit_mask": edit_mask,
        "preserve_mask": preserve_mask,
    }


def _save_sample(example: dict[str, Any], sample_dir: Path) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "source": classes_to_rgb(example["source_classes"]),
        "target": classes_to_rgb(example["target_classes"]),
        "source_classes": example["source_classes"],
        "target_classes": example["target_classes"],
        "edit_mask": example["edit_mask"] * 255,
        "preserve_mask": example["preserve_mask"] * 255,
    }
    filenames: dict[str, str] = {}
    for name, array in arrays.items():
        filename = f"{name}.png"
        Image.fromarray(array.astype(np.uint8)).save(sample_dir / filename, optimize=True)
        filenames[name] = filename
    metadata = {
        key: value
        for key, value in example.items()
        if not isinstance(value, np.ndarray)
    }
    metadata["files"] = filenames
    (sample_dir / "scene.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def generate_split(split: str, count: int, settings: Settings) -> dict[str, Any]:
    split_dir = settings.data_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.data_dir / f"{split}.jsonl"
    task_counts = {task: 0 for task in TASKS}
    unique_prompts: set[str] = set()
    lines: list[str] = []
    split_offset = 0 if split == "train" else 10_000_000
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        seed = settings.seed * 1_000_003 + split_offset + index
        example = generate_example(task, seed, settings)
        sample_id = f"{split}_{index:06d}"
        relative_dir = Path(split) / sample_id
        metadata = _save_sample(example, settings.data_dir / relative_dir)
        record = {
            "sample_id": sample_id,
            "split": split,
            "relative_dir": relative_dir.as_posix(),
            **metadata,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        task_counts[task] += 1
        unique_prompts.add(example["instruction"])
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "split": split,
        "samples": count,
        "task_counts": task_counts,
        "unique_prompts": len(unique_prompts),
        "manifest": str(manifest_path),
    }


def prepare_dataset(settings: Settings) -> dict[str, Any]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    train = generate_split("train", settings.train_samples, settings)
    test = generate_split("test", settings.test_samples, settings)
    summary = {
        "type": "synthetic_prompt_conditioned_geometric_editing_v1",
        "image_size": settings.image_size,
        "grid_size": settings.grid_size,
        "tasks": list(TASKS),
        "palette": {name: PALETTE[index].tolist() for index, name in enumerate(COLOR_NAMES)},
        "train": train,
        "test": test,
        "validation_split": False,
        "unseen_composition_split": False,
        "split_rule": "same task/template/composition distribution; disjoint deterministic scene seeds",
    }
    (settings.data_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


__all__ = [
    "COLOR_NAMES",
    "PALETTE",
    "SHAPES",
    "TASKS",
    "TASK_TO_INDEX",
    "classes_to_rgb",
    "edge_classes",
    "generate_example",
    "prepare_dataset",
    "prompt_key",
    "render_classes",
]
