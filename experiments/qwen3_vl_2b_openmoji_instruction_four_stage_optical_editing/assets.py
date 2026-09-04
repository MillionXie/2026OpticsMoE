from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .settings import Settings


@dataclass(frozen=True, slots=True)
class IconSpec:
    index: int
    name: str
    codepoint: str


ICON_SPECS = (
    IconSpec(1, "apple", "1F34E"),
    IconSpec(2, "bicycle", "1F6B2"),
    IconSpec(3, "car", "1F697"),
    IconSpec(4, "bus", "1F68C"),
    IconSpec(5, "dog", "1F415"),
    IconSpec(6, "cat", "1F408"),
    IconSpec(7, "bird", "1F426"),
    IconSpec(8, "tree", "1F333"),
    IconSpec(9, "flower", "1F33B"),
    IconSpec(10, "cup", "2615"),
    IconSpec(11, "book", "1F4D8"),
    IconSpec(12, "phone", "1F4F1"),
    IconSpec(13, "ball", "26BD"),
    IconSpec(14, "umbrella", "2602"),
    IconSpec(15, "house", "1F3E0"),
    IconSpec(16, "light bulb", "1F4A1"),
)
INDEX_TO_SPEC = {spec.index: spec for spec in ICON_SPECS}
NAME_TO_SPEC = {spec.name: spec for spec in ICON_SPECS}


def validate_assets(settings: Settings) -> dict[str, object]:
    missing = [
        spec.codepoint
        for spec in ICON_SPECS
        if not (settings.asset_dir / f"{spec.codepoint}.png").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing OpenMoji 17.0.0 PNG assets under {settings.asset_dir}: {missing}"
        )
    metadata = {
        "source": "OpenMoji",
        "version": "17.0.0",
        "source_url": "https://github.com/hfg-gmuend/openmoji/releases/tag/17.0.0",
        "license": "CC BY-SA 4.0",
        "asset_format": "official 72x72 color PNG",
        "classes": [
            {"index": spec.index, "name": spec.name, "codepoint": spec.codepoint}
            for spec in ICON_SPECS
        ],
    }
    settings.asset_dir.mkdir(parents=True, exist_ok=True)
    (settings.asset_dir / "asset_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def load_icons(settings: Settings) -> dict[int, Image.Image]:
    validate_assets(settings)
    return {
        spec.index: Image.open(settings.asset_dir / f"{spec.codepoint}.png")
        .convert("RGBA")
        .resize((settings.icon_size, settings.icon_size), Image.Resampling.LANCZOS)
        for spec in ICON_SPECS
    }


def grid_centers(settings: Settings) -> list[int]:
    return [
        round((index + 1) * settings.image_size / (settings.grid_size + 1))
        for index in range(settings.grid_size)
    ]


def render_grid(
    grid: np.ndarray,
    settings: Settings,
    icons: dict[int, Image.Image] | None = None,
) -> Image.Image:
    if grid.shape != (settings.grid_size, settings.grid_size):
        raise ValueError(f"Invalid grid shape {grid.shape}")
    icons = load_icons(settings) if icons is None else icons
    canvas = Image.new("RGB", (settings.image_size, settings.image_size), (250, 250, 247))
    centers = grid_centers(settings)
    half = settings.icon_size // 2
    for row, col in zip(*np.nonzero(grid)):
        category = int(grid[row, col])
        icon = icons[category]
        canvas.paste(icon, (centers[col] - half, centers[row] - half), icon)
    return canvas


def render_changed_cells(changed: np.ndarray, settings: Settings) -> Image.Image:
    image = np.zeros((settings.image_size, settings.image_size), dtype=np.uint8)
    boundaries = [
        round(index * settings.image_size / settings.grid_size)
        for index in range(settings.grid_size + 1)
    ]
    for row, col in zip(*np.nonzero(changed)):
        image[boundaries[row] : boundaries[row + 1], boundaries[col] : boundaries[col + 1]] = 255
    return Image.fromarray(image, mode="L")


__all__ = [
    "ICON_SPECS",
    "INDEX_TO_SPEC",
    "NAME_TO_SPEC",
    "grid_centers",
    "load_icons",
    "render_changed_cells",
    "render_grid",
    "validate_assets",
]
