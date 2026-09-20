"""Reproducible evaluation gallery for the proportional layered-scene split."""

from __future__ import annotations

import html
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.scenes import TASKS
from .layered_scene_data import _artwork
from .layout_preview import font


def _compact_anchor_grid(grid: np.ndarray, svg_dir: Path) -> Image.Image:
    """Draw arbitrary predictions without pretending invalid anchors form a scene."""
    canvas = Image.new('RGB', (224, 224), (250, 250, 247))
    draw = ImageDraw.Draw(canvas)
    edges = [round(index * 224 / 6) for index in range(7)]
    for edge in edges:
        draw.line((edge, 0, edge, 224), fill=(220, 220, 216), width=1)
        draw.line((0, edge, 224, edge), fill=(220, 220, 216), width=1)
    for row, col in zip(*np.nonzero(grid)):
        category = int(grid[row, col])
        icon = _artwork(str(svg_dir.resolve()), category).copy()
        maximum = min(edges[col + 1] - edges[col] - 6, edges[row + 1] - edges[row] - 6)
        scale = min(maximum / icon.width, maximum / icon.height)
        size = (max(1, round(icon.width * scale)), max(1, round(icon.height * scale)))
        icon = icon.resize(size, Image.Resampling.LANCZOS)
        x = (edges[col] + edges[col + 1] - icon.width) // 2
        y = (edges[row] + edges[row + 1] - icon.height) // 2
        canvas.paste(icon.convert('RGB'), (x, y), icon.getchannel('A'))
    return canvas


def _error_grid(target: np.ndarray, prediction: np.ndarray) -> Image.Image:
    canvas = Image.new('RGB', (224, 224), 'white')
    draw = ImageDraw.Draw(canvas)
    edges = [round(index * 224 / 6) for index in range(7)]
    for row in range(6):
        for col in range(6):
            mismatch = int(target[row, col]) != int(prediction[row, col])
            color = (230, 80, 72) if mismatch else (232, 242, 232)
            draw.rectangle((edges[col], edges[row], edges[col + 1], edges[row + 1]), fill=color)
    for edge in edges:
        draw.line((edge, 0, edge, 224), fill=(90, 90, 90), width=1)
        draw.line((0, edge, 224, edge), fill=(90, 90, 90), width=1)
    return canvas


def _card(sample: dict[str, Any], settings: Any) -> Image.Image:
    sample_dir = settings.data_dir / 'test' / sample['sample_id']
    source = Image.open(sample_dir / 'source.png').convert('RGB')
    target_image = Image.open(sample_dir / 'target.png').convert('RGB')
    target = np.asarray(sample['target'])
    prediction = np.asarray(sample['prediction'])
    panels = [
        source,
        target_image,
        _compact_anchor_grid(prediction, settings.svg_asset_dir),
        _error_grid(target, prediction),
    ]
    header, footer = 72, 24
    canvas = Image.new('RGB', (224 * 4, header + 224 + footer), 'white')
    draw = ImageDraw.Draw(canvas)
    title = f"{sample['sample_id']} | task={sample['task']} | {sample['instruction']}"
    draw.multiline_text((8, 6), '\n'.join(textwrap.wrap(title, width=115)),
                        fill='black', spacing=3, font=font(13))
    labels = ('INPUT SCENE', 'TARGET SCENE', 'PREDICTED ANCHORS', 'ERROR ANCHORS')
    for index, (label, panel) in enumerate(zip(labels, panels)):
        x = index * 224
        canvas.paste(panel, (x, header))
        draw.text((x + 6, header + 228), label, fill='black', font=font(12))
    return canvas


def save_layered_gallery(
    directory: Path,
    galleries: dict[str, list[dict[str, Any]]],
    settings: Any,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    links: list[tuple[str, str]] = []
    for task in TASKS:
        samples = galleries.get(task, [])
        if not samples:
            continue
        cards = [_card(sample, settings) for sample in samples]
        sheet = Image.new('RGB', (cards[0].width, sum(card.height for card in cards)), 'white')
        y = 0
        for card in cards:
            sheet.paste(card, (0, y)); y += card.height
        filename = f'{task}_examples.png'
        sheet.save(directory / filename, optimize=True)
        links.append((task, filename))
    body = ["<!doctype html><meta charset='utf-8'><title>Layered OpenMoji examples</title>",
            "<style>body{font-family:sans-serif}img{max-width:100%;border:1px solid #bbb}</style>"]
    for task, filename in links:
        body.append(f"<h2>{html.escape(task)}</h2><img src='{html.escape(filename)}'>")
    (directory / 'index.html').write_text('\n'.join(body), encoding='utf-8')


__all__ = ['save_layered_gallery']
