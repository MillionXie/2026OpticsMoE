"""Layered OpenMoji v3: fixed semantic anchors, proportional sprites, mild occlusion.

The network target remains a 6x6 category/edit grid. Each occupied cell is an
object bottom-center anchor; category deterministically defines size and depth.
Thus the renderer may span cells and overlap, without passing target geometry to
the model or silently changing the optical/readout dimensions.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch

from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.assets import (
    ICON_SPECS, INDEX_TO_SPEC,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.scenes import (
    PROMPT_TEMPLATES, RELATIONS, RELATION_TEXT, TASKS, TASK_TO_INDEX, prompt_key,
)
from .embedding_data import build_token_cache, identity, sha256

ANCHOR_X = (30, 63, 96, 128, 161, 194)
ANCHOR_BOTTOM = (40, 72, 104, 136, 168, 200)
VISIBLE_HEIGHT = {
    1: 26, 2: 30, 3: 29, 4: 31, 5: 31, 6: 27, 7: 19, 8: 132,
    9: 22, 10: 23, 11: 28, 12: 28, 13: 25, 14: 37, 15: 154, 16: 24,
}
# Stable painter groups; lower object feet are painted later within a group.
DEPTH_GROUP = {8: 0, 15: 0, 7: 1, 14: 1, 3: 2, 4: 2, 2: 3}
BACKGROUND_CATEGORIES = (8, 15)
ANIMAL_CATEGORIES = (5, 6, 7)
VEHICLE_CATEGORIES = (2, 3, 4)
SMALL_CATEGORIES = (1, 9, 10, 11, 12, 13, 14, 16)
MIN_VISIBLE = 0.70
RENDERER_VERSION = 'layered_anchor6_proportional_svg_v3'


@lru_cache(maxsize=32)
def _artwork(svg_dir: str, category: int) -> Image.Image:
    try:
        import resvg_py
    except ImportError as exc:
        raise RuntimeError('Layered SVG preparation requires: pip install resvg-py==0.5.0') from exc
    code = INDEX_TO_SPEC[category].codepoint
    data = resvg_py.svg_to_bytes(
        svg_path=str(Path(svg_dir) / f'{code}.svg'), width=1024, height=1024,
        shape_rendering='geometric_precision', image_rendering='optimize_quality',
    )
    image = Image.open(BytesIO(data)).convert('RGBA')
    return image.crop(image.getchannel('A').getbbox())


@lru_cache(maxsize=32)
def _sprite(svg_dir: Path, category: int) -> Image.Image:
    image = _artwork(str(svg_dir.resolve()), category)
    height = VISIBLE_HEIGHT[category]
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _bounds(svg_dir: Path, category: int, row: int, col: int) -> tuple[int, int, int, int]:
    image = _sprite(svg_dir, category)
    x = ANCHOR_X[col] - image.width // 2
    y = ANCHOR_BOTTOM[row] - image.height
    return x, y, x + image.width, y + image.height


def valid_cell(svg_dir: Path, category: int, row: int, col: int) -> bool:
    x0, y0, x1, y1 = _bounds(svg_dir, category, row, col)
    return 0 <= x0 and 0 <= y0 and x1 <= 224 and y1 <= 224


def render_grid(grid: np.ndarray, svg_dir: Path) -> tuple[Image.Image, list[dict[str, Any]]]:
    if grid.shape != (6, 6):
        raise ValueError('Layered renderer requires a 6x6 semantic anchor grid')
    objects = []
    for row, col in zip(*np.nonzero(grid)):
        category = int(grid[row, col])
        if not valid_cell(svg_dir, category, int(row), int(col)):
            raise ValueError(f'Category {category} clipped at anchor {(int(row), int(col))}')
        objects.append((DEPTH_GROUP.get(category, 4), int(row), int(col), category))
    objects.sort()  # background first; within group, lower anchors are foreground.
    canvas = Image.new('RGBA', (224, 224), (250, 250, 247, 255))
    masks, records = [], []
    for depth, row, col, category in objects:
        sprite = _sprite(svg_dir, category)
        x0, y0, x1, y1 = _bounds(svg_dir, category, row, col)
        layer = Image.new('RGBA', canvas.size)
        layer.paste(sprite, (x0, y0))
        canvas = Image.alpha_composite(canvas, layer)
        masks.append(np.asarray(layer.getchannel('A'), dtype=np.float32) / 255.0)
        records.append({'category': category, 'name': INDEX_TO_SPEC[category].name,
                        'anchor_rc': [row, col], 'depth_group': depth,
                        'visible_height_px': VISIBLE_HEIGHT[category], 'xyxy': [x0, y0, x1, y1]})
    for index, record in enumerate(records):
        transmission = np.ones_like(masks[index])
        for later in masks[index + 1:]:
            transmission *= 1.0 - later
        record['visible_alpha_fraction'] = float(
            (masks[index] * transmission).sum() / max(float(masks[index].sum()), 1.0)
        )
    return canvas.convert('RGB'), records


def _scene_ok(grid: np.ndarray, svg_dir: Path) -> tuple[bool, list[dict[str, Any]]]:
    try:
        _, objects = render_grid(grid, svg_dir)
    except ValueError:
        return False, []
    return bool(objects) and min(o['visible_alpha_fraction'] for o in objects) >= MIN_VISIBLE, objects


def _empty() -> np.ndarray:
    return np.zeros((6, 6), dtype=np.uint8)


def _place(grid: np.ndarray, category: int, cells: list[tuple[int, int]], rng: random.Random, svg_dir: Path) -> bool:
    choices = [(r, c) for r, c in cells if grid[r, c] == 0 and valid_cell(svg_dir, category, r, c)]
    rng.shuffle(choices)
    for row, col in choices:
        candidate = grid.copy(); candidate[row, col] = category
        if _scene_ok(candidate, svg_dir)[0]:
            grid[row, col] = category
            return True
    return False


def _base_scene(rng: random.Random, svg_dir: Path) -> tuple[np.ndarray, str]:
    family = rng.choices(('home', 'grove', 'street', 'still_life'), weights=(3, 3, 3, 2))[0]
    grid = _empty()
    lower = [(r, c) for r in (4, 5) for c in range(6)]
    central = [(r, c) for r in range(1, 6) for c in range(1, 5)]
    if family == 'home':
        _place(grid, 15, [(4, 2), (4, 3), (5, 2), (5, 3)], rng, svg_dir)
        if rng.random() < .75:
            _place(grid, 8, [(4, 1), (4, 4), (5, 1), (5, 4)], rng, svg_dir)
        pool = ANIMAL_CATEGORIES + (9, 2, 1, 16)
    elif family == 'grove':
        _place(grid, 8, [(4, 2), (4, 3), (5, 2), (5, 3)], rng, svg_dir)
        pool = ANIMAL_CATEGORIES + (9, 13, 14, 16)
    elif family == 'street':
        _place(grid, rng.choice(BACKGROUND_CATEGORIES), [(4, 2), (4, 3), (5, 2), (5, 3)], rng, svg_dir)
        pool = VEHICLE_CATEGORIES + (8, 15, 5, 6)
    else:
        pool = SMALL_CATEGORIES + ANIMAL_CATEGORIES + VEHICLE_CATEGORIES
    target_count = rng.randint(3, 5)
    attempts = 0
    while np.count_nonzero(grid) < target_count and attempts < 100:
        category = rng.choice(pool); attempts += 1
        if category in grid:
            continue
        cells = lower if category in BACKGROUND_CATEGORIES else central + lower
        _place(grid, category, cells, rng, svg_dir)
    if np.count_nonzero(grid) < 3 or not _scene_ok(grid, svg_dir)[0]:
        return _base_scene(rng, svg_dir)
    return grid, family


def _prompt(rng: random.Random, task: str, **values: str) -> str:
    return rng.choice(PROMPT_TEMPLATES[task]).format(**values)


def generate_example(task: str, seed: int, svg_dir: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    for _ in range(2000):
        source, family = _base_scene(rng, svg_dir)
        occupied = list(zip(*np.nonzero(source)))
        target = source.copy(); program: dict[str, Any]
        if task == 'add':
            reference = rng.choice(occupied)
            candidates = []
            for relation, (dr, dc) in RELATIONS.items():
                row, col = reference[0] + dr, reference[1] + dc
                if 0 <= row < 6 and 0 <= col < 6 and target[row, col] == 0:
                    candidates.append((relation, row, col))
            rng.shuffle(candidates)
            absent = [s.index for s in ICON_SPECS if s.index not in source]
            rng.shuffle(absent)
            found = None
            for relation, row, col in candidates:
                for category in absent:
                    candidate = target.copy(); candidate[row, col] = category
                    if valid_cell(svg_dir, category, row, col) and _scene_ok(candidate, svg_dir)[0]:
                        found = relation, row, col, category, candidate; break
                if found: break
            if not found: continue
            relation, row, col, category, target = found
            refcat = int(source[reference])
            instruction = _prompt(rng, task, new=INDEX_TO_SPEC[category].name,
                                  relation=RELATION_TEXT[relation], reference=INDEX_TO_SPEC[refcat].name)
            program = {'operation':task, 'new_category':category, 'reference_anchor':list(reference),
                       'relation':relation, 'new_anchor':[row,col]}
        elif task == 'replace':
            selected = rng.choice(occupied); old = int(source[selected])
            absent = [s.index for s in ICON_SPECS if s.index not in source]; rng.shuffle(absent)
            found = None
            for category in absent:
                candidate=target.copy();candidate[selected]=category
                if valid_cell(svg_dir, category, *selected) and _scene_ok(candidate,svg_dir)[0]:
                    found=category,candidate;break
            if not found: continue
            category,target=found
            instruction=_prompt(rng,task,old=INDEX_TO_SPEC[old].name,new=INDEX_TO_SPEC[category].name)
            program={'operation':task,'target_anchor':list(selected),'old_category':old,'new_category':category}
        elif task == 'move':
            selected = rng.choice(occupied); reference = rng.choice([p for p in occupied if p != selected])
            category=int(source[selected]); candidates=[]
            for relation,(dr,dc) in RELATIONS.items():
                row,col=reference[0]+dr,reference[1]+dc
                if 0<=row<6 and 0<=col<6 and (target[row,col]==0 or (row,col)==selected):
                    candidates.append((relation,row,col))
            rng.shuffle(candidates);found=None
            for relation,row,col in candidates:
                if (row,col)==selected: continue
                candidate=target.copy();candidate[selected]=0;candidate[row,col]=category
                if valid_cell(svg_dir,category,row,col) and _scene_ok(candidate,svg_dir)[0]:
                    found=relation,row,col,candidate;break
            if not found:continue
            relation,row,col,target=found;refcat=int(source[reference])
            instruction=_prompt(rng,task,target=INDEX_TO_SPEC[category].name,
                                relation=RELATION_TEXT[relation],reference=INDEX_TO_SPEC[refcat].name)
            program={'operation':task,'target_anchor':list(selected),'reference_anchor':list(reference),
                     'relation':relation,'new_anchor':[row,col]}
        elif task == 'remove':
            selected=rng.choice(occupied);category=int(source[selected]);target[selected]=0
            if not _scene_ok(target,svg_dir)[0]:continue
            instruction=_prompt(rng,task,target=INDEX_TO_SPEC[category].name)
            program={'operation':task,'target_anchor':list(selected),'category':category}
        else:
            raise ValueError(task)
        _, source_objects=render_grid(source,svg_dir);_,target_objects=render_grid(target,svg_dir)
        return {'task':task,'task_index':TASK_TO_INDEX[task],'seed':seed,'family':family,
                'instruction':instruction,'prompt_key':prompt_key(instruction),'program':program,
                'source_grid':source,'target_grid':target,'edit_grid':(source!=target).astype(np.uint8),
                'source_objects':source_objects,'target_objects':target_objects}
    raise RuntimeError(f'Unable to generate valid {task} scene at seed {seed}')


def _save(example: dict[str, Any], directory: Path, svg_dir: Path) -> dict[str, Any]:
    directory.mkdir(parents=True,exist_ok=True)
    source,_=render_grid(example['source_grid'],svg_dir);target,_=render_grid(example['target_grid'],svg_dir)
    source.save(directory/'source.png',optimize=True);target.save(directory/'target.png',optimize=True)
    changed=np.any(np.asarray(source)!=np.asarray(target),axis=-1)
    Image.fromarray((changed*255).astype(np.uint8)).save(directory/'visible_change_mask.png',optimize=True)
    # Existing trainer consumes semantic-anchor edit_mask, not the visible pixel mask.
    anchor=np.zeros((224,224),dtype=np.uint8)
    boundaries=[round(i*224/6) for i in range(7)]
    for row,col in zip(*np.nonzero(example['edit_grid'])):
        anchor[boundaries[row]:boundaries[row+1],boundaries[col]:boundaries[col+1]]=255
    Image.fromarray(anchor).save(directory/'edit_mask.png',optimize=True)
    def jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    metadata={k:jsonable(v) for k,v in example.items()}
    metadata['files']={'source':'source.png','target':'target.png','edit_mask':'edit_mask.png',
                       'visible_change_mask':'visible_change_mask.png'}
    (directory/'scene.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


def prepare_layered_embedding_data(settings: Any) -> dict[str, Any]:
    root=settings.data_dir;svg_dir=settings.svg_asset_dir;root.mkdir(parents=True,exist_ok=True)
    for spec in ICON_SPECS:
        if not (svg_dir/f'{spec.codepoint}.svg').is_file():
            raise FileNotFoundError(f'Missing official SVG: {spec.codepoint}.svg')
    kind='openmoji_layered_anchor6_proportional_svg_v3'
    summary_path=root/'dataset_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('type')!=kind or summary.get('seed')!=settings.seed:
            raise ValueError('Incompatible dataset; use a new empty data directory')
    else:
        if any(root.iterdir()):
            raise RuntimeError('Incomplete layered dataset directory; inspect it and use a new empty directory')
        seen=set();summary={'type':kind,'seed':settings.seed,'renderer_version':RENDERER_VERSION,
                            'validation_split':False,'svg_asset_dir':str(svg_dir),
                            'anchors':{'x':ANCHOR_X,'bottom':ANCHOR_BOTTOM},
                            'visible_height_px':VISIBLE_HEIGHT,'minimum_visible_alpha_fraction':MIN_VISIBLE}
        for split,count,offset in [('train',settings.train_samples,0),('test',settings.test_samples,20_000_000)]:
            records=[];rejected=0;task_counts=Counter();family_counts=Counter();category_counts=Counter();occluded=0
            for index in range(count):
                task=TASKS[index%4]
                for attempt in range(10000):
                    example=generate_example(task,settings.seed*1_000_003+offset+index+attempt*100_000_000,svg_dir)
                    key=identity(example)
                    if key not in seen:seen.add(key);break
                    rejected+=1
                sample_id=f'{split}_{index:06d}';relative=Path(split)/sample_id
                metadata=_save(example,root/relative,svg_dir)
                record={'sample_id':sample_id,'split':split,'relative_dir':relative.as_posix(),**metadata}
                records.append(record);task_counts[task]+=1;family_counts[example['family']]+=1
                for obj in example['source_objects']:
                    category_counts[obj['name']]+=1
                    occluded += obj['visible_alpha_fraction'] < .995
            manifest=root/f'{split}.jsonl'
            manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
            summary[split]={'samples':count,'task_counts':dict(task_counts),'family_counts':dict(family_counts),
                            'source_category_counts':dict(category_counts),'occluded_source_objects':occluded,
                            'duplicates_resampled':rejected,'sha256':sha256(manifest)}
        summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    for split in ('train','test'):
        manifest=root/f'{split}.jsonl'
        if sha256(manifest)!=summary[split]['sha256']:raise ValueError('Layered dataset manifest changed')
        records.extend(json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines())
    if len({identity(r) for r in records})!=len(records):raise ValueError('Duplicate source-grid/instruction')
    build_token_cache(settings,records,summary)
    return summary


__all__=['prepare_layered_embedding_data','generate_example','render_grid','valid_cell']
