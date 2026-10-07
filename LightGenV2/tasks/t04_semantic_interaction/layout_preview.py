"""CPU-only design study. Never changes training data, checkpoints or labels.

The proposed scenes are authored design examples, NOT model predictions.
Run: python -m LightGenV2.tasks.t04_semantic_interaction.layout_preview
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

from PIL import Image, ImageDraw, ImageFont

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
ASSETS = REPO / 'experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/assets/openmoji-17.0.0-72-color'
# Design sizes are longest visible side at 224px input, NOT physical measurements.
# Small everyday objects retain visibility; real-world scale is compressed.
SPECS = [
    ('apple', '1F34E', 22), ('bicycle', '1F6B2', 34), ('car', '1F697', 38),
    ('bus', '1F68C', 42), ('dog', '1F415', 30), ('cat', '1F408', 27),
    ('bird', '1F426', 22), ('tree', '1F333', 40), ('flower', '1F33B', 24),
    ('cup', '2615', 22), ('book', '1F4D8', 25), ('phone', '1F4F1', 23),
    ('ball', '26BD', 23), ('umbrella', '2602', 32), ('house', '1F3E0', 42),
    ('light bulb', '1F4A1', 22),
]
STYLES = ('legacy6', 'balanced6', 'spacious4')


def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render(objects, style, scale=1):
    """objects: (class index 0..15, row, col). No recentering after editing."""
    n = 4 if style == 'spacious4' else 6
    margin, step = (40, 48) if n == 4 else (32, 32)
    canvas = Image.new('RGB', (224 * scale, 224 * scale), (250, 250, 247))
    boxes = []
    for category, row, col in objects:
        assert 0 <= row < n and 0 <= col < n
        name, code, size = SPECS[category]
        icon = Image.open(ASSETS / (code + '.png')).convert('RGBA')
        if style == 'legacy6':
            dimensions = (30, 30)
        else:
            icon = icon.crop(icon.getchannel('A').getbbox())
            size = round(size * .64) if n == 6 else size
            factor = size / max(icon.size)
            dimensions = tuple(max(1, round(v * factor)) for v in icon.size)
        w, h = dimensions
        x, y = margin + col * step - w // 2, margin + row * step - h // 2
        icon = icon.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        canvas.paste(icon, (x * scale, y * scale), icon)
        boxes.append({'name': name, 'xyxy': [x, y, x+w, y+h]})
    return canvas, boxes


def audit():
    rng = random.Random(20260920)
    results = {}
    for style in STYLES:
        n = 4 if style == 'spacious4' else 6
        for _ in range(128):
            cells = rng.sample([(r, c) for r in range(n) for c in range(n)], 5)
            objects = [(k, *cell) for k, cell in zip(rng.sample(range(16), 5), cells)]
            _, boxes = render(objects, style)
            for i, a in enumerate(boxes):
                x0,y0,x1,y1 = a['xyxy']
                assert min(x0,y0) >= 0 and max(x1,y1) <= 224
                for b in boxes[i+1:]:
                    u0,v0,u1,v1 = b['xyxy']
                    assert x1 <= u0 or u1 <= x0 or y1 <= v0 or v1 <= y0
        results[style] = {'random_layouts': 128, 'max_objects': 5, 'overlaps': 0, 'clipped': 0}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=TASK / 'reports/layout_design_20260920')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # Same conceptual compositions across proposals; 4->6 coordinate mapping
    # is for design comparison only, never reused to relabel benchmark samples.
    scenes = [
        ('Street', [(7,1,1),(14,1,2),(1,2,1),(2,2,2)]),
        ('Everyday objects', [(10,1,1),(11,1,2),(0,2,1),(9,2,2)]),
        ('Animals', [(6,1,1),(7,1,2),(5,2,1),(4,2,2)]),
        ('Mixed-scale stress test', [(3,1,1),(0,1,2),(15,2,1),(14,2,2)]),
    ]
    board = Image.new('RGB', (1500, 2090), 'white')
    draw = ImageDraw.Draw(board)
    titles = ['Original 6 x 6 / equal icon boxes', 'A: 6 x 6 / restrained scale hierarchy', 'B: 4 x 4 / more breathing room']
    for j, title in enumerate(titles):
        draw.text((18+j*500,20), title, font=font(21), fill='#222222')
        for i,(label,objects) in enumerate(scenes):
            mapped = objects if j == 2 else [(k, r+1, c+1) for k,r,c in objects]
            image, _ = render(mapped, STYLES[j], 2)
            board.paste(image, (20+j*500,85+i*500))
            draw.text((20+j*500,550+i*500), label, font=font(20), fill='#444444')
            render(mapped, STYLES[j])[0].save(out / f'{STYLES[j]}_{i}_input224.png')
    board.save(out / '01_layout_comparison.png')
    # Before / authored target, never inference.
    examples = [
        ('Add a flower below the house.', [(7,1,1),(14,1,2),(1,2,1)], [(7,1,1),(14,1,2),(1,2,1),(8,2,2)]),
        ('Replace the car with a bus.', scenes[0][1], [(7,1,1),(14,1,2),(1,2,1),(3,2,2)]),
        ('Move the cat below the tree.', [(6,1,0),(7,1,2),(5,2,1)], [(6,1,0),(7,1,2),(5,2,2)]),
        ('Remove the phone.', scenes[1][1], [(10,1,1),(0,2,1),(9,2,2)]),
    ]
    pairs = Image.new('RGB', (1050, 2250), 'white')
    d = ImageDraw.Draw(pairs)
    d.text((25,15), 'B proposal | source and authored target (NOT model predictions)', font=font(24), fill='#222222')
    for i,(instruction,source,target) in enumerate(examples):
        y=75+i*540
        d.text((25,y), instruction, font=font(24), fill='#222222')
        for j,(name,objects) in enumerate([('Source',source),('Target',target)]):
            img,_=render(objects,'spacious4',2)
            pairs.paste(img,(25+j*520,y+40))
            d.text((25+j*520,y+492),name,font=font(20),fill='#444444')
    pairs.save(out/'02_edit_examples_NOT_predictions.png')
    atlas = Image.new('RGB', (1000, 700), 'white')
    d = ImageDraw.Draw(atlas)
    d.text((20,15), 'B: all 16 categories | compressed size hierarchy, not metric scale', font=font(22), fill='#222222')
    for k,(name,code,size) in enumerate(SPECS):
        icon = Image.open(ASSETS/(code+'.png')).convert('RGBA')
        icon = icon.crop(icon.getchannel('A').getbbox())
        factor=2*size/max(icon.size)
        icon=icon.resize(tuple(round(v*factor) for v in icon.size),Image.Resampling.LANCZOS)
        x,y=125+(k%4)*250,105+(k//4)*165
        atlas.paste(icon,(x-icon.width//2,y-icon.height//2),icon)
        d.text((x-85,y+52),f'{name}: {size}px',font=font(20),fill='#444444')
    atlas.save(out/'03_all_categories_size_study.png')
    results={'status':'design_preview_only_awaiting_user_approval','gpu_used':False,
             'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'sizes_are_design_choices_not_measured_physical_dimensions':True,
             'audit':audit(), 'size_table':SPECS,
             'assets_sha256':{code:hashlib.sha256((ASSETS/(code+'.png')).read_bytes()).hexdigest() for _,code,_ in SPECS}}
    (out/'preview_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'audit':results['audit']},indent=2))


if __name__ == '__main__':
    main()
