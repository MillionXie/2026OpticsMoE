"""Deterministic scene-design previews, not trained-model predictions.

CPU/Pillow renderer using explicit object size, bottom anchor and painter order.
Existing grid datasets and optical models are deliberately untouched.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .layout_preview import SPECS, TASK, font

CODES = {name: code for name, code, _ in SPECS}
BACKGROUND = (250, 250, 247)
# Keep previews self-contained and identical across local/server checkouts.  The
# former fallback pointed at an experiment-local 72 px PNG cache that was not
# versioned, so a clean checkout could not reproduce the approved figures.
SVG_DIR = TASK / 'assets' / 'openmoji-17.0.0-svg'


@dataclass(frozen=True)
class Object:
    id: str
    category: str
    x: int  # center x at model resolution, pixels
    bottom: int
    height: int  # visible alpha bounding-box height, not full PNG height
    z: int


@lru_cache(maxsize=32)
def artwork(category):
    import resvg_py
    data = resvg_py.svg_to_bytes(svg_path=str(SVG_DIR/(CODES[category]+'.svg')), width=1024, height=1024)
    image = Image.open(BytesIO(data)).convert('RGBA')
    return image.crop(image.getchannel('A').getbbox())


def sprite(obj, scale=1):
    image = artwork(obj.category)
    height = obj.height * scale
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def render(objects, scale=1):
    if len({o.id for o in objects}) != len(objects):
        raise ValueError('Object IDs must be unique')
    if len({o.z for o in objects}) != len(objects):
        raise ValueError('Use an explicit unique depth order')
    canvas = Image.new('RGBA', (224*scale, 224*scale), BACKGROUND + (255,))
    masks, bounds = [], []
    ordered = sorted(objects, key=lambda o: o.z)
    for obj in ordered:
        icon = sprite(obj, scale)
        x, y = obj.x*scale-icon.width//2, obj.bottom*scale-icon.height
        if x < 0 or y < 0 or x+icon.width > canvas.width or y+icon.height > canvas.height:
            raise ValueError(f'Clipped object: {obj.id}')
        layer = Image.new('RGBA', canvas.size)
        layer.paste(icon, (x,y))
        canvas = Image.alpha_composite(canvas, layer)
        masks.append(np.asarray(layer.getchannel('A'), dtype=np.float32)/255.)
        bounds.append([x/scale,y/scale,(x+icon.width)/scale,(y+icon.height)/scale])
    rows = []
    for i,obj in enumerate(ordered):
        transmission = np.ones_like(masks[i])
        for later in masks[i+1:]:
            transmission *= 1-later
        visible = float((masks[i]*transmission).sum()/max(masks[i].sum(),1))
        rows.append({**asdict(obj), 'xyxy':bounds[i], 'visible_alpha_fraction':visible})
    return canvas.convert('RGB'), rows


def examples():
    home = [Object('tree','tree',59,173,106,0), Object('house','house',146,175,154,1),
            Object('dog','dog',129,186,29,2), Object('flower','flower',188,190,18,3)]
    grove = [Object('tree','tree',105,186,156,0), Object('bird','bird',154,98,18,1),
             Object('cat','cat',106,190,25,2), Object('flower','flower',177,192,21,3)]
    street = [Object('tree','tree',60,167,110,0), Object('house','house',146,163,136,1),
              Object('car','car',131,184,26,2), Object('bicycle','bicycle',49,196,27,3)]
    return [
        {'id':'home_move','title':'a  Coming home', 'instruction':'Move the dog to the left of the house.',
         'operation':'move','edited_id':'dog','source':home,
         'target':[replace(o,x=57,bottom=188) if o.id=='dog' else o for o in home]},
        {'id':'grove_remove','title':'b  Under the tree', 'instruction':'Remove the cat.',
         'operation':'remove','edited_id':'cat','source':grove,'target':[o for o in grove if o.id!='cat']},
        {'id':'street_replace','title':'c  A quiet street', 'instruction':'Replace the car with a bus.',
         'operation':'replace','edited_id':'car','source':street,
         'target':[replace(o,category='bus',height=27) if o.id=='car' else o for o in street]},
        {'id':'home_add','title':'d  A flower by the door', 'instruction':'Add a flower to the right of the dog.',
         'operation':'add','edited_id':'flower','source':[o for o in home if o.id!='flower'],'target':home},
    ]


def validate_pair(example):
    a={o.id:o for o in example['source']};b={o.id:o for o in example['target']}
    edited=example['edited_id']
    assert {k:v for k,v in a.items() if k!=edited} == {k:v for k,v in b.items() if k!=edited}
    for side in ('source','target'):
        _,rows=render(example[side])
        assert min(r['visible_alpha_fraction'] for r in rows)>=.70, (example['id'], rows)
    if example['operation']=='remove':
        # Reconstruction from complete remaining layers reveals the background,
        # never paint the object's old box with a flat color.
        expected,_=render([o for o in example['source'] if o.id!=edited])
        actual,_=render(example['target'])
        assert np.array_equal(expected,actual)


def main():
    global SVG_DIR
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=TASK/'reports/layered_scene_design_20260920')
    parser.add_argument('--svg-dir',type=Path,default=SVG_DIR,
                        help='Official OpenMoji SVG directory; requires resvg-py')
    args=parser.parse_args();SVG_DIR=args.svg_dir;out=args.output;out.mkdir(parents=True,exist_ok=True)
    proposals=examples()
    board=Image.new('RGB',(1500,1250),'white');d=ImageDraw.Draw(board)
    d.text((24,18),'Scene-scale study | SOURCE / AUTHORED TARGET',font=font(27),fill='#222222')
    d.text((24,56),'Design previews only - not model predictions',font=font(19),fill='#555555')
    metadata=[]
    for i,example in enumerate(proposals):
        validate_pair(example)
        col,row=i%2,i//2;x=24+col*750;y=110+row*565
        d.text((x,y),example['title'],font=font(25),fill='#222222')
        d.text((x,y+38),example['instruction'],font=font(20),fill='#444444')
        record={k:v for k,v in example.items() if k not in ('source','target')}
        for j,side in enumerate(('source','target')):
            img,objects=render(example[side]);record[side]=objects
            img.save(out/f'{example["id"]}_{side}_224.png')
            large,_=render(example[side],4)
            large.save(out/f'{example["id"]}_{side}_896.png')
            board.paste(large.resize((336,336),Image.Resampling.LANCZOS),(x+j*360,y+82))
            d.text((x+j*360,y+434),side.capitalize(),font=font(20),fill='#555555')
        source,_=render(example['source']);target,_=render(example['target'])
        change=np.any(np.asarray(source)!=np.asarray(target),axis=-1)
        Image.fromarray((change*255).astype('uint8')).save(out/f'{example["id"]}_visible_change_mask.png')
        metadata.append(record)
    board.save(out/'01_scene_editing_overview.png')
    # A sequential, unselected geometry preview batch, NOT a benchmark split.
    rng=random.Random(20260920);regular=[]
    sheet=Image.new('RGB',(1008,820),'white');ds=ImageDraw.Draw(sheet)
    ds.text((16,12),'Sequential preview variations | seed 20260920 | no performance selection',font=font(22),fill='#222222')
    for i in range(12):
        original=proposals[i%3]['source']
        dx,dy=rng.randint(-3,3),rng.randint(-3,3)
        objects=[replace(o,x=o.x+dx,bottom=o.bottom+dy) for o in original]
        image,rows=render(objects)
        assert all(o['visible_alpha_fraction']>=.7 for o in rows)
        image.save(out/f'regular_preview_{i:02d}_224.png')
        x,y=16+(i%4)*248,55+(i//4)*252
        sheet.paste(image,(x,y));ds.text((x,y+225),f'{i:02d}',font=font(16),fill='#555555')
        regular.append(rows)
    sheet.save(out/'02_regular_preview_contact_sheet.png')
    report={'status':'preview_only_awaiting_visual_approval','gpu_used':False,
            'renderer':'alpha compositing; fixed bottom anchors; explicit z order',
            'input_size':[224,224], 'assets':'official OpenMoji 17.0.0 SVG rasterized at 1024px before sizing',
            'git_commit_before_run':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'examples':metadata,'regular_preview':regular,
            'asset_sha256':{CODES[o.category]:hashlib.sha256((SVG_DIR/(CODES[o.category]+'.svg')).read_bytes()).hexdigest() for e in proposals for o in e['source']+e['target']}}
    (out/'scene_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'authored_pairs':len(metadata),'sequential_previews':len(regular),
                      'minimum_visible_fraction':min(o['visible_alpha_fraction'] for e in metadata for side in ('source','target') for o in e[side]),
                      'gpu_used':False},indent=2))


if __name__=='__main__':
    main()
