"""Local-only processor, image transform and artifact identity utilities."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
from .data import INSTRUCTION


def source_commit():
    import subprocess
    root=Path(__file__).resolve().parent.parent
    if (root/'MANIFEST.json').is_file():
        return json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))['source_commit']
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):return 'unavailable'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def picture(path):
    # Exact original optical preprocessing: do NOT introduce EXIF transpose here.
    with Image.open(path) as source:
        image = source.convert('RGB')
    return ImageOps.fit(image,(224,224),method=Image.Resampling.BICUBIC,centering=(.5,.5))


def template(processor, image):
    messages = [{'role':'system','content':[{'type':'text','text':INSTRUCTION}]},
                {'role':'user','content':[{'type':'image','image':image}]}]
    return processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)


def inputs(processor, images, device):
    result = processor(text=[template(processor,i) for i in images],images=images,padding=True,return_tensors='pt')
    return {k:result[k].to(device) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}


def verify_assets(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        target = (path/name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or sha256(target)!=digest:
            raise RuntimeError(f'Missing/changed artifact: {name}')
    return manifest

