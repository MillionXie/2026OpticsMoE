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


def evaluation_checkpoint(assets, checkpoint=None, expected_sha256=None):
    """Select a pinned external best without modifying the packaged assets."""
    if checkpoint is None:
        if expected_sha256 is not None:raise ValueError('Checkpoint SHA requires --checkpoint')
        path=Path(assets)/'best.pt'
        return path.resolve(),sha256(path)
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(c not in '0123456789abcdefABCDEF' for c in expected_sha256):
        raise ValueError('Explicit checkpoint requires its 64-character SHA256')
    path=Path(checkpoint).resolve();actual=sha256(path)
    if actual!=expected_sha256.lower():raise ValueError('Explicit checkpoint SHA256 mismatch')
    return path,actual


def write_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def picture(path, mode='center_crop'):
    # Exact original optical preprocessing: do NOT introduce EXIF transpose here.
    with Image.open(path) as source:
        image = source.convert('RGB')
    if mode == 'center_crop':
        return ImageOps.fit(image,(224,224),method=Image.Resampling.BICUBIC,centering=(.5,.5))
    if mode in ('contain_white','contain_min_half'):
        # Optional input-only control: retain ALL pixels, but bound extreme
        # aspect ratio to2:1. This is anisotropic resizing, not optical geometry.
        if mode=='contain_min_half' and 2*min(image.size)<max(image.size):
            size=(112,224) if image.height>image.width else (224,112)
            resized=image.resize(size,Image.Resampling.BICUBIC)
        else:
            resized = ImageOps.contain(image, (224,224), method=Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (224,224), (255,255,255))
        canvas.paste(resized, ((224-resized.width)//2, (224-resized.height)//2))
        return canvas
    raise ValueError(f'Unknown input preprocessing: {mode}')


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
