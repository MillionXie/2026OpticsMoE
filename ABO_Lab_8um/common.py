"""Portable paths and immutable identities; no sister-lab hardware constants."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
STAGES = ('vision_router','vision_expert','vision_global',
          'language_router','language_expert','language_global')
CHECKPOINT_SHA = 'a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,p)

def config(path=None):
    p=Path(path).resolve() if path else ROOT/('LAB.local.json' if (ROOT/'LAB.local.json').exists() else 'lab.json')
    c=read(p)
    if c['model_active_pixels']!=478 or c['model_pitch_um']!=17 or c['distance_m']!=0.1 or c['wavelength_nm']!=532:
        raise ValueError('The fixed checkpoint requires 478 pixels, 17 um, 532 nm, 10 cm; interpolation changes device raster only.')
    return c,p

def setup_imports():
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    for path in (ROOT/'runtime'/'backend',ROOT/'runtime'):
        if not path.is_dir(): raise FileNotFoundError(f'Missing imported source: {path}; run import_release.py first.')
        sys.path.insert(0,str(path))

def model_config():
    return ROOT/'runtime/backend/LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_router_moe_dc20_kd1_balance1_optimized.yaml'

def session_path(name):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',name): raise ValueError('Use a short alphanumeric session ID.')
    return ROOT/'sessions'/name

def hardware_identity(c):
    value=json.loads(json.dumps(c))
    lut=c['amplitude_slm'].get('gray_lut_file')
    if lut: value['amplitude_gray_lut_sha256']=sha(ROOT/lut)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
