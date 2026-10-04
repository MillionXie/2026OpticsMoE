"""Sealed rank72 native image geometry, no model or device imports."""
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
BASE_CORNERS=np.float32([[586,147],[1379,159],[1369,946],[573,932]])
ACTIVE=478
PHYSICAL=1016
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def camera_variants(a):
    return {'identity':a,'flip_h':np.fliplr(a),'flip_v':np.flipud(a),'rot180':np.rot90(a,2),
            'transpose':a.T,'rot90':np.rot90(a,1),'rot270':np.rot90(a,3),'anti_transpose':np.rot90(a.T,2)}

def warp(frame):
    dst=np.float32([[0,0],[477,0],[477,477],[0,477]])
    matrix=cv2.getPerspectiveTransform(BASE_CORNERS,dst)
    return cv2.warpPerspective(frame,matrix,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def orient(a,name):
    return camera_variants(a)[name].copy()

def active_to_native(active,kind='amplitude'):
    resized=cv2.resize(np.asarray(active), (PHYSICAL,PHYSICAL), interpolation=cv2.INTER_NEAREST)
    h=1080 if kind=='amplitude' else 1200;full=np.zeros((h,1920),np.uint8)
    x=(1920-PHYSICAL)//2;y=(h-PHYSICAL)//2;full[y:y+PHYSICAL,x:x+PHYSICAL]=resized
    return full

def phase_gray(radians,spatial,inverted):
    a=np.mod(np.asarray(radians,dtype=np.float32),2*math.pi)
    g=np.floor(a/(2*math.pi)*256).clip(0,255).astype(np.uint8)
    if 'h' in spatial:g=np.fliplr(g)
    if 'v' in spatial:g=np.flipud(g)
    full=active_to_native(g,'phase')
    return 255-full if inverted else full
