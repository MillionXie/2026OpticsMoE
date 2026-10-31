"""Fast raw access with exact saved row order, labels and pixel hashes preserved."""
import hashlib
from pathlib import Path
import numpy as np
from .train_linear_ccd_8class import prepare_index
from .test_exposed_data import load_raw_splits
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def align_raw_pixels(images, labels, rows, ids):
    if len(images)!=len(rows) or len(labels)!=len(rows):raise ValueError('Raw count mismatch')
    expected=np.array([int(r['class_id']) for r in rows])
    if not np.array_equal(np.asarray(labels).reshape(-1),expected):raise ValueError('Raw label order mismatch')
    for image,row in zip(images,rows):
        if hashlib.sha256(image.tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('Raw pixel order/hash mismatch')
    return dict(images=images,labels=expected,ids=ids,hashes=np.array([r['pixel_sha256'] for r in rows]))


def load_raw(data_root,cfg):
    if cfg['dataset']=='crc9':return load_raw_splits(data_root,cfg)
    rows,ids,identity=prepare_index(data_root,cfg);result={}
    cache=Path(cfg['raw_cache'])
    if sha256(cache)!=cfg['raw_cache_sha256']:raise ValueError('Raw cache SHA differs')
    with np.load(cache,allow_pickle=False) as z:
        for split in ('train','val','test'):
            if cfg['dataset']=='bloodmnist128':
                indices=np.array([int(r['source_index']) for r in rows[split]])
                images=z[split+'_images'][indices];labels=z[split+'_labels'][indices]
            else:
                images=z[split+'_images'];labels=z[split+'_labels']
            result[split]=align_raw_pixels(images,labels,rows[split],ids[split])
    identity.update(raw_cache_sha256=cfg['raw_cache_sha256'],raw_pixel_and_label_alignment_checked=True)
    return result,identity
