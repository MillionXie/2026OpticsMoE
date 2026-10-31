"""Panel D keeps the original Organ split and reuses A pathology identities."""
import hashlib,json
from pathlib import Path
import numpy as np
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def load_organ(cfg):
    source=Path(cfg['source_archive']);cache=Path(cfg['raw_cache']);manifest_path=Path(cfg['organ_manifest'])
    if sha256(source)!=cfg['source_archive_sha256'] or sha256(cache)!=cfg['raw_cache_sha256'] or sha256(manifest_path)!=cfg['organ_manifest_sha256']:raise ValueError('Organ source/cache/manifest SHA mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest['cache_sha256']!=cfg['raw_cache_sha256'] or manifest['source_sha256']!=cfg['source_archive_sha256']:raise ValueError('Organ provenance mismatch')
    result={}
    with np.load(source,allow_pickle=False) as raw,np.load(cache,allow_pickle=False) as old:
        for split in ('train','val','test'):
            grey=raw[split+'_images'];labels=raw[split+'_labels'].reshape(-1).astype(np.int64);images=np.repeat(grey[:,:,:,None],3,axis=3)
            ids=np.array([f'organcmnist:{split}:{i:06d}' for i in range(len(labels))])
            if len(labels)!=cfg['expected_counts'][split] or set(labels.tolist())!=set(range(11)):raise ValueError('Organ count/classes mismatch')
            if split!='test':
                if not np.array_equal(images,old[split+'_images']) or not np.array_equal(labels,old[split+'_labels']) or not np.array_equal(ids,old[split+'_ids']):raise ValueError('Organ original TRAIN/VAL pixels/IDs differ')
                if ids.tolist()!=manifest['split_ids'][split]:raise ValueError('Organ split IDs mismatch')
            result[split]=dict(images=images,labels=labels,ids=ids,hashes=np.array([hashlib.sha256(v.tobytes()).hexdigest() for v in images]))
    return result,dict(source_archive_sha256=cfg['source_archive_sha256'],raw_cache_sha256=cfg['raw_cache_sha256'],manifest_sha256=cfg['organ_manifest_sha256'],classes=manifest['classes'],license=manifest['license'],counts=cfg['expected_counts'],original_TRAIN_VAL_pixels_labels_IDs_exact=True,original_official_TEST_all_retained=True,patient_independence_verified=manifest.get('patient_independence_verified',False))


def validate_panel_d_initialization(payload,cfg):
    if cfg.get('dataset') not in ('kather2016','bloodmnist128') or cfg.get('architecture')!='LG' or cfg.get('main_layers')!=2 or cfg.get('top_k') not in (1,3,4) or cfg.get('teacher'):raise ValueError('A initialization is only for approved D top-k points')
    prior=payload['metadata']['profile']
    for key in ('dataset','num_classes','main_layers','source_archive_sha256','raw_cache_sha256','expected_counts'):
        if prior.get(key)!=cfg.get(key):raise ValueError('Panel D/A parent identity mismatch: '+key)
