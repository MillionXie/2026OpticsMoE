"""Original panel-F two-domain subset; no replacement with joint-modal G inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def load_panel_f(cfg):
    result={};groups={};identity={}
    for name,prefixes in [('trainval',[('train','train'),('val','validation')]),('holdout',[('test','test')])]:
        source=cfg[name];data=Path(source['data']);manifest_path=Path(source['manifest'])
        if sha256(data)!=source['data_sha256'] or sha256(manifest_path)!=source['manifest_sha256']:
            raise ValueError('Original F archive/manifest SHA mismatch')
        manifest=json.loads(manifest_path.read_text());rows=manifest['records']
        if manifest['original_split_sha256']!='cfe8373dd33cc0fe64f083b9ca32377e767c21b078c3f1d91f2dacecc25cb776':raise ValueError('F original spatial split changed')
        with np.load(data,allow_pickle=False) as z:
            for split,prefix in prefixes:
                part={k:z[prefix+'_'+k].copy() for k in ['images','labels','domains','ids']}
                selected=[r for r in rows if r['split']==prefix] if name=='trainval' else rows
                if len(part['labels'])!=cfg['expected_counts'][split] or len(selected)!=len(part['labels']):raise ValueError('Original F count changed')
                if not np.array_equal(part['domains'].reshape(-1,2),np.tile([0,1],(len(selected)//2,1))):raise ValueError('RGB/SAR pair order')
                if not np.array_equal(part['labels'][::2],part['labels'][1::2]):raise ValueError('Pair labels')
                pairs=np.char.rpartition(part['ids'].astype(str),':')[:,0]
                if not np.array_equal(pairs[::2],pairs[1::2]):raise ValueError('Pair location IDs')
                for image,label,domain,pair,row in zip(part['images'],part['labels'],part['domains'],pairs,selected):
                    if (int(label),int(domain),str(pair))!=(row['label'],row['domain'],row['pair_id']):raise ValueError('F row identity changed')
                    if hashlib.sha256(image.tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('F pixel hash changed')
                result[split]=part;groups[split]={r['spatial_group'] for r in selected}
        identity[name]=dict(data_sha256=sha256(data),manifest_sha256=sha256(manifest_path),original_split_sha256=manifest['original_split_sha256'])
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if groups[a]&groups[b]:raise ValueError('Original spatial groups overlap')
        if set(result[a]['ids'])&set(result[b]['ids']):raise ValueError('Original IDs overlap')
    identity.update(task='same ten-class land-cover prediction, individual RGB or SAR input',groups_disjoint=True,
                    paired_modalities_not_jointly_input=True,counts={s:len(p['labels']) for s,p in result.items()})
    return result,identity
