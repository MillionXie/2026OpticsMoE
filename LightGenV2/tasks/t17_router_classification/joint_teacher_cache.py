"""SHA-checked shared frozen visual features for one RGB/SAR teacher."""
import json
from pathlib import Path
import numpy as np
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def load_joint_features(raw,cfg,task):
    if set(cfg['feature_cache_runs'])!={'RGB','SAR'}:raise ValueError('Both feature domains are required')
    features={s:np.empty((len(v['labels']),8,512),dtype=np.float16) for s,v in raw.items()};sources={};backbones=set()
    for domain,number in [('RGB',0),('SAR',1)]:
        folder=task/cfg['feature_cache_runs'][domain]
        meta=json.loads((folder/'metadata.json').read_text());manifest=json.loads((folder/'cache_manifest.json').read_text())
        if meta['original_data']!=cfg['data'] or meta['training_domain']!=domain:raise ValueError('Feature data/domain mismatch')
        backbones.add(meta['backbone_sha256']);sources[domain]={}
        for split,part in raw.items():
            fp=folder/f'features_{split}.npy';ip=folder/f'identities_{split}.npz'
            if any(sha256(p)!=manifest[p.name] for p in (fp,ip)):raise ValueError('Feature cache SHA mismatch')
            keep=part['domains']==number
            with np.load(ip,allow_pickle=False) as identities:
                if any(not np.array_equal(identities[k],part[k][keep]) for k in ('ids','labels','domains')):raise ValueError('Feature identities mismatch')
            value=np.load(fp,allow_pickle=False)
            if value.shape!=(int(keep.sum()),8,512) or not np.isfinite(value).all():raise ValueError('Invalid frozen features')
            features[split][keep]=value;sources[domain][split]=dict(features_sha256=sha256(fp),identities_sha256=sha256(ip),images=int(keep.sum()))
    if len(backbones)!=1:raise ValueError('A joint teacher requires one shared frozen backbone')
    return features,dict(backbone_sha256=next(iter(backbones)),sources=sources)


def domain_metrics(labels,scores,domains):
    from .test_exposed_data import metrics_from_scores
    result=metrics_from_scores(labels,scores)
    result['domains']={name:metrics_from_scores(labels[domains==number],scores[domains==number]) for number,name in [(0,'RGB'),(1,'SAR')]}
    return result


def joint_development_key(metrics):
    if set(metrics.get('domains',{}))!={'RGB','SAR'}:raise ValueError('Both domain development scores are required')
    values=[metrics['domains'][d]['accuracy'] for d in ('RGB','SAR')]
    return min(values),sum(values)/2,-metrics['cross_entropy']
