"""Deterministic stratified groups: never select examples by model correctness."""
import hashlib
import json

import numpy as np


def select_groups(strata, groups_needed, seed=20261024):
    strata = np.asarray(strata).astype(str)
    if not 0 < groups_needed <= len(strata):
        raise ValueError('Invalid group budget')
    names = sorted(set(strata.tolist()))
    if groups_needed < len(names):
        raise ValueError('Budget cannot cover every stratum')
    rng = np.random.default_rng(seed)
    pools = {s:rng.permutation(np.flatnonzero(strata==s)).tolist() for s in names}
    chosen = []
    while len(chosen) < groups_needed:
        for name in names:
            if pools[name]:
                chosen.append(pools[name].pop())
                if len(chosen)==groups_needed:
                    break
    return np.sort(chosen)


def source_metadata(data, task):
    """data is the FULL original T16 wrapper without its optional visual cache.

    In particular Speech/Physical negative candidates must already have been
    constructed using the complete split before selecting these indices.
    """
    if task == 'eurosat':
        base = data.data
        return np.asarray(base.labels).astype(str), np.asarray(base.pair_ids).astype(str), 1
    if task == 'clevr':
        if not np.all(data.labels.reshape(-1,2)==[1,0]):
            raise ValueError('Incomplete CLEVR positive/negative pairs')
        tokens = data.token_ids[::2]
        colors = [5,10,11,12,13,14,15,16]
        shapes = [6,8,9]
        strata = []
        for row in tokens:
            color = [v for v in colors if v in row]
            shape = [v for v in shapes if v in row]
            if len(color)!=1 or len(shape)!=1:
                raise ValueError('Unrecognized original color/shape query')
            strata.append(f'{color[0]}:{shape[0]}')
        return np.array(strata), data.image_index[::2].astype(str), 2
    if task == 'speech_binary':
        rows = data.base.rows
        return (np.array([str(r['audio_class']) for r in rows]),
                np.array([str(r['image_id']) for r in rows]), 2)
    if task == 'physical_binary_raw':
        rows = data.base.rows
        # Original per-concept local record identity is also bound by the
        # source-protocol hash; never merge records across concepts.
        return (data.base.labels().astype(str),
                np.array([f"{r['concept']}:{i}" for i,r in enumerate(rows)]),2)
    raise ValueError('Unknown original T16 task')


def make_manifest(data, task, *, split, source_protocol_sha256, count=1000):
    strata, ids, unit = source_metadata(data,task)
    if count % unit:
        raise ValueError('Paired tasks require an even record budget')
    chosen = select_groups(strata,min(count//unit,len(strata)))
    indices = (chosen[:,None]*unit + np.arange(unit)).reshape(-1)
    labels = np.asarray(data.labels)[indices]
    payload = dict(schema='T21_stratified_source_records_v1',task=task,split=split,
                   seed=20261024,source_protocol_sha256=source_protocol_sha256,
                   source_indices=indices.tolist(),group_ids=ids[chosen].tolist(),
                   strata={str(s):int((strata[chosen]==s).sum()) for s in sorted(set(strata))},
                   class_counts=np.bincount(labels,minlength=10 if task=='eurosat' else 2).tolist(),
                   pairing_complete=True,selection_uses_model_outputs=False)
    payload['identity_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload
