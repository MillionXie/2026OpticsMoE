"""Freeze a conservative date-camera-grouped split, preserving the author fold."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from .audit_deepweeds import digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads((a.data/'data_manifest.json').read_text())
    rows = sorted(json.loads((a.data/'image_manifest.json').read_text()), key=lambda r:r['sample_id'])
    candidates = json.loads((a.audit/'cross_split_candidates.json').read_text())
    audit_config = json.loads((a.audit/'config.json').read_text())
    assert digest(a.data/'image_manifest.json') == audit_config['manifest_sha256']
    assert json.loads((a.audit/'audit.json').read_text())['state'] == 'complete'
    ids = [r['sample_id'] for r in rows]
    def group(name):
        day, time, camera = Path(name).stem.split('-')
        return day+'_'+camera
    parent = {group(name):group(name) for name in ids}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for pair in candidates:
        x, y = sorted([root(group(pair['a'])), root(group(pair['b']))])
        parent[y] = x
    groups = np.array([root(group(name)) for name in ids])
    labels = np.array([r['label'] for r in rows])
    fold = np.full(len(rows), -1)
    for i, (_, index) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=17).split(np.zeros(len(rows)), labels, groups)):
        fold[index] = i
    # Fixed fold assignment, no seed/fold search or model metrics.
    indices = dict(test=np.flatnonzero(fold==0), val=np.flatnonzero(fold==1), train=np.flatnonzero(fold>=2))
    assignment = {ids[i]:split for split, index in indices.items() for i in index}
    assert len(assignment)==len(rows)==len(set(ids))
    for p in candidates:
        assert assignment[p['a']]==assignment[p['b']]
    sets = {s:set(groups[ix]) for s,ix in indices.items()}
    assert not sets['train'] & sets['val'] and not sets['train'] & sets['test'] and not sets['val'] & sets['test']
    supports = {s:np.bincount(labels[ix], minlength=len(manifest['classes'])).tolist() for s,ix in indices.items()}
    assert all(min(v)>0 for v in supports.values())
    source = a.data/'deepweeds_fold0.npz'
    assert digest(source)==manifest['cache_sha256']
    a.out.mkdir(parents=True, exist_ok=False)
    save = lambda name, obj:(a.out/name).write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')
    record = dict(protocol='deepweeds_date_camera_grouped_s17_v1', seed=17, groups=len(set(groups)),
        grouping='date + camera; union groups linked by all audited cross-split dHash<=4 candidates',
        folds='StratifiedGroupKFold 5: test=0, val=1, train=2/3/4; no search',
        supports=supports, split_group_counts={s:len(g) for s,g in sets.items()},
        known_candidate_cross_split_pairs=0, sklearn_version=sklearn.__version__,
        manifest_sha256=digest(a.data/'image_manifest.json'), candidate_sha256=digest(a.audit/'cross_split_candidates.json'),
        source_cache_sha256=manifest['cache_sha256'], git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=sys.argv, limitation='Date-camera groups are scene proxies, not verified plant or site IDs; no claim of complete biological independence.')
    save('split_protocol.json', record)
    save('group_assignment.json', [dict(sample_id=ids[i],group=groups[i],split=assignment[ids[i]],label=int(labels[i])) for i in range(len(rows))])
    print(json.dumps(record), flush=True)
    arrays = {}
    # Read each original split once and reindex by immutable sample IDs.
    with np.load(source, allow_pickle=False) as z:
        source_ids = np.concatenate([z[s+'_ids'] for s in ['train','val','test']]).astype(str)
        source_images = np.concatenate([z[s+'_images'] for s in ['train','val','test']])
        source_labels = np.concatenate([z[s+'_labels'] for s in ['train','val','test']])
        lookup = {name:i for i,name in enumerate(source_ids)}
        for split,index in indices.items():
            take = np.array([lookup[ids[i]] for i in index])
            assert np.array_equal(source_labels[take], labels[index])
            arrays[split+'_images'] = source_images[take]
            arrays[split+'_labels'] = labels[index]
            arrays[split+'_ids'] = np.array(ids)[index]
    cache = a.out/'deepweeds_grouped_s17.npz'
    np.savez(cache, **arrays)
    manifest.update(cache_sha256=digest(cache), split_ids={s:arrays[s+'_ids'].tolist() for s in indices},
        supports=supports, training_ready=True, scene_audit_pending=False,
        split_protocol=record, verified_plant_ids=False, official_fold=False)
    save('data_manifest.json', manifest)
    print('Grouped cache complete and training-ready under documented proxy-group protocol.', flush=True)


if __name__=='__main__':
    main()
