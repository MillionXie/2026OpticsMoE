"""Independent local pairing, class, duplicate, and near-neighbour split checks."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
records=read('SPLIT.json')['records'];plan=read('PLAN.json');audit=read('SPLIT_AUDIT.json');checks=read('DATA_CHECKS.json');pairs=audit['pairs']
assert checks['passed'] and len(records)==2*len(pairs)==checks['images']
assert hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['split_records_sha256']
by_pair={};by_group={};by_grid={}
for r in records:
    by_pair.setdefault(r['pair_id'],[]).append(r)
    by_group.setdefault(r['spatial_group'],set()).add(r['split'])
    by_grid.setdefault(r['grid_id'],set()).add(r['split'])
assert all(len(x)==1 for x in by_group.values()) and all(len(x)==1 for x in by_grid.values())
for pair,rs in by_pair.items():
    assert len(rs)==2 and {r['domain'] for r in rs}=={'A','B'}
    assert len({(r['split'],r['label'],r['spatial_group']) for r in rs})==1
for d in ('A','B'):
    for s in ('train','validation','test'):
        selected=[r for r in records if r['domain']==d and r['split']==s]
        assert len(selected)==plan['counts'][d][s]
        assert [sum(r['label']==c for r in selected) for c in range(10)]==audit['class_counts_per_domain'][s]
assert plan['counts']['A']==plan['counts']['B']
# Independent spatial hash implementation, rather than the preparation KD tree.
buckets={};compared=0
for p in pairs:
    bx,by=math.floor(p['x_m']/3000),math.floor(p['y_m']/3000)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for q in buckets.get((bx+dx,by+dy),[]):
                if (p['x_m']-q['x_m'])**2+(p['y_m']-q['y_m'])**2<3000**2:
                    compared+=1;assert p['split']==q['split'],(p['pair_id'],q['pair_id'])
    buckets.setdefault((bx,by),[]).append(p)
assert not checks['cross_split_or_label_duplicates']
assert read('DATA_LICENSES.json')['verified'] and read('USER_AUTHORIZATION.json')['approved']
vendor=read('VENDOR_MANIFEST.json');old=ROOT.parent/'CORe50_Training_20260912'
assert all((ROOT/name).read_bytes()==(old/name).read_bytes() for name in vendor)
result=dict(passed=True,paired_samples=len(pairs),images=len(records),counts=plan['counts'],spatial_groups=len(by_group),independent_near_pair_checks=compared,all_near_pairs_same_split=True,unchanged_architecture_vendor_files=len(vendor),source_archive_checksums=checks['archive_checksums'])
(ROOT/'PRELAUNCH_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='source_archive_checksums'},indent=2))
