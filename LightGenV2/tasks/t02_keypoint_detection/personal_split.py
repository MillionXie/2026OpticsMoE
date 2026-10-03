"""Freeze visually audited capture groups BEFORE any augmentation or model fitting."""
import argparse,json,random,math,hashlib
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from .personal_prepare import sha256

# Indices in the sorted 101-photo ingest, manually checked on all five contact sheets.
# Same burst / two viewpoints / obviously near-identical pose are indivisible.
GROUPS=[[0,7,8,9,10,47,48],[1,99,100],[2,45,46],[3,29,37],[4,5],[6,32,33],
        [11,12,13,28],[14,15,16,26,27,56,57],[17,18,19,20,21,22,30,31,51,52],
        [23,24,25],[34,35],[36],[38,39],[40,41,79,80,81],[42,43,44],[49,50],
        [53,67,68,96,97,98],[54,55],[58,59,60,61,62],[63,64,65,66],[69,70,71],
        [72,73,74],[75,76,77,78],[82,83],[84,85,86,87,88],[89,94],[90,91,92,95],[93]]
SOURCE_SET_SHA='711699af7b8d344a08029399b8dd58135d675c1253ca039f3913f96b2ed51d0c'
EDGES=[(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11),(8,12),(9,12),(12,13)]


def split_groups(groups,seed=42,target=20):
    order=list(range(len(groups)));random.Random(seed).shuffle(order)
    # Deterministic closest feasible held-out photo count; never splits a group.
    chosen={0:[]}
    for g in order:
        for n,gs in list(chosen.items()):chosen.setdefault(n+len(groups[g]),gs+[g])
    return set(chosen[min(chosen,key=lambda n:(abs(n-target),n))])


def crop_box(points,margin=1.25):
    valid=[p for p in points if p[2]]
    if not valid:raise ValueError('No valid joints')
    xs=[p[0] for p in valid];ys=[p[1] for p in valid]
    x=(min(xs)+max(xs))/2;y=(min(ys)+max(ys))/2
    s=max(max(xs)-min(xs),max(ys)-min(ys),32)*margin
    return [math.floor(x-s/2),math.floor(y-s/2),math.ceil(x+s/2),math.ceil(y+s/2)]


def run(root):
    root=Path(root);data=json.loads((root/'preannotations.json').read_text(encoding='utf-8'))
    if len(data['images'])!=101:raise ValueError('Manual grouping applies only to this 101-photo collection')
    identity=hashlib.sha256(json.dumps([(r['original_name'],r['original_sha256']) for r in data['images']],ensure_ascii=True).encode()).hexdigest()
    if identity!=SOURCE_SET_SHA:raise ValueError('Photos changed: redo visual grouping; do not reuse index groups')
    assert sorted(sum(GROUPS,[]))==list(range(101))
    held=split_groups(GROUPS)
    for gi,ids in enumerate(GROUPS):
        for i in ids:
            row=data['images'][i];row.update(group_id=f'capture_{gi:02d}',split='test' if gi in held else 'train',orientation_reviewed=True)
            candidates=[p for p in row['people'] if p['include']]
            def area(person):
                pts=[v for v in person['keypoints'][:12] if v[2]]
                return (max(v[0] for v in pts)-min(v[0] for v in pts))*(max(v[1] for v in pts)-min(v[1] for v in pts))
            keep={p['id'] for p in sorted(candidates,key=area,reverse=True)[:2 if i in (2,40,41) else 1]}
            for person in row['people']:
                if person['include'] and person['id'] not in keep:
                    person.update(include=False,excluded_reason='non-target background/occluded person; proposal retained for review')
    # Exact duplicates must not cross splits.
    identity={}
    for r in data['images']:
        if r['original_sha256'] in identity and identity[r['original_sha256']]!=r['split']:raise ValueError('Duplicate leakage')
        identity[r['original_sha256']]=r['split']
    data['split_protocol']={'seed':42,'grouping':'visual burst/pose groups, including cross-phone near duplicates',
                           'subject_disjoint':False,'scene_disjoint':False,'augmentation_after_split':True,
                           'test_used_for_selection':False,'groups':len(GROUPS),'test_groups':sorted(held),
                           'scope':'same-session small-sample domain adaptation, not unseen-person/scene generalization'}
    path=root/'annotations_provisional.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    template=Path(__file__).with_name('personal_review.html').read_text(encoding='utf-8')
    (root/'review.html').write_text(template.replace('__ANNOTATIONS__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    overlays=root/'overlays';overlays.mkdir(exist_ok=True)
    crops=root/'crops224_proposed14';crops.mkdir(exist_ok=True)
    for row in data['images']:
        im=Image.open(root/row['image']).convert('RGB');ov=im.copy();draw=ImageDraw.Draw(ov)
        for person in row['people']:
            if not person['include']:continue
            pts=person['keypoints'];box=crop_box(pts)
            for a,b in EDGES:
                if pts[a][2] and pts[b][2]:draw.line([tuple(pts[a][:2]),tuple(pts[b][:2])],fill='#00ee66',width=3)
            for j,(x,y,v) in enumerate(pts):
                if v:draw.ellipse((x-5,y-5,x+5,y+5),fill='#ffbb00' if j>=12 else '#ff3355');draw.text((x+7,y),str(j),fill='white',stroke_width=1,stroke_fill='black')
            dest=crops/row['split'];dest.mkdir(exist_ok=True)
            im.crop(box).resize((224,224),Image.Resampling.BILINEAR).save(dest/f"{row['id']}_{person['id']}.png")
            pilot_points=[v if j<12 else [v[0],v[1],0] for j,v in enumerate(pts)]
            pilot_dest=root/'crops224_pilot12'/row['split'];pilot_dest.mkdir(parents=True,exist_ok=True)
            im.crop(crop_box(pilot_points)).resize((224,224),Image.Resampling.BILINEAR).save(pilot_dest/f"{row['id']}_{person['id']}.png")
            person['preview_crop_box']=box
        ov.thumbnail((800,800));ov.save(overlays/f"{row['id']}.jpg",quality=92)
    summary={'photos':dict(Counter(r['split'] for r in data['images'])),
             'included_people':dict(Counter(r['split'] for r in data['images'] for p in r['people'] if p['include'])),
             'reviewed_people':sum(p['reviewed'] for r in data['images'] for p in r['people'] if p['include']),
             'annotation_sha256':sha256(path),'formal_ground_truth_ready':False,'protocol':data['split_protocol']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);run(p.parse_args().dataset)
