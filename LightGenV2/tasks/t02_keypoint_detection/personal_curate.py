"""Versioned user-requested exclusions; original photos and old runs remain intact."""
import argparse,copy,json,shutil
from pathlib import Path
from .personal_data import fewshot_split,validate_manifest
from .personal_prepare import sha256

EXCLUDED=[f'photo_{i:03d}_p00' for i in [40,41,5,9,10,14,16,19,18,21,22,23,25,27,39,46,48,50,52,68,75,78,77,80,83,97,98]]+['photo_040_p01','photo_041_p01']

def curate(source,output):
    source=Path(source);output=Path(output)
    data,_=validate_manifest(source/'annotations_provisional.json',True)
    found={r['id']+'_'+p['id'] for r in data['images'] for p in r['people'] if p['include']}
    if not set(EXCLUDED)<=found:raise ValueError('Requested exclusion ID missing or already excluded')
    result=copy.deepcopy(data);retained=[];mapping=[]
    for row in result['images']:
        old=row['id'];row['people']=[p for p in row['people'] if p['include'] and old+'_'+p['id'] not in EXCLUDED]
        if not row['people']:continue
        row['source_id']=old;row['source_image']=row['image'];row['id']=f'photo_{len(retained):03d}'
        row['image']='images/'+row['id']+'.jpg'
        row['quality_note']='partial person retained; user noted difficulty' if old in ['photo_003','photo_089'] else None
        for j,p in enumerate(row['people']):
            previous=old+'_'+p['id'];p['source_id']=previous;p['id']=f'p{j:02d}'
            mapping.append({'original_person_id':previous,'new_person_id':row['id']+'_'+p['id'],'group_id':row['group_id']})
        retained.append(row)
    result['images']=retained;result=fewshot_split(result,20,42)
    result['curation']={'source_annotation_sha256':sha256(source/'annotations_provisional.json'),
        'excluded_person_ids':EXCLUDED,'retained_partial_source_ids':['photo_003','photo_089'],
        'selection_bias_notice':'User curated after viewing model outputs; curated exploratory evaluation, not original full-set performance'}
    output.mkdir(parents=True,exist_ok=False);(output/'images').mkdir()
    for row in result['images']:shutil.copy2(source/row['source_image'],output/row['image'])
    for name,value in [('annotations_provisional.json',result),('id_mapping.json',mapping),('exclusions.json',result['curation'])]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    template=Path(__file__).with_name('personal_review.html').read_text(encoding='utf-8')
    (output/'review.html').write_text(template.replace('__ANNOTATIONS__',json.dumps(result,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    _,counts=validate_manifest(output/'annotations_provisional.json',True)
    summary={'photos':len(retained),'people':sum(counts.values()),'people_by_split':counts,
        'photos_by_split':{s:sum(r['split']==s for r in retained) for s in ['train','test']},
        'annotation_sha256':sha256(output/'annotations_provisional.json'),'protocol':result['split_protocol']}
    summary['photos_by_split']={s:sum(r['split']==s for r in result['images']) for s in ['train','test']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();curate(a.source,a.output)
