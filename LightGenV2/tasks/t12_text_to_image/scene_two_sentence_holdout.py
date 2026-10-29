"""Prepare sealed target-free TEST500 requests, matching the two-sentence task.

Only the previously audited ten held-out ABO products are used. Downloading
their original white-reference assets is data preparation, not model inference.
There are 80 reference/prompt contexts and 500 seeded draws, not 500 products.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import traceback
from PIL import Image, ImageOps
from .appearance_scene_protocol import SHORT_PROFILE, conditions, seed_indices
from .prepare_abo_cleanrender import ARCHIVE_URL, _archive, _fetch_remote_member
from .scene_generation_data import request_seed
from .white_reference_scene_probe import digest, write


def test_inventory(refs, paired):
    held = [r for r in refs if r['split']=='test']
    assert len(held)==10 and Counter(r['category'] for r in held)=={'lamp':5,'pillow':5}
    assert len({r['source_product_id'] for r in held})==10
    forbidden_products={r['source_product_id'] for r in paired}
    forbidden_pixels={r['reference_sha256'] for r in paired}
    forbidden_families={r['source_family_id'] for r in refs if r['split'] in ['train','dev']}
    for r in held:
        assert r['source_product_id'] not in forbidden_products
        assert r['source_family_id'] not in forbidden_families
        assert r['image_sha256'] not in forbidden_pixels and r['license']=='CC BY 4.0'
        assert r['source_member'].endswith('.png')
    return sorted(held,key=lambda r:(r['category'],r['source_product_id']))


def prepare(a):
    old=json.loads((a.source_requests/'report.json').read_text())
    audit=json.loads((a.source_requests/'data_audit.json').read_text())
    current=json.loads((a.paired_data/'report.json').read_text())
    assert old['complete'] and audit['role_disjoint'] and audit['reference_pixels_disjoint']
    assert not audit['generation_or_features_performed'] and current['complete'] and current['role_disjoint']
    pairs=json.loads((a.paired_data/'pairs.json').read_text())
    assert digest(a.paired_data/'pairs.json')==current['pairs_sha256']
    rows=test_inventory(json.loads((a.source_requests/'references.json').read_text()),pairs)
    a.output.mkdir();(a.output/'rgba').mkdir();(a.output/'references').mkdir()
    with _archive(ARCHIVE_URL) as archive:
        infos={r['source_member']:archive.getinfo(r['source_member']) for r in rows}
    def fetch(row):
        raw=_fetch_remote_member(ARCHIVE_URL,infos[row['source_member']])
        rid=row['reference_id'];(a.output/'rgba'/(rid+'.png')).write_bytes(raw)
        with Image.open(io.BytesIO(raw)) as image:
            assert image.mode=='RGBA' and min(image.size)>=512
            native_size=list(image.size);white=Image.new('RGBA',image.size,'white');white.alpha_composite(image)
            white=ImageOps.pad(white.convert('RGB'),(1024,1024),method=Image.Resampling.LANCZOS,color='white')
            name='references/'+rid+'.png';white.save(a.output/name)
            pixel_sha=hashlib.sha256(white.tobytes()).hexdigest()
        return {**row,'reference':name,'reference_sha256':digest(a.output/name),'native_RGBA_size':native_size,
                'RGBA_sha256':hashlib.sha256(raw).hexdigest(),'RGB_pixel_sha256':pixel_sha,
                'source_archive':ARCHIVE_URL,'upstream_attribution':'Amazon Berkeley Objects (ABO)'}
    with ThreadPoolExecutor(max_workers=4) as pool:refs=list(pool.map(fetch,rows))
    forbidden_pixel_values=set()
    for path in {r['reference'] for r in pairs}:
        with Image.open(a.paired_data/path) as im:forbidden_pixel_values.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    assert not {r['RGB_pixel_sha256'] for r in refs}&forbidden_pixel_values
    inputs=[];metadata=[]
    for ref in refs:
        for scene in conditions(ref['category'],SHORT_PROFILE):
            assert scene['prompt']==scene['native_caption'] and scene['prompt'].count('.')==2
            for index in seed_indices('test',int(scene['condition_id'])):
                row={k:scene[k] for k in ['prompt','condition_id','category','native_caption','reward_caption','prompt_profile']}
                row.update(sample_id=ref['reference_id']+'-scene'+scene['condition_id']+'-draw'+str(index),
                    reference=ref['reference'],reference_id=ref['reference_id'],reference_sha256=ref['reference_sha256'],
                    seed=request_seed(ref['RGB_pixel_sha256'],index))
                inputs.append(row);metadata.append({**row,'split':'test','seed_index':index,
                    'source_product_id':ref['source_product_id'],'source_family_id':ref['source_family_id'],
                    'requested_attributes':scene['requested_attributes']})
    assert len(inputs)==500 and len({(r['reference_sha256'],r['prompt'],r['seed']) for r in inputs})==500
    contexts=len({(r['reference_sha256'],r['prompt']) for r in inputs});assert contexts==80
    write(a.output/'inference_inputs.json',inputs);write(a.output/'requests.json',metadata);write(a.output/'references.json',refs)
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'source_sha256':digest(Path(__file__)),'argv':sys.argv,'prompt_profile':SHORT_PROFILE,
        'request_sha256':digest(a.output/'inference_inputs.json'),'role_disjoint':True,
        'parent_role_audit_sha256':digest(a.source_requests/'data_audit.json'),
        'current_pairs_sha256':current['pairs_sha256'],'TEST_products':10,'TEST_views':10,
        'distinct_reference_prompt_contexts':contexts,'seeded_generation_requests':500,
        'per_category':dict(Counter(r['category'] for r in inputs)),
        'unique_TEST_GT_required':False,'model_inference_performed':False,'model_feature_cache_created':False,
        'TEST_used_for_training_selection':False,'RGB_source_members_CRC_checked':True,
        'source_license':'CC BY 4.0','native_generation_resolution_planned':512,
        'statistical_unit_warning':'500 draws from 80 contexts / 10 products; report clustered uncertainty, not 500 independent products.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-requests','paired-data','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists() and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:prepare(a)
    except Exception:
        if a.output.exists():write(a.output/'failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
