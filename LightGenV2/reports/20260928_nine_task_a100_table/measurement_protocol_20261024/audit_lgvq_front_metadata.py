"""Read original archived LGVQ cache metadata, not cached online inference.

Never imports Torch, reconstructs a model or reads multi-GiB tensor storage.
Full-member SHAs below are original bundle declarations; only the bounded
pickle prefix is freshly hashed. A metadata match is NOT numerical replay.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(HERE))
from torch_metadata import read_metadata

PREFIX='experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/deployment/data/'
MEMBERS={
    'raw_frames':('lgvq_four_frames_center100_224_uint8.pt','76fd642fd149dbae516f7632c2e0b4cc06d44a37da03b10a486c769229c50f63'),
    'conv5_quality':('lgvq_quality_conv5_4f_196x192.pt','d1d18aa7f316d42f8060ca98b37fa18f64aefc10906b84fa243e125049963855'),
    'qwen_vision':('qwen3vl_front_4f_196x1024_quality14.pt','410dead1236ef3fbe43edb498f7ef22875cef870b9af50a2705839d56e2f20c1'),
    'qwen_spatial_language':('qwen3vl_front_spatial_prompt_2048.pt','bdce3439754e84ffbcbec601fbd7392f04b3545191d8d57cca639e9a6a8ddb73'),
}
MANIFEST_SHA='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'


def canonical_sha(record):
    return hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!='sha256'},
        ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()


def hashed_records(value,path=''):
    if isinstance(value,dict):
        if 'contract' in value and 'sha256' in value:
            if canonical_sha(value)!=value['sha256']:raise ValueError('Original hashed metadata changed: '+path)
            yield dict(path=path,contract=value['contract'],sha256=value['sha256'])
        for key,item in value.items():yield from hashed_records(item,path+'/'+key)
    elif isinstance(value,list):
        for i,item in enumerate(value):yield from hashed_records(item,path+'/'+str(i))


def source_map():
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),HERE/'torch_metadata.py')}


def audit(archive,manifest):
    archive,manifest=Path(archive),Path(manifest)
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=MANIFEST_SHA:raise ValueError('Original LGVQ manifest differs')
    with manifest.open(encoding='utf-8-sig',newline='') as f:expected=list(csv.DictReader(f))
    if len(expected)!=2808:raise ValueError('Original TRAIN2250/TEST558 required')
    records={}
    with zipfile.ZipFile(archive) as z:
        bundle_bytes=z.read('BUNDLE_MANIFEST.json');bundle=json.loads(bundle_bytes)
        by_path={r['path']:r for r in bundle['files']}
        for label,(name,pin) in MEMBERS.items():
            full_name=PREFIX+name;declared=by_path[full_name];entry=z.getinfo(full_name)
            if declared['sha256']!=pin or declared['bytes']!=entry.file_size:
                raise ValueError('Wrong original bundle member metadata')
            with z.open(full_name) as stream:
                value,pickle_bytes=read_metadata(stream)
                prefix_bytes_read=stream.tell()
            if label!='qwen_spatial_language':
                if value.get('sample_ids')!=[r['sample_id'] for r in expected]:
                    raise ValueError('Cache sample order does not match original manifest')
            checks=list(hashed_records(value))
            records[label]=dict(member=full_name,declared_full_member_sha256=pin,
                full_tensor_storage_content_rehashed=False,declared_bytes=declared['bytes'],
                fresh_pickle_sha256=hashlib.sha256(pickle_bytes).hexdigest(),pickle_bytes=len(pickle_bytes),
                actual_prefix_bytes_read=prefix_bytes_read,hashed_metadata_checks=checks,metadata=value)
    return dict(status='READ_ONLY_ORIGINAL_METADATA_AUDIT_NOT_NUMERICAL_EQUIVALENCE',
        source_archive=str(archive.resolve()),archive_bytes=archive.stat().st_size,
        original_bundle_manifest_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        original_bundle_built_utc=bundle['built_utc'],manifest_sha256=MANIFEST_SHA,
        ordered2808_ids_sha256=hashlib.sha256('\n'.join(r['sample_id'] for r in expected).encode('utf8')).hexdigest(),
        sources=source_map(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        no_torch_import=True,no_model_loaded=True,no_tensor_storage_materialized=True,
        GPU_forwards=0,formal_timing_or_power_measurements=0,created_unix=time.time(),records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Keep original identity evidence')
    value=audit(args.archive,args.manifest)
    with args.output.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=value['status'],output=str(args.output),
        prefix_bytes_read={k:r['actual_prefix_bytes_read'] for k,r in value['records'].items()},
        metadata_roots={k:list(r['metadata']) for k,r in value['records'].items()})))


if __name__=='__main__':main()
