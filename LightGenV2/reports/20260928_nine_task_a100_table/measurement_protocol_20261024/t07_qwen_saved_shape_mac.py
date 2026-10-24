"""Original enrolled-SKU Qwen64 graph MAC from saved600 dynamic image shapes."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from native_shape_mac import native_mac,TRANSFORMERS_QWEN3VL_SOURCE_SHA

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
TASK=REPO/'LightGenV2/tasks/t07_abo_image_retrieval'
ROOT=TASK/'runs/simulation/a100_input_inclusive_20261024_qwen01'
CONFIG=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_title_operations_20261024_01/audit_original/model_config.json'
CONFIG_SHA='9172f55b0b9cce70b7f67b10c58a408ccf3ec15c587e6efd4d5f41631237fded'
WEIGHT_SHA='c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1'
PROTOCOL_SHA='f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde'
REFERENCE_SHA='0f8a23f8f029d8578ed2da60fe1be98de68b589dcf9036ca53b562e93f481a27'
GALLERY_SHA='c6eb631c268d2446a2f783854c86d8493cdbcaa04c0669148b16d9785016d8d7'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def request_terms(config,row):
    ids=ast.literal_eval(row['input_ids']);mask=ast.literal_eval(row['attention_mask']);grid=ast.literal_eval(row['image_grid_thw'])
    length=int(row['valid_input_tokens']);pixels=ast.literal_eval(row['pixel_values_shape'])
    if (len(ids)!=1 or len(ids[0])!=length or mask!=[[1]*length] or ast.literal_eval(row['input_ids_shape'])!=[1,length]
            or len(grid)!=1 or len(grid[0])!=3 or min(grid[0])<1 or grid[0][0]!=1):raise ValueError('Original unpadded B1 actual grid/sequence required')
    t,h,w=grid[0];v=config['vision_config'];patch_width=v['in_channels']*v['temporal_patch_size']*v['patch_size']**2
    if pixels!=[t*h*w,patch_width]:raise ValueError('Original processor patch/grid shape differs')
    terms=native_mac(config,length,[h*w]*t)
    terms['GPU_FP32_gallery1600_descriptor64_dot']=1600*64
    return terms,dict(length=length,grid=grid[0],premerger_tokens=t*h*w)


def derive(config_path=CONFIG):
    if sha(config_path)!=CONFIG_SHA:raise ValueError('Exact original Embedding2B config required')
    config=read(config_path);repeats=[];refs=[];signatures=[]
    for repeat in (1,2,3):
        folder=ROOT/f'qwen_repeat{repeat:02d}';identity=read(folder/'identity.json');report=read(folder/'report.json')
        sealed={r['file']:r['sha256'] for r in read(folder/'files_manifest.json')}
        for name in ('identity.json','report.json','calls.csv'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Sealed actual dynamic shape or cost differs')
        if (report['status']!='complete' or report['measured_samples']!=200 or identity['repeat']!=repeat
                or identity['model_assets']['config.json']!=CONFIG_SHA or identity['model_assets']['model.safetensors']!=WEIGHT_SHA
                or identity['manifest_sha256']!=PROTOCOL_SHA or identity['reference_sha256']!=REFERENCE_SHA
                or identity['gallery_cache_sha256']!=GALLERY_SHA or identity['gallery_count']!=1600
                or identity['explicit_warmup_forwards']!=0 or identity['mps_percent']!=100
                or identity['gpu_uuid']!='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'):
            raise ValueError('Original qualified enrolled-SKU Qwen64 scope differs')
        source='LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024/measure_qwen.py'
        blob=subprocess.check_output(['git','show',identity['git_commit']+':'+source],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=identity['sources'][source]:raise ValueError('Original base-model/no-LM-head/MRL/search graph differs')
        with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
        if len(calls)!=200:raise ValueError('All original200 records required')
        records=[]
        for i,row in enumerate(calls):
            if int(row['index'])!=i:raise ValueError('Original order differs')
            terms,shapes=request_terms(config,row);records.append(dict(index=i,sample_id=row['sample_id'],**shapes,MAC=sum(terms.values()),MAC_terms=terms))
        signatures.append(records);repeats.append(dict(repeat=repeat,MAC200=sum(r['MAC'] for r in records),records=records))
        refs.append(dict(run=folder.relative_to(REPO).as_posix(),report_sha256=sealed['report.json'],calls_sha256=sealed['calls.csv'],
            original_forward_source_sha256=identity['sources'][source],source_commit=identity['git_commit'],
            seconds200=report['continuous_wall_s'],joules200=report['total_system_energy_j']))
    if signatures[1:]!=[signatures[0],signatures[0]]:raise ValueError('Original three dynamic shape workloads differ')
    op=2*sum(r['MAC200'] for r in repeats);seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs)
    return dict(status='ORIGINAL_T07_QWEN64_AND_SAVED600_DYNAMIC_SHAPES_MAC_DERIVED',calls=600,MAC_OP600=op,
        TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
        repeat_results=repeats,cost_references=refs,config_sha256=CONFIG_SHA,native_weight_sha256=WEIGHT_SHA,
        GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,complete_scalar_OP_claim=False,
        model_LM_head_executed=False,offline_gallery_inference_included=False,
        convention='1 real dense MAC=2OP; actual dynamic grid24Vision/28Language/QKAV/GQA/all4mergers; original64-prefix descriptor search1600',
        non_MAC_not_in_subtotal=['bias','norm','activation','softmax','embedding lookup','position interpolation/vision rotary outer/trig','RoPE elementwise','MRL slice/normalization','stable argsort','layout'],
        transformers_runtime_source_sha256=TRANSFORMERS_QWEN3VL_SOURCE_SHA,derivation_source_sha256=sha(__file__),
        derivation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=CONFIG);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous derivation')
    v=derive(a.config)
    with a.output.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
    print(json.dumps({k:v[k] for k in ('TOP_per_input','TOPS','TOPS_per_W','GPU_forwards')}))
