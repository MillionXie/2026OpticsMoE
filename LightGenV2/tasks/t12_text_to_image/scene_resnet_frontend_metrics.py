"""Three retained scores only; no ImageReward model or selection dependency."""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from .scene_instruction_data import sha, write
from .scene_resnet_frontend import PARENT, REQUESTS, experiment_profile

FIELDS=('CLIP_T','DINOv2_I','CLIP_I')


def fixed_last_records(folder,data,profile,requested_epoch=None):
    report=json.loads((folder/'report.json').read_text())
    expected_scope='TEST_fixed_last50'if requested_epoch in [None,50] else'TEST_fixed_requested_epoch'
    if (not report['complete']or report.get('scope')!=expected_scope
        or report['fixed_generator_sha256']!=profile['generator_sha256']
        or report.get('TEST_used_for_model_selection')is not False
        or report['TEST_selection_protocol']['request_sha256']!=REQUESTS):
        raise ValueError('Fixed electronic-Qwen decoder and last50 cohort required')
    if requested_epoch is not None and report.get('evaluation_epoch')!=requested_epoch:
        raise ValueError('Requested epoch and actual checkpoint differ')
    request=data/'inference_inputs.json'
    if sha(request)!=REQUESTS:raise ValueError('Changed target-free TEST requests')
    expected={r['sample_id']:r for r in json.loads(request.read_text())}
    rows=json.loads((folder/'images.json').read_text())
    if len(rows)!=500 or len({r['sample_id']for r in rows})!=500:
        raise ValueError('Exactly500 unique outputs required')
    for row in rows:
        wanted=expected[row['sample_id']]
        if (any(row[k]!=v for k,v in wanted.items())or row['split']!='test'
            or row['checkpoint_sha256']!=report['checkpoint_sha256']
            or Path(row['generated_PNG']).name!=row['generated_PNG']
            or sha(folder/row['generated_PNG'])!=row['generated_sha256']
            or not sha(folder/(row['reference_id']+'.png'))==row['input_sha256']==row['reference_sha256']):
            raise ValueError('TEST identity or output drift')
        yield dict(row,run_id=folder.name,run_path=str(folder))


def native_peer_records(folder,data):
    report=json.loads((folder/'report.json').read_text())
    if not(report['complete']and report.get('native_pretrained_peer')and report['scope']=='TEST_native_frozen_model'
           and report['request_sha256']==REQUESTS and report['training_performed']is False and report['trainable_parameters']==0):
        raise ValueError('Complete frozen native image-text model required')
    request=data/'inference_inputs.json'
    if sha(request)!=REQUESTS:raise ValueError('Changed canonical TEST requests')
    expected={r['sample_id']:r for r in json.loads(request.read_text())};rows=json.loads((folder/'images.json').read_text())
    if len(rows)!=500 or len({r['sample_id']for r in rows})!=500:raise ValueError('Full500 unique outputs required')
    for row in rows:
        if (any(row[k]!=v for k,v in expected[row['sample_id']].items())or row['split']!='test'
            or row['checkpoint_sha256']!=report['checkpoint_sha256']or Path(row['generated_PNG']).name!=row['generated_PNG']
            or sha(folder/row['generated_PNG'])!=row['generated_sha256']
            or not sha(folder/(row['reference_id']+'.png'))==row['input_sha256']==row['reference_sha256']):raise ValueError('Native TEST identity drift')
        yield dict(row,run_id=folder.name,run_path=str(folder))


def aggregate(rows):
    result={}
    for name in sorted({r['run_id'] for r in rows}):
        for category in ('all','lamp','pillow'):
            group=[r for r in rows if r['run_id']==name and category in ('all',r['category'])]
            scores={}
            for field in FIELDS:
                values=[float(r[field])for r in group]
                if not values or not all(math.isfinite(v)for v in values):
                    raise ValueError('Complete finite scores required')
                scores[field]=dict(mean=statistics.mean(values),sample_std=statistics.stdev(values))
            result[name+'/'+category]=dict(n=len(group),scores=scores)
    return result


def run(a):
    import torch
    import clip
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModel
    from .white_reference_metrics import load_records
    native=getattr(a,'native_peers',False);epochs=getattr(a,'requested_epochs',None)
    profile=None if native else experiment_profile(a.profile)
    fixed=native or profile['steps']==30
    if epochs and(native or a.profile!='qwen70_30_supervised'or len(epochs)!=len(a.runs)or len(set(epochs))!=len(epochs)
                  or any(e not in [0,1,2,5,10,50]for e in epochs)):raise ValueError('Exact user-requested epochs and runs required')
    if fixed and(a.data is None or(not native and not epochs and len(a.runs)!=1)):raise ValueError('Canonical fixed cohort required')
    rows=[]
    for i,folder in enumerate(a.runs):
        rows.extend(native_peer_records(folder,a.data)if native else fixed_last_records(folder,a.data,profile,epochs[i]if epochs else None)if fixed else load_records(folder))
    signatures=[]
    for folder in a.runs:
        report=json.loads((folder/'report.json').read_text())
        if not native and(report.get('fixed_generator_sha256')!=profile['generator_sha256'] or report['TEST_selection_protocol']['request_sha256']!=REQUESTS):
            raise ValueError('Same fixed70 tail and complete target-free requests required')
        signatures.append({r['sample_id']:(r['input_sha256'],r['prompt'],r['seed'])+(()if native else(r['initial_latent_sha256'],))
                           for r in rows if r['run_id']==folder.name})
    if not all(s==signatures[0]for s in signatures)or len(signatures[0])!=500:
        raise ValueError('Candidate RGB/prompt/seed/noise mismatch')
    assets={str(a.clip_checkpoint):sha(a.clip_checkpoint)}
    for p in sorted(a.dino_path.iterdir()):
        if p.is_file()and p.suffix in ['.bin','.json','.safetensors']:assets[str(p)]=sha(p)
    write(a.output/'model_assets.json',assets)
    model,preprocess=clip.load(str(a.clip_checkpoint),device='cuda',jit=False)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        for i,row in enumerate(rows):
            root=Path(row['run_path'])
            with Image.open(root/row['generated_PNG'])as im:x=preprocess(im.convert('RGB'))[None].cuda()
            with Image.open(root/(row['reference_id']+'.png'))as im:y=preprocess(im.convert('RGB'))[None].cuda()
            text=clip.tokenize([row['prompt']],truncate=False).cuda()
            x=torch.nn.functional.normalize(model.encode_image(x).float(),dim=-1)
            y=torch.nn.functional.normalize(model.encode_image(y).float(),dim=-1)
            text=torch.nn.functional.normalize(model.encode_text(text).float(),dim=-1)
            row.update(CLIP_T=float((text*x).sum()),CLIP_I=float((x*y).sum()))
            write(a.output/'progress.json',dict(stage='CLIP_full_prompt',done=i+1,total=len(rows)))
    del model;gc.collect();torch.cuda.empty_cache()
    processor=AutoImageProcessor.from_pretrained(a.dino_path,local_files_only=True)
    model=AutoModel.from_pretrained(a.dino_path,local_files_only=True).cuda().eval().requires_grad_(False)
    with torch.inference_mode():
        for i,row in enumerate(rows):
            root=Path(row['run_path'])
            with Image.open(root/row['generated_PNG'])as im:x=im.convert('RGB').copy()
            with Image.open(root/(row['reference_id']+'.png'))as im:y=im.convert('RGB').copy()
            batch=processor(images=[x,y],return_tensors='pt').to('cuda')
            f=torch.nn.functional.normalize(model(**batch).last_hidden_state[:,0].float(),dim=-1)
            row['DINOv2_I']=float((f[0]*f[1]).sum())
            write(a.output/'progress.json',dict(stage='DINOv2',done=i+1,total=len(rows)))
    write(a.output/'per_image.json',rows);summary=aggregate(rows);write(a.output/'summary.json',summary)
    with (a.output/'summary.csv').open('w',newline='',encoding='utf-8')as stream:
        writer=csv.writer(stream);writer.writerow(['run/category','n',*FIELDS])
        for key,item in summary.items():writer.writerow([key,item['n'],*[item['scores'][f]['mean']for f in FIELDS]])
    # Declare the new user-approved selector explicitly; never reuse the old
    # ImageReward-first TEST selector after removing that metric.
    candidates=[]
    for folder in a.runs:
        scores=summary[folder.name+'/all']['scores']
        candidates.append(dict(run=folder.name,checkpoint_sha256=json.loads((folder/'report.json').read_text())['checkpoint_sha256'],
                               evaluation_epoch=epochs[len(candidates)]if epochs else None,
                               **{f:scores[f]['mean']for f in FIELDS}))
    comparison=native or bool(epochs)
    if not comparison:candidates.sort(key=lambda r:(-r['CLIP_T'],-r['DINOv2_I'],-r['CLIP_I'],r['checkpoint_sha256']))
    write(a.output/'selection.json',dict(complete=True,primary='prespecified_comparison_no_winner_selection'if comparison else'human_fixed_last50'if fixed else'full_TEST500_mean_CLIP_T',selection_uses_TEST=not fixed,
        tie_breakers=['DINOv2_I','CLIP_I','checkpoint_SHA'],ImageReward_used=False,
        TEST_used_for_gradient=False,independent_generalization_claim=False,selected=None if comparison else candidates[0],candidates=candidates))
    write(a.output/'report.json',dict(complete=True,n=len(rows),metrics=list(FIELDS),ImageReward_computed=False,
        TEST_used_for_model_selection=not fixed,TEST_used_for_gradient=False,independent_generalization_claim=False,
        fixed_generator_sha256=None if native else profile['generator_sha256'],requested_epochs=epochs,native_peers=native,request_sha256=REQUESTS,source_commit=os.environ['T12_SOURCE_COMMIT']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--profile',choices=['lg70_8','qwen70_30','qwen70_30_supervised'],default='lg70_8')
    p.add_argument('--data',type=Path)
    p.add_argument('--requested-epochs',nargs='+',type=int)
    p.add_argument('--native-peers',action='store_true')
    for k in ['output','clip-checkpoint','dino-path']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    from .scene_execution_gpu import configured_gpu
    gpu=configured_gpu()
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('Fresh published run and authorized GPU required')
    import torch
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),GPU_UUID=gpu,source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__)))
    try:run(a)
    except BaseException:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
