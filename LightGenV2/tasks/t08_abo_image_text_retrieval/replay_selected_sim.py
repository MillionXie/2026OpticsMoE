"""Restore full TEST predictions with the exact adopted epoch25 EMA student.

CPU-only, one fixed batch4 evaluation, no training, model changes or capture.
The original encode/metrics functions are used without timing instrumentation.
Historical .8058 is compared, never forced or attributed to new predictions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parent
REPO=TASK.parents[2]
PROTOCOL=REPO/'LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024'
sys.path.insert(0,str(PROTOCOL))
EXPECTED='a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'
CSV_SHA={'test.csv':'4ca5834abf4bb4d614b41f05b5b2481b3d41720dd78c1c37b05923c97657dfb8',
         'titles.csv':'c97e8da3c6c608a706f2cc0d284aeafb873fc6cc17d04a7c9f9f66b2135846c1'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8')

def student_sha(replacement,readout):
    h=hashlib.sha256()
    for name,model in [('vision',replacement.vision_surrogate),('language',replacement.language_surrogate),('readout',readout)]:
        for k,v in model.state_dict().items():
            value=v.detach().cpu().contiguous()
            h.update((name+'.'+k+str(value.dtype)+str(tuple(value.shape))).encode())
            h.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args();args.output=args.output.resolve()
    relative=Path(__file__).relative_to(REPO).as_posix()
    published=subprocess.check_output(['git','show',args.source_commit+':'+relative],cwd=REPO)
    if published.replace(b'\r\n',b'\n')!=Path(__file__).read_bytes().replace(b'\r\n',b'\n'):
        raise ValueError('Replay entry differs from its published Git source')
    if args.output.exists() or not args.output.is_relative_to((TASK/'runs/simulation').resolve()):
        raise ValueError('A new immutable T08 simulation run is required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only process required')
    if sha(args.checkpoint)!=EXPECTED:raise ValueError('Not the adopted epoch25 EMA checkpoint')
    for name,expected in CSV_SHA.items():
        if sha(args.data/name)!=expected:raise ValueError('Original TEST/title manifest mismatch: '+name)
    args.output.mkdir(parents=True)
    started=time.monotonic();replacement=None
    def status(state,**extra):
        record=dict(status=state,pid=os.getpid(),device='cpu',elapsed_seconds=time.monotonic()-started,**extra)
        write(args.output/'status.json',record);print(json.dumps(record),flush=True)
    status('loading',completed_queries=0,expected_queries=2400)
    try:
        import torch
        from abo_image_student_graph import load,imported_sources
        from LightGenV2.tasks.t08_abo_image_text_retrieval import optical_moe as original
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        loaded,replacement,readout,settings,contract,identity=load(args,'cpu')
        settings.inference_batch_size=4;settings.num_workers=0
        if len(contract.test)!=2400 or len(contract.titles)!=100:raise ValueError('Full 2400/100 TEST contract required')
        if any(p.requires_grad for module in (replacement.vision_surrogate,replacement.language_surrogate,readout) for p in module.parameters()):
            raise ValueError('Student parameters must be frozen')
        before=student_sha(replacement,readout);source_before=imported_sources();source_before[Path(__file__).relative_to(REPO).as_posix()]=sha(__file__)
        write(args.output/'identity.json',dict(identity=identity,source=source_before,
            command=sys.argv,torch=torch.__version__,batch_size=4,num_workers=0,device='cpu',
            historical_torch='2.8.0+cu128',historical_gpu='NVIDIA GeForce RTX 5090',
            runner_source_commit=args.source_commit,
            student_state_sha256=before,checkpoint_sha256=EXPECTED,dataset_sha256=CSV_SHA,
            training=False,hardware_access=False,inference_instrumentation=False,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
        chunks=[]
        with torch.inference_mode():
            for start in range(0,2400,80):
                chunks.append(original.encode_student_samples(loaded,replacement,readout,contract.test[start:start+80],settings))
                status('encoding_images',completed_queries=start+80,expected_queries=2400)
            query=torch.cat(chunks)
            status('encoding_titles',completed_queries=2400,expected_titles=100)
            titles=original._encode_titles_student(loaded,replacement,readout,contract.titles,settings)
            metrics,rows=original._metrics(query,titles,[sample.sku_index for sample in contract.test])
        for row,sample in zip(rows,contract.test):row.update(sample_id=sample.sample_id,product_id=sample.sku_name)
        if len(rows)!=2400 or len({r['sample_id'] for r in rows})!=2400:raise ValueError('Missing or duplicate TEST predictions')
        after=student_sha(replacement,readout)
        if before!=after or sha(args.checkpoint)!=EXPECTED:raise ValueError('Frozen student changed')
        if torch.cuda.is_initialized():raise ValueError('CPU replay created CUDA')
        for name,expected in source_before.items():
            if sha(REPO/name)!=expected:raise ValueError('Source changed during replay: '+name)
        original.write_csv(args.output/'retrieval_predictions.csv',rows,list(rows[0]))
        torch.save({'query':query.float(),'titles':titles.float()},args.output/'student_embeddings.pt')
        correct=sum(r['true_rank']==1 for r in rows)
        report=dict(status='complete',mode='frozen_selected_student_CPU_replay',metrics=metrics,
            queries=2400,candidates=100,correct_at_1=correct,checkpoint_sha256=EXPECTED,
            historical_correct_at_1=1934,historical_hit_at_1=1934/2400,
            historical_summary_reproduced=correct==1934,weights_unchanged=True,
            student_state_sha256_before=before,student_state_sha256_after=after,
            training=False,hardware_access=False,device='cpu',torch=torch.__version__,
            batch_size=4,original_evaluation_functions=True,source_unchanged=True,
            elapsed_seconds=time.monotonic()-started)
        write(args.output/'report.json',report)
        manifest={name:sha(args.output/name) for name in ('identity.json','retrieval_predictions.csv','student_embeddings.pt','report.json')}
        write(args.output/'artifact_manifest.json',manifest)
        status('complete',completed_queries=2400,correct_at_1=correct,hit_at_1=metrics['recall_at_1'],historical_summary_reproduced=correct==1934)
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        if replacement is not None:replacement.close()

if __name__=='__main__':main()
