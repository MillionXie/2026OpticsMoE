"""FP32-only supplement for original A10040, no finalized B1 edits or training."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from . import rental_precision_recheck as shared
from . import evaluate_temporal_baseline_batches as frozen
from . import measure_frozen_batch_costs as costs
from . import unified_resident_utilization as resident
from . import quality_token_common as core

TASK=Path(__file__).resolve().parent
REPO=TASK.parents[2]
ROOT=TASK/'runs/simulation/lgvq_fp32_a100_20261019'


def runtime(args,guard,identity):
    import torch
    assert args.precision=='fp32'
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    digest=frozen.sha256(args.checkpoint)
    if digest!=costs.CHECKPOINTS[args.model]:raise RuntimeError('Exact trained head required')
    weight=payload['state_dict']['weight'].float()
    head=torch.nn.Linear(weight.shape[1],5,bias=False).cuda().eval().requires_grad_(False)
    head.weight.copy_(weight.cuda());levels=payload['level_scores'].cuda().float()
    rows=[r for r in core.read_manifest(args.manifest) if r['split']=='test']
    if frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA or len(rows)!=558:raise RuntimeError('Fixed558')
    assets=costs.model_hashes(args.model_path)
    handles=[]
    if args.model=='qwen':
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from . import quality_token_batch_benchmark_200 as q
        processor=AutoProcessor.from_pretrained(args.model_path,min_pixels=448**2,max_pixels=448**2,local_files_only=True,trust_remote_code=True)
        processor.tokenizer.padding_side='left'
        model=Qwen3VLForConditionalGeneration.from_pretrained(args.model_path,local_files_only=True,
            trust_remote_code=True,dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        prompt=core.render_prompt(processor,'temporal');first=model.model.visual.blocks[0]
        def prepare(selected):return q.prepare_batch(selected,processor,prompt,torch.device('cpu'),448)
        def features(inputs):
            hidden=model.model(**inputs,return_dict=True,use_cache=False).last_hidden_state
            mask=inputs['attention_mask'].bool();last=torch.arange(mask.shape[1],device='cuda').expand_as(mask).masked_fill(~mask,-1).amax(1)
            value=hidden[torch.arange(len(hidden),device='cuda'),last].float()
            model.lm_head(value)
            return value
    elif args.model=='glm':
        from transformers import AutoProcessor,Glm4vForConditionalGeneration
        from . import glm46v_flash_temporal_baseline as glm
        processor=AutoProcessor.from_pretrained(args.model_path,local_files_only=True);processor.tokenizer.padding_side='left'
        model=Glm4vForConditionalGeneration.from_pretrained(args.model_path,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        model._quality_processor=processor
        model._quality_prompt=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':glm.shared.PROMPT}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        first=model.model.visual.blocks[0]
        def prepare(selected):return glm.prepare_batch(selected,model)
        def features(inputs):
            hidden=model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
            mask=inputs['attention_mask'].bool();last=torch.arange(mask.shape[1],device='cuda').expand_as(mask).masked_fill(~mask,-1).amax(1)
            return hidden[torch.arange(len(hidden),device='cuda'),last].float()
    else:
        from .visual_backbone_quality import decode_ordered_frames
        if args.model=='clip':
            import clip
            model,transform=clip.load(str(args.model_path),device='cuda',jit=False);model=model.float().eval().requires_grad_(False)
            first=model.visual.transformer.resblocks[0];size=224
            def encode(frames):return torch.stack([transform(f) for f in frames])
            def features(inputs):return torch.nn.functional.normalize(model.encode_image(inputs).float(),dim=-1).reshape(-1,2048)
        else:
            import numpy as np
            from ultralytics import YOLO
            model=YOLO(str(args.model_path)).model.cuda().float().eval().requires_grad_(False)
            captured={};handles.append(model.model[10].register_forward_hook(lambda m,a,o:captured.update(feature=o)))
            first=model.model[0];size=640
            def encode(frames):return torch.from_numpy(np.stack([np.asarray(f).copy() for f in frames])).permute(0,3,1,2).contiguous().float()/255
            def features(inputs):
                captured.clear();model(inputs)
                return torch.nn.functional.normalize(captured['feature'].float().mean((-2,-1)),dim=-1).reshape(-1,2048)
        def prepare(selected):
            frames=[];positions=[]
            for row in selected:
                images,where=decode_ordered_frames(Path(row['video_path']),core.FRAME_FRACTIONS[4],size);frames.extend(images);positions.append(where)
            return encode(frames),positions
    if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('FP32 parameter audit failed')
    guard.bind()
    def transfer(cpu):
        if isinstance(cpu,dict):return {k:v.cuda().float() if v.is_floating_point() else v.cuda() for k,v in cpu.items()}
        return cpu.cuda().float()
    def compute(inputs):
        with torch.inference_mode():return head(features(inputs)).softmax(-1)@levels,None
    identity.update(checkpoint_sha256=digest,assets=assets,tf32=False,autocast=False,
        floating_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
        fp16_feature_storage_roundtrip_removed=args.model in ('clip','yolo'),
        fp32_note='Frozen deployed weights cast to FP32; no recovery of previously rounded weights')
    return model,rows,prepare,transfer,compute,first,assets


def worker(args):
    import torch
    args.baseline=args.model;args.model_path=Path(args.model_path);args.checkpoint=Path(args.checkpoint)
    # Standard preflight accepts the model path in args.model, preserve its interface.
    if args.output.exists():raise RuntimeError('Preserve prior output')
    if args.model!='glm':
        pre=SimpleNamespace(**vars(args));pre.model=args.model_path;frozen.preflight_files(pre)
    args.output.mkdir(parents=True);guard=shared.Guard(args)
    identity=dict(model=args.model,batch=args.batch,precision='fp32',gpu_uuid=args.gpu_uuid,
        sources={**shared.sources(),str(Path(__file__).relative_to(REPO)):frozen.sha256(Path(__file__))},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(8)
    try:
        guard.check()
        model,rows,prepare,transfer,compute,first,assets=runtime(args,guard,identity)
        shared.old.write(args.output/'identity.json',identity)
        if args.stage=='util':
            cpu,positions=prepare(rows[:args.batch]);inputs=transfer(cpu)
            reference,_=compute(inputs);reference=reference.cpu()
            for paced in (False,True):
                report=shared.measure_util(args,compute,inputs,args.output/('paced' if paced else 'native'),guard,paced)
                if any(not torch.allclose(torch.tensor(r['output']),reference,atol=.0002,rtol=0) for r in report['repeats']):raise RuntimeError('Frozen repeated outputs differ')
        else:
            util=json.loads(Path(args.utilization_source).read_text())
            if util['model']!=args.model or util['batch']!=args.batch or util['precision']!='fp32':raise RuntimeError('Util point mismatch')
            duty=util['frozen_duty'] if args.stage=='paced-cost' else 1
            identity['independent_utilization']=dict(path=args.utilization_source,sha256=frozen.sha256(Path(args.utilization_source)),value=util['mean_gpu_busy_percent'])
            shared.evaluate(args,compute,prepare,transfer,first,args.output/'evaluation',guard,duty,identity,rows)
        if costs.model_hashes(args.model_path)!=assets:raise RuntimeError('Official assets changed')
        guard.check();shared.old.write(args.output/'status.json',dict(status='complete',time=time.time()))
    except BaseException as error:
        shared.old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    from . import run_temporal_telemetry_suite as catalog
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ROOT.exists():raise RuntimeError('Preserve prior suite')
        ROOT.mkdir();state=dict(status='waiting_for_idle',completed=[],deadline=args.deadline)
        def save():shared.old.write(ROOT/'suite_status.json',state)
        stable=0;waiting_until=min(args.deadline,time.time()+1800)
        while stable<3:
            if time.time()>=waiting_until:state.update(status='waiting_expired');save();return
            records=shared.guard_records(args.gpu_uuid);stable=stable+1 if not records else 0
            state.update(stable=stable,foreign=records,time=time.time());save()
            if stable<3:time.sleep(10)
        try:
            for model,batch in [('qwen',3),('clip',10),('yolo',3),('glm',2)]:
                if model=='glm':
                    original=TASK/'runs/simulation'/resident.SPECIAL['glm'][0]
                    report=json.loads((original/'a100_batch01/report.json').read_text());command=report['identity']['command']
                    mp=command[command.index('--model')+1];pt=original/'train/best_checkpoint.pt';python=resident.SPECIAL['glm'][2]
                else:
                    mp=catalog.QWEN_MODEL if model=='qwen' else catalog.reproduction.MODEL_PATHS[model]
                    pt=catalog.QWEN_PT if model=='qwen' else catalog.PREVIOUS/model/'checkpoints/temporal/best_checkpoint.pt'
                    python=catalog.QWEN_ENV if model=='qwen' else catalog.reproduction.ENV_PATHS[model]
                for stage in ('util','paced-cost','native-cost'):
                    if time.time()+180>=args.deadline:raise TimeoutError('Finite window')
                    if shared.guard_records(args.gpu_uuid):raise RuntimeError('Foreign GPU process, stop')
                    name=f'{model}_{stage}';util=ROOT/f'{model}_util'/('paced' if stage=='paced-cost' else 'native')/'report.json'
                    cmd=[str(python),'-u','-m',__package__+'.lab_fp32_recheck','--model',model,'--batch',str(batch),
                        '--model-path',str(mp),'--checkpoint',str(pt),'--manifest',str(args.manifest),'--stage',stage,
                        '--gpu-uuid',args.gpu_uuid,'--deadline',str(args.deadline),'--output',str(ROOT/name)]
                    if model!='qwen' and model!='glm':cmd+=['--reproduction-receipt',str(catalog.PREVIOUS/model/'reproduction_receipt.json')]
                    if stage!='util':cmd+=['--utilization-source',str(util)]
                    shared.old.write(ROOT/(name+'.command.json'),cmd)
                    with (ROOT/(name+'.log')).open('x') as log:
                        child=subprocess.Popen(cmd,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(status='running',phase=name,child_pid=child.pid);save()
                        try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                        except BaseException:
                            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
                    if code:raise RuntimeError(f'{name} failed exit{code}; retain all evidence')
                    state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',action='store_true')
    p.add_argument('--model',choices=('qwen','clip','yolo','glm'));p.add_argument('--model-path')
    p.add_argument('--checkpoint');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--batch',type=int);p.add_argument('--stage',choices=('util','paced-cost','native-cost'))
    p.add_argument('--gpu-uuid',default=frozen.A100_UUID);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--utilization-source');p.add_argument('--reproduction-receipt',type=Path)
    args=p.parse_args();args.precision='fp32';args.power_limit_w=250;args.output=args.output.resolve()
    if args.gpu_uuid!=frozen.A100_UUID or not args.output.is_relative_to(ROOT) or not 0<args.deadline-time.time()<=7200:raise ValueError('Original A10040 and bounded2h')
    (suite if args.suite else worker)(args)


if __name__=='__main__':main()
