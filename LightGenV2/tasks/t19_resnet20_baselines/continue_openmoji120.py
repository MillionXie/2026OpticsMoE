"""Fixed continuation of the unchanged balanced OpenMoji objective and AdamW.

The parent last90 is immutable. Full TRAIN fit diagnostics are observational,
not validation or stopping gates. Only last120 is evaluated on complete TEST.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from . import optimization_stage as original
from .optimization_losses import balanced_edit_loss,forward_features
from .optimization_shared_worker import wait_shared
from .models import build_model,parameter_counts
from .task_problems import Problem
from .continue_lgvq50 import structured_sha256,shutdown_loader
from .frozen_resnet18 import file_sha256,encoder_state_sha256,assert_encoder_unchanged
from .train_easy100 import TASK,git,require_published_source,write_json,save_checkpoint

SOURCES=(*original.SOURCES,'optimization_shared_worker.py','continue_openmoji120.py')


def restore_parent(model,optimizer,parent,expected_sha,identity,parent_epoch=90):
    if file_sha256(parent)!=expected_sha:raise ValueError('Parent checkpoint SHA differs')
    payload=torch.load(parent,map_location='cpu',weights_only=False)
    if payload['epoch']!=parent_epoch or payload['manifest']['data_identity']!=identity:
        raise ValueError('Parent epoch or data identity differs')
    model.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    if structured_sha256(optimizer.state_dict())!=structured_sha256(payload['optimizer']):
        raise ValueError('AdamW state was not restored exactly')
    return payload


def index_loader(count):
    return DataLoader(range(count),batch_size=32,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(42))


def replay_loader(loader,ids,history):
    for entry in history:
        order=[]
        for batch in loader:order.extend(ids[i] for i in batch.tolist())
        if structured_sha256(order)!=entry['sample_order_SHA']:
            raise ValueError('Parent stage sampler replay differs')
    return structured_sha256(loader.generator.get_state())


def validate_gate(gate,manifest):
    for key in ('profile_SHA','source_sha256','native_source_sha256','parent_SHA','initial_trainable_SHA','optimizer_state_SHA_after_restore'):
        if gate[key]!=manifest[key]:raise ValueError('Own continuation probe identity differs: '+key)
    if gate.get('status')!='complete' or not gate['gate_passed'] or gate['test_evaluations']!=0:
        raise ValueError('TRAIN-only resume probe not passed')


@torch.no_grad()
def cached_TRAIN_metrics(model,cache,device):
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    accumulator=MetricAccumulator();model.eval()
    for start in range(0,len(cache['sample_id']),32):
        batch,features=original.cached_batch(cache,list(range(start,min(start+32,len(cache['sample_id'])))),device)
        accumulator.update(forward_features(model,'openmoji',features,batch['tokens']),batch)
    return accumulator.compute()


def run(args):
    require_published_source()
    if args.phase not in ('probe','formal'):raise ValueError('Unknown phase')
    plan=json.loads(args.plan.read_text())
    if (plan['parent_epoch'],plan['final_epoch'],plan['stage_epochs'])!=(90,120,30):
        raise ValueError('Fixed continuation budget differs')
    if any(plan[k]!=v for k,v in dict(seed=42,learning_rate=.001,weight_decay=.0001,effective_batch=32,gradient_clip=1.).items()):
        raise ValueError('Approved optimization protocol differs')
    category='smoke' if args.phase=='probe' else 'simulation';run=args.run_dir.resolve()
    if not run.is_relative_to((TASK/'runs'/category).resolve()):raise ValueError('Run outside task')
    run.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='preflight',task='openmoji',phase=args.phase,pid=os.getpid(),
        command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],started_unix=time.time(),git_commit=git('rev-parse','HEAD'),
        source_sha256={n:file_sha256(TASK/n) for n in SOURCES},profile_SHA=file_sha256(args.plan),
        native_source_sha256={n:file_sha256(Path(__file__).resolve().parents[3]/n) for n in original.NATIVE_SOURCES['openmoji']},
        python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu_uuid=args.gpu_uuid,
        seed=42,learning_rate=.001,schedule='constant',warmup_epochs=0,effective_batch=32,
        precision='FP32_TF32_disabled',loss_profile=plan['loss'],historical_TEST_observed=True,
        test_selected_checkpoint=False,parent_epoch=90,final_epoch=120,optimizer_restart=False,
        training_scope='frozen_vision_train_GRU_and_head',frozen_encoder_audits=[])
    write_json(run/'run_manifest.json',manifest);loaders=[];began=time.monotonic()
    try:
        torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        problem=Problem('openmoji')
        if (len(problem.train),len(problem.test))!=(5000,1000):raise ValueError('Original complete data counts differ')
        model=build_model('openmoji',backbone='resnet18_imagenet_frozen',text_encoder='byte_gru')
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=.001,weight_decay=.0001)
        parent=TASK/'runs/simulation'/plan['parent']/'last_checkpoint.pt'
        old=restore_parent(model,optimizer,parent,plan['parent_sha256'],problem.identity)
        parent_manifest=json.loads((parent.parent/'run_manifest.json').read_text())
        parent_result=json.loads((parent.parent/'result.json').read_text())
        if parent_manifest['status']!='complete' or parent_result['status']!='complete':raise ValueError('Parent not complete')
        if parent_result['checkpoint_SHA']!=plan['parent_sha256'] or old['manifest']['loss_profile']!=plan['loss']:
            raise ValueError('Parent objective identity differs')
        counts=parameter_counts(model);frozen=encoder_state_sha256(model.encoder)
        if counts['total']>=15000000 or counts['trainable']!=110966:raise ValueError('Model parameter contract differs')
        if any(int(s['step'])!=4710 for s in optimizer.state.values()):raise ValueError('Parent AdamW step differs')
        if any(g['lr']!=.001 or g['weight_decay']!=.0001 for g in optimizer.param_groups):raise ValueError('Parent AdamW hyperparameters differ')
        initial=structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})
        initial_optimizer=structured_sha256(optimizer.state_dict())
        manifest.update(status='waiting_for_gpu',parameter_counts=counts,data_identity=problem.identity,
            parent_run=plan['parent'],parent_SHA=plan['parent_sha256'],initial_trainable_SHA=initial,
            optimizer_state_SHA_after_restore=initial_optimizer,optimizer_steps_before=4710,
            encoder_SHA_before=frozen,pretrained_provenance=model.encoder.pretrained_provenance)
        if args.phase=='formal':
            gate=json.loads(args.probe_result.read_text())
            validate_gate(gate,manifest)
            manifest['probe_result_SHA']=file_sha256(args.probe_result)
        write_json(run/'run_manifest.json',manifest)
        if args.device=='cuda':manifest['gpu_admission']=wait_shared(args)
        elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU checks must hide CUDA')
        device=torch.device(args.device);model.to(device)
        optimizer.load_state_dict(old['optimizer'])
        if structured_sha256(optimizer.state_dict())!=initial_optimizer:raise ValueError('Device transfer changed AdamW state')
        if args.phase=='probe':
            ids=[r['sample_id'] for r in problem.train.rows]
            selected=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('T19_resume120/'+ids[i]).encode()).hexdigest())[:64]
            cache,_=original.cache_train(model,Subset(problem.train,selected),'openmoji',device,run,probe=True)
            model.train();losses=[]
            for start in (0,32):
                batch,features=original.cached_batch(cache,list(range(start,start+32)),device)
                optimizer.zero_grad(set_to_none=True)
                loss=balanced_edit_loss(forward_features(model,'openmoji',features,batch['tokens']),batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite continuation smoke loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1);optimizer.step();losses.append(float(loss.detach()))
            audit=assert_encoder_unchanged(model.encoder,frozen)
            if any(int(s['step'])!=4712 for s in optimizer.state.values()):raise ValueError('Smoke optimizer continuation differs')
            if file_sha256(parent)!=plan['parent_sha256']:raise ValueError('Parent changed during smoke')
            write_json(run/'result.json',dict(status='complete',gate_passed=True,test_evaluations=0,TRAIN_samples=64,
                losses=losses,smoke_steps=2,optimizer_steps_before=4710,optimizer_steps_after=4712,frozen_encoder_audit=audit,
                **{k:manifest[k] for k in ('profile_SHA','source_sha256','native_source_sha256','parent_SHA','initial_trainable_SHA','optimizer_state_SHA_after_restore')}))
            manifest['status']='complete';return
        cache,identity=original.cache_train(model,problem.train,'openmoji',device,run,probe=False)
        manifest['TRAIN_cache_identity']=identity
        write_json(run/'TRAIN_parent90_diagnosis.json',dict(epoch=90,metrics=cached_TRAIN_metrics(model,cache,device),
            data='all5000 previously seen TRAIN; not independent validation',checkpoint_SHA=plan['parent_sha256'],new_TEST_evaluations=0))
        loader=index_loader(5000);parent_history=json.loads((parent.parent/'history.json').read_text())
        if [r['epoch'] for r in parent_history]!=list(range(61,91)):raise ValueError('Parent stage sampler history incomplete')
        manifest['sampler_generator_SHA_after_parent30_replay']=replay_loader(loader,cache['sample_id'],parent_history)
        manifest.update(status='training',optimizer_steps_after_parent_replay=4710,
            global_RNG='fresh_seed42; deterministic graph; parent stage index sampler replayed exactly')
        write_json(run/'run_manifest.json',manifest);history=[]
        for epoch in range(91,121):
            model.train();total=count=0;order=[]
            for step,item in enumerate(loader):
                batch,features=original.cached_batch(cache,item.tolist(),device)
                optimizer.zero_grad(set_to_none=True)
                loss=balanced_edit_loss(forward_features(model,'openmoji',features,batch['tokens']),batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite continuation loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1);optimizer.step()
                n=len(item);total+=float(loss.detach())*n;count+=n;order.extend(batch['sample_id'])
                if (step+1)%25==0:write_json(run/'progress.json',dict(status='training',epoch=epoch,step=step+1,train_loss=total/count))
            if count!=5000 or len(set(order))!=5000:raise ValueError('Incomplete TRAIN epoch')
            manifest['frozen_encoder_audits'].append(dict(epoch=epoch,**assert_encoder_unchanged(model.encoder,frozen)))
            entry=dict(epoch=epoch,train_loss=total/count,samples=count,sample_order_SHA=structured_sha256(order),elapsed_seconds=time.monotonic()-began)
            history.append(entry);manifest['last_completed_epoch']=epoch
            save_checkpoint(run/'last_checkpoint.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,manifest=manifest,metrics=entry))
            write_json(run/'history.json',history);write_json(run/'run_manifest.json',manifest);write_json(run/'progress.json',entry)
            print(entry,flush=True)
        if any(int(s['step'])!=9420 for s in optimizer.state.values()):raise ValueError('Final AdamW step differs')
        write_json(run/'TRAIN_last120_diagnosis.json',dict(epoch=120,metrics=cached_TRAIN_metrics(model,cache,device),
            data='all5000 previously seen TRAIN; not independent validation',new_TEST_evaluations=0))
        last_SHA=file_sha256(run/'last_checkpoint.pt')
        model.load_state_dict(torch.load(run/'last_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True)
        raw=[]
        def hook(module,inputs,output):
            raw.append(dict(edit_mask=output['edit_logits'].detach().cpu().ge(0),category=output['category_logits'].detach().cpu().argmax(1),
                            task=output['task_logits'].detach().cpu().argmax(1)))
        handle=model.register_forward_hook(hook)
        test=DataLoader(problem.test,batch_size=32,shuffle=False,num_workers=2,multiprocessing_context='spawn',persistent_workers=True)
        loaders.append(test)
        try:metrics,rows=problem.evaluate(model,test,device)
        finally:handle.remove()
        masks=torch.cat([x['edit_mask'] for x in raw]);categories=torch.cat([x['category'] for x in raw]);tasks=torch.cat([x['task'] for x in raw])
        for i,row in enumerate(rows):row.update(predicted_edit_mask=masks[i].tolist(),generated_category_grid=categories[i].tolist(),predicted_task_index=int(tasks[i]))
        if len(rows)!=1000 or len({r['sample_id'] for r in rows})!=1000:raise ValueError('Incomplete TEST')
        if file_sha256(parent)!=plan['parent_sha256'] or file_sha256(run/'last_checkpoint.pt')!=last_SHA:raise ValueError('Parent/last altered')
        write_json(run/'fixed_weight_predictions.json',rows)
        write_json(run/'result.json',dict(status='complete',task='openmoji',selected_epoch=120,selection='fixed_last_epoch',selection_uses_test=False,
            test_evaluations=1,metrics=metrics,checkpoint_SHA=last_SHA,parent_SHA=plan['parent_sha256'],optimizer_steps_before=4710,
            optimizer_steps_after=9420,new_optimizer_steps=4710,parameter_counts=counts,profile_SHA=manifest['profile_SHA'],
            source_sha256=manifest['source_sha256'],frozen_encoder_audit=assert_encoder_unchanged(model.encoder,frozen),elapsed_seconds=time.monotonic()-began))
        manifest['status']='complete'
    except BaseException as exc:manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['finished_unix']=time.time();write_json(run/'run_manifest.json',manifest)
        for loader in loaders:shutdown_loader(loader)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('probe','formal'),required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--queue-status-path',type=Path,required=True)
    parser.add_argument('--probe-result',type=Path);parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--max-jobs-per-gpu',type=int,default=2)
    run(parser.parse_args())
