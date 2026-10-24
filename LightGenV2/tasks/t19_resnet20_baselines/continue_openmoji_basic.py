"""One bounded60-to80 model continuation, no candidate/weight/TEST epoch search."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader,Subset
from . import optimization_stage as cache_tools
from .openmoji_validation_repair import fresh_model,regularize_features,validate_gate,grouped_indices,cache_metrics
from .optimization_losses import balanced_edit_loss,forward_features
from .optimization_shared_worker import wait_shared
from .task_problems import Problem
from .models import parameter_counts
from .frozen_resnet18 import file_sha256,encoder_state_sha256,assert_encoder_unchanged
from .continue_lgvq50 import structured_sha256,shutdown_loader
from .train_easy100 import TASK,ROOT,git,require_published_source,write_json,save_checkpoint


def validate_plan(plan):
    if (plan['profile']!='openmoji_basic_resume60to80_20261024' or (plan['parent_epoch'],plan['final_epoch'])!=(60,80)
        or plan['feature_dropout']!=.2 or plan['learning_rate']!=.001 or plan['weight_decay']!=.0001 or plan['seed']!=42
        or plan['effective_batch']!=32 or plan['gradient_clip']!=1
        or plan['parent']!='runs/simulation/openmoji_validation_formal_s42_20261024_native_objective1/last_checkpoint.pt'
        or plan['parent_SHA']!='4ea842276d6ceb28b98210cd9b0b19253495238f6bd36d54b65cf841656003ef'):
        raise ValueError('Fixed continuation contract differs')


def run(args):
    require_published_source();plan=json.loads(args.plan.read_text());validate_plan(plan)
    run=args.run_dir.resolve();category='smoke' if args.phase=='probe' else 'simulation'
    if not run.is_relative_to((TASK/'runs'/category).resolve()):raise ValueError('Run outside T19')
    run.mkdir(parents=True,exist_ok=False);parent=TASK/plan['parent']
    if file_sha256(parent)!=plan['parent_SHA']:raise ValueError('Protected parent60 identity differs')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model=fresh_model('cpu');official=encoder_state_sha256(model.encoder);payload=torch.load(parent,map_location='cpu',weights_only=False)
    if payload['epoch']!=60:raise ValueError('Wrong parent epoch')
    model.load_state_dict(payload['model'],strict=True)
    assert_encoder_unchanged(model.encoder,official);counts=parameter_counts(model)
    if counts['total']!=11287478 or counts['trainable']!=110966:raise ValueError('Task graph/parameters differ')
    initial=structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})
    source=(*cache_tools.SOURCES,'openmoji_validation_repair.py','optimization_shared_worker.py','continue_openmoji_basic.py')
    problem=Problem('openmoji')
    if (len(problem.train),len(problem.test))!=(5000,1000):raise ValueError('Original full dataset differs')
    fit,val,groups=grouped_indices(problem.train.rows)
    fold=json.loads((parent.parent/'grouped_TRAIN_validation_split.json').read_text())
    fold_SHA=structured_sha256(fold)
    if (fold['TRAIN_ids']!=[problem.train.rows[i]['sample_id'] for i in fit]
        or fold['validation_ids']!=[problem.train.rows[i]['sample_id'] for i in val]
        or fold['source_group_assignments']!=groups or fold_SHA!=payload['manifest']['fold_SHA']):raise ValueError('Parent grouped fit/validation partition differs')
    manifest=dict(status='preflight',phase=args.phase,task='openmoji',pid=os.getpid(),gpu_uuid=args.gpu_uuid,
        command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],git_commit=git('rev-parse','HEAD'),
        source_sha256={n:file_sha256(TASK/n) for n in source},native_source_sha256={n:file_sha256(ROOT/n) for n in cache_tools.NATIVE_SOURCES['openmoji']},
        profile_SHA=file_sha256(args.plan),parent_SHA=plan['parent_SHA'],parent=str(parent),parent_epoch=60,final_epoch=80,
        initial_trainable_SHA=initial,fold_SHA=fold_SHA,
        parameter_counts=counts,data_identity=problem.identity,pretrained_provenance=model.encoder.pretrained_provenance,fit_samples=len(fit),validation_samples=len(val),
        optimizer_restart=True,sampler_new_stage_seed=42,feature_dropout=.2,loss_preserve_coefficient=.25,
        seed=42,learning_rate=.001,weight_decay=.0001,schedule='constant',warmup_epochs=0,effective_batch=32,precision='FP32_TF32_disabled',
        started_unix=time.time(),test_evaluations=0,historical_TEST_observed=True)
    write_json(run/'run_manifest.json',manifest);loaders=[];began=time.monotonic()
    try:
        if args.phase=='formal':validate_gate(json.loads(args.probe_result.read_text()),manifest)
        manifest['gpu_admission']=wait_shared(args);device=torch.device('cuda');model.to(device)
        write_json(run/'run_manifest.json',manifest)
        dataset=Subset(problem.train,fit[:64]) if args.phase=='probe' else problem.train
        cache,cache_identity=cache_tools.cache_train(model,dataset,'openmoji',device,run,probe=args.phase=='probe')
        params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
        if args.phase=='probe':
            model.train();losses=[]
            for start in (0,32):
                batch,features=cache_tools.cached_batch(cache,list(range(start,start+32)),device);optimizer.zero_grad(set_to_none=True)
                output=forward_features(model,'openmoji',regularize_features(features,.2,True),batch['tokens'])
                loss=balanced_edit_loss(output,batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1);optimizer.step();losses.append(float(loss.detach()))
            audit=assert_encoder_unchanged(model.encoder,official)
            if structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})==initial:raise ValueError('Probe did not update task weights')
            write_json(run/'result.json',dict(status='complete',gate_passed=True,test_evaluations=0,losses=losses,frozen_encoder_audit=audit,
                **{k:manifest[k] for k in ('profile_SHA','source_sha256','native_source_sha256','initial_trainable_SHA','fold_SHA')}))
            manifest['status']='complete';return
        manifest.update(status='training',TRAIN_cache_identity=cache_identity);write_json(run/'run_manifest.json',manifest)
        loader=DataLoader(fit,batch_size=32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
        history=[];audits=[]
        for epoch in range(61,81):
            model.train();total=count=0;order=[]
            for item in loader:
                batch,features=cache_tools.cached_batch(cache,item.tolist(),device);optimizer.zero_grad(set_to_none=True)
                output=forward_features(model,'openmoji',regularize_features(features,.2,True),batch['tokens'])
                loss=balanced_edit_loss(output,batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite formal loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1);optimizer.step()
                n=len(item);total+=float(loss.detach())*n;count+=n;order.extend(batch['sample_id'])
            if count!=len(fit) or set(order)!=set(fold['TRAIN_ids']):raise ValueError('Fit-only sampler differs')
            audits.append(dict(epoch=epoch,**assert_encoder_unchanged(model.encoder,official)))
            entry=dict(epoch=epoch,train_loss=total/count,samples=count,sample_order_SHA=structured_sha256(order),elapsed_seconds=time.monotonic()-began)
            history.append(entry)
            save_checkpoint(run/'last_checkpoint.pt',dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),manifest={**manifest,'frozen_audits':audits},metrics=entry))
            write_json(run/'history.json',history);write_json(run/'progress.json',entry)
        if any(int(s['step'])!=math.ceil(len(fit)/32)*20 for s in optimizer.state.values()):raise ValueError('New-stage update budget differs')
        val_metrics,val_loss=cache_metrics(model,cache,val,device)
        write_json(run/'validation_result.json',dict(epoch=80,validation_metrics=val_metrics,validation_loss=val_loss,selection_performed=False))
        last_SHA=file_sha256(run/'last_checkpoint.pt');model.load_state_dict(torch.load(run/'last_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True)
        raw=[]
        def hook(module,inputs,output):raw.append(dict(edit=output['edit_logits'].detach().cpu().ge(0),category=output['category_logits'].detach().cpu().argmax(1),task=output['task_logits'].detach().cpu().argmax(1)))
        handle=model.register_forward_hook(hook)
        test=DataLoader(problem.test,batch_size=32,shuffle=False,num_workers=2,multiprocessing_context='spawn',persistent_workers=True);loaders.append(test)
        try:metrics,rows=problem.evaluate(model,test,device)
        finally:handle.remove()
        edit=torch.cat([r['edit'] for r in raw]);category=torch.cat([r['category'] for r in raw]);tasks=torch.cat([r['task'] for r in raw])
        for i,row in enumerate(rows):row.update(predicted_edit_mask=edit[i].tolist(),generated_category_grid=category[i].tolist(),predicted_task_index=int(tasks[i]))
        if len(rows)!=1000 or len({r['sample_id'] for r in rows})!=1000:raise ValueError('Incomplete original TEST')
        write_json(run/'fixed_weight_predictions.json',rows)
        if file_sha256(parent)!=plan['parent_SHA'] or file_sha256(run/'last_checkpoint.pt')!=last_SHA:raise ValueError('Fixed weights changed')
        write_json(run/'result.json',dict(status='complete',selected_epoch=80,parent_epoch=60,additional_epochs=20,optimizer_restart=True,
            optimizer_new_stage_steps=2500,metrics=metrics,checkpoint_SHA=last_SHA,parent_SHA=plan['parent_SHA'],parameter_counts=counts,
            test_evaluations=1,selection='prespecified_fixed_last80',selection_uses_test=False,frozen_encoder_audit=assert_encoder_unchanged(model.encoder,official)))
        manifest.update(status='complete',test_evaluations=1)
    except BaseException as exc:manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['finished_unix']=time.time();write_json(run/'run_manifest.json',manifest)
        for loader in loaders:shutdown_loader(loader)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('probe','formal'));parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--queue-status-path',required=True,type=Path);parser.add_argument('--probe-result',type=Path)
    parser.add_argument('--max-jobs-per-gpu',type=int,default=2)
    run(parser.parse_args())
