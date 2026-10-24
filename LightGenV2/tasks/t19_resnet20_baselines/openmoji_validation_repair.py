"""Fixed source-scene-grouped TRAIN validation experiments.

Same frozen visual backbone, GRU and task head. Head/GRU start afresh so the
internal validation scenes were not previously fitted by the initialization.
Only the VAL-selected candidate's fixed last60 is run on original TEST once.
The separately named equal-preservation profile has one prespecified candidate;
its VAL report does not select an epoch or enforce a desired TEST interval.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from . import optimization_stage as original
from .optimization_losses import balanced_edit_loss,forward_features
from .optimization_shared_worker import wait_shared
from .models import build_model,parameter_counts
from .task_problems import Problem
from .continue_lgvq50 import structured_sha256,shutdown_loader
from .frozen_resnet18 import file_sha256,encoder_state_sha256,assert_encoder_unchanged
from .train_easy100 import TASK,git,require_published_source,write_json,save_checkpoint

SOURCES=(*original.SOURCES,'optimization_shared_worker.py','openmoji_validation_repair.py')


def grouped_indices(rows):
    training=[];validation=[];groups={}
    for index,row in enumerate(rows):
        key=json.dumps(row['source_grid'],separators=(',',':'))
        digest=hashlib.sha256(('T19_OpenMoji_source_validation42/'+key).encode()).hexdigest()
        fold=int(digest,16)%5==0;groups[digest]=fold
        (validation if fold else training).append(index)
    return training,validation,groups


def regularize_features(features,probability,training):
    if not 0<=probability<1:raise ValueError('Invalid feature dropout')
    return tuple(F.dropout2d(x,p=probability,training=training) for x in features)


def select_candidate(results):
    return min(results,key=lambda r:(-r['validation_metrics']['overall']['changed_cell_accuracy'],r['validation_loss'],r['candidate']['id']))


def candidate_contract(plan):
    if plan.get('profile')=='openmoji_equal_preservation_20261024':
        expected=[dict(id='equal_changed_preserved',feature_dropout=.2,weight_decay=.0001,preserve_weight=1.)]
    elif plan.get('profile')=='openmoji_native_objective_20261024':
        expected=[dict(id='native_original',feature_dropout=.2,weight_decay=.0001,objective='native_original')]
    elif 'profile' not in plan:
        expected=[dict(id='plain',feature_dropout=0.,weight_decay=.0001),
                  dict(id='feature_dropout20',feature_dropout=.2,weight_decay=.0001),
                  dict(id='feature_dropout20_wd10',feature_dropout=.2,weight_decay=.001)]
    else:raise ValueError('Unknown fixed experimental profile')
    if plan['candidates']!=expected:raise ValueError('Predeclared candidates differ')
    return expected


def candidate_loss(output,batch,candidate,settings=None):
    if candidate.get('objective')=='native_original':
        if settings is None:raise ValueError('Native task loss requires explicit original settings')
        from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.objectives import editing_objective
        zero=output['category_logits'].new_zeros(())
        return editing_objective(dict(output,ccd_operating_loss=zero,router_balance_loss=zero),batch,settings)['total']
    return balanced_edit_loss(output,batch,preserve_weight=candidate.get('preserve_weight',.25))


def validate_gate(gate,manifest):
    for key in ('profile_SHA','source_sha256','native_source_sha256','initial_trainable_SHA','fold_SHA'):
        if gate[key]!=manifest[key]:raise ValueError('Own validation-experiment probe identity differs: '+key)
    if gate['status']!='complete' or not gate['gate_passed'] or gate['test_evaluations']!=0:
        raise ValueError('TRAIN-only probe required')


def fresh_model(device):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    return build_model('openmoji',backbone='resnet18_imagenet_frozen',text_encoder='byte_gru').to(device)


@torch.no_grad()
def cache_metrics(model,cache,indices,device,*,preserve_weight=.25,candidate=None,settings=None):
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    accumulator=MetricAccumulator();total=0;count=0;model.eval()
    for start in range(0,len(indices),32):
        item=indices[start:start+32];batch,features=original.cached_batch(cache,item,device)
        output=forward_features(model,'openmoji',features,batch['tokens'])
        loss=candidate_loss(output,batch,candidate or dict(preserve_weight=preserve_weight),settings);total+=float(loss)*len(item);count+=len(item)
        accumulator.update(output,batch)
    return accumulator.compute(),total/count


def run(args):
    require_published_source();plan=json.loads(args.plan.read_text())
    expected=candidate_contract(plan)
    native_sources=original.NATIVE_SOURCES['openmoji']
    if any(c.get('objective')=='native_original' for c in expected):
        native_sources=(*native_sources,'experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/objectives.py',
            'experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/settings.py')
    if plan['candidates']!=expected or plan['epochs']!=60 or plan['seed']!=42 or plan['learning_rate']!=.001:
        raise ValueError('Predeclared candidate set or budget differs')
    if (plan['validation_group']!='source_grid' or plan['validation_hash_modulus']!=5
            or plan['effective_batch']!=32 or plan['gradient_clip']!=1):
        raise ValueError('Grouped split or optimization contract differs')
    category='smoke' if args.phase=='probe' else 'simulation';run=args.run_dir.resolve()
    if not run.is_relative_to((TASK/'runs'/category).resolve()):raise ValueError('Run outside task')
    run.mkdir(parents=True,exist_ok=False)
    m=dict(status='preflight',task='openmoji',phase=args.phase,pid=os.getpid(),started_unix=time.time(),
        command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],git_commit=git('rev-parse','HEAD'),
        source_sha256={n:file_sha256(TASK/n) for n in SOURCES},profile_SHA=file_sha256(args.plan),
        native_source_sha256={n:file_sha256(Path(__file__).resolve().parents[3]/n) for n in native_sources},
        gpu_uuid=args.gpu_uuid,seed=42,learning_rate=.001,schedule='constant',warmup_epochs=0,effective_batch=32,
        precision='FP32_TF32_disabled',historical_TEST_observed=True,old_trained_head_loaded=False,
        initialization='official frozen visual; fresh seed42 GRU and unchanged task head',test_selected_checkpoint=False)
    write_json(run/'run_manifest.json',m);loaders=[];began=time.monotonic()
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        problem=Problem('openmoji')
        if (len(problem.train),len(problem.test))!=(5000,1000):raise ValueError('Original data counts differ')
        training,validation,groups=grouped_indices(problem.train.rows)
        if not 3000<=len(training)<=4500 or not 500<=len(validation)<=2000:raise ValueError('Grouped fold count unexpected')
        train_ids=[problem.train.rows[i]['sample_id'] for i in training];val_ids=[problem.train.rows[i]['sample_id'] for i in validation]
        if set(train_ids)&set(val_ids):raise ValueError('Fold IDs overlap')
        fold=dict(TRAIN_ids=train_ids,validation_ids=val_ids,source_group_assignments=groups,
            construction='hash of original TRAIN source_grid; no original TEST rows used',source_group_overlap=0)
        fold_SHA=structured_sha256(fold);write_json(run/'grouped_TRAIN_validation_split.json',fold)
        model=fresh_model('cpu');counts=parameter_counts(model);frozen=encoder_state_sha256(model.encoder)
        if counts['total']>=15000000 or counts['trainable']!=110966:raise ValueError('Original complete model contract differs')
        initial=structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})
        m.update(status='waiting_for_gpu',data_identity=problem.identity,parameter_counts=counts,encoder_SHA_before=frozen,
            initial_trainable_SHA=initial,fold_SHA=fold_SHA,fit_samples=len(training),validation_samples=len(validation),
            pretrained_provenance=model.encoder.pretrained_provenance)
        if args.phase=='formal':validate_gate(json.loads(args.probe_result.read_text()),m)
        write_json(run/'run_manifest.json',m)
        if args.device=='cuda':m['gpu_admission']=wait_shared(args)
        elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU validation must hide CUDA')
        device=torch.device(args.device);model.to(device)
        if args.phase=='probe':
            chosen=sorted(training,key=lambda i:hashlib.sha256(('T19_Open_val_probe/'+problem.train.rows[i]['sample_id']).encode()).hexdigest())[:64]
            cache,_=original.cache_train(model,Subset(problem.train,chosen),'openmoji',device,run,probe=True)
            reports=[]
            for candidate in expected:
                model=fresh_model(device)
                if structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})!=initial:raise ValueError('Candidate initialization differs')
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=candidate['weight_decay'])
                model.train();losses=[]
                for start in (0,32):
                    batch,features=original.cached_batch(cache,list(range(start,start+32)),device)
                    optimizer.zero_grad(set_to_none=True)
                    output=forward_features(model,'openmoji',regularize_features(features,candidate['feature_dropout'],True),batch['tokens'])
                    loss=candidate_loss(output,batch,candidate,problem.settings)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite candidate probe loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step();losses.append(float(loss.detach()))
                reports.append(dict(candidate=candidate,losses=losses,frozen_encoder_audit=assert_encoder_unchanged(model.encoder,frozen),steps=2))
            write_json(run/'result.json',dict(status='complete',gate_passed=True,test_evaluations=0,validation_evaluations=0,
                reports=reports,**{k:m[k] for k in ('profile_SHA','source_sha256','native_source_sha256','initial_trainable_SHA','fold_SHA')}))
            m['status']='complete';return
        cache,cache_identity=original.cache_train(model,problem.train,'openmoji',device,run,probe=False)
        m.update(status='training',TRAIN_validation_cache_identity=cache_identity);write_json(run/'run_manifest.json',m)
        model.cpu();del model;results=[]
        for candidate in expected:
            model=fresh_model(device);parameters=[p for p in model.parameters() if p.requires_grad]
            if structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})!=initial:raise ValueError('Candidate initial state differs')
            optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=candidate['weight_decay'])
            loader=DataLoader(training,batch_size=32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
            folder=run/'candidates'/candidate['id'];folder.mkdir(parents=True,exist_ok=False);history=[];audits=[]
            for epoch in range(1,61):
                total=count=0;order=[];model.train()
                for step,item in enumerate(loader):
                    batch,features=original.cached_batch(cache,item.tolist(),device);optimizer.zero_grad(set_to_none=True)
                    output=forward_features(model,'openmoji',regularize_features(features,candidate['feature_dropout'],True),batch['tokens'])
                    loss=candidate_loss(output,batch,candidate,problem.settings)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite candidate loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
                    n=len(item);total+=float(loss.detach())*n;count+=n;order.extend(batch['sample_id'])
                if count!=len(training) or set(order)!=set(train_ids):raise ValueError('TRAIN fold incomplete or contaminated')
                audit=assert_encoder_unchanged(model.encoder,frozen);audits.append(dict(epoch=epoch,**audit))
                entry=dict(epoch=epoch,train_loss=total/count,samples=count,sample_order_SHA=structured_sha256(order),elapsed_seconds=time.monotonic()-began)
                history.append(entry)
                save_checkpoint(folder/'last_checkpoint.pt',dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    manifest={**m,'candidate':candidate,'candidate_frozen_audits':audits},metrics=entry))
                write_json(folder/'history.json',history);write_json(run/'progress.json',dict(status='training',candidate=candidate['id'],**entry))
            expected_steps=math.ceil(len(training)/32)*60
            if any(int(s['step'])!=expected_steps for s in optimizer.state.values()):raise ValueError('Candidate budget step mismatch')
            val_metrics,val_loss=cache_metrics(model,cache,validation,device,candidate=candidate,settings=problem.settings)
            TRAIN_metrics,_=cache_metrics(model,cache,training,device,candidate=candidate,settings=problem.settings)
            result=dict(candidate=candidate,epoch=60,optimizer_steps=expected_steps,validation_metrics=val_metrics,validation_loss=val_loss,
                TRAIN_metrics=TRAIN_metrics,checkpoint_SHA=file_sha256(folder/'last_checkpoint.pt'),test_evaluations=0)
            write_json(folder/'validation_result.json',result);results.append(result);model.cpu();del model,optimizer
        winner=select_candidate(results);selected=run/'candidates'/winner['candidate']['id']/'last_checkpoint.pt'
        write_json(run/'validation_selection.json',dict(selection_uses_test=False,results=results,winner=winner['candidate']['id'],rule=plan['selection']))
        shutil.copyfile(selected,run/'last_checkpoint.pt');last_SHA=file_sha256(run/'last_checkpoint.pt')
        if last_SHA!=winner['checkpoint_SHA']:raise ValueError('Selected fixed last was not copied exactly')
        model=fresh_model(device);model.load_state_dict(torch.load(run/'last_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True)
        raw=[]
        def hook(module,inputs,output):raw.append(dict(edit_mask=output['edit_logits'].detach().cpu().ge(0),
            category=output['category_logits'].detach().cpu().argmax(1),task=output['task_logits'].detach().cpu().argmax(1)))
        handle=model.register_forward_hook(hook)
        test=DataLoader(problem.test,batch_size=32,shuffle=False,num_workers=2,multiprocessing_context='spawn',persistent_workers=True);loaders.append(test)
        try:metrics,rows=problem.evaluate(model,test,device)
        finally:handle.remove()
        masks=torch.cat([x['edit_mask'] for x in raw]);categories=torch.cat([x['category'] for x in raw]);tasks=torch.cat([x['task'] for x in raw])
        for i,row in enumerate(rows):row.update(predicted_edit_mask=masks[i].tolist(),generated_category_grid=categories[i].tolist(),predicted_task_index=int(tasks[i]))
        if len(rows)!=1000 or len({r['sample_id'] for r in rows})!=1000:raise ValueError('Incomplete original TEST')
        write_json(run/'fixed_weight_predictions.json',rows)
        write_json(run/'result.json',dict(status='complete',selected_epoch=60,selection=plan['selection'],
            selection_uses_test=False,test_evaluations=1,selected_candidate=winner['candidate'],metrics=metrics,checkpoint_SHA=last_SHA,
            parameter_counts=counts,fit_samples=len(training),validation_samples=len(validation),fold_SHA=fold_SHA,
            source_sha256=m['source_sha256'],profile_SHA=m['profile_SHA'],frozen_encoder_audit=assert_encoder_unchanged(model.encoder,frozen)))
        m.update(status='complete',selected_candidate=winner['candidate']['id'])
    except BaseException as exc:m.update(status='failed',error=repr(exc));raise
    finally:
        m['finished_unix']=time.time();write_json(run/'run_manifest.json',m)
        for loader in loaders:shutdown_loader(loader)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('probe','formal'),required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--queue-status-path',type=Path,required=True);parser.add_argument('--probe-result',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');parser.add_argument('--max-jobs-per-gpu',type=int,default=2)
    run(parser.parse_args())
