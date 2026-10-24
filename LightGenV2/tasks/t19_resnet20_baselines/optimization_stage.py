"""Bounded TRAIN-only fit gate, then fixed-budget task-head optimization.

LSP/OpenMoji preserve their full existing graph and load immutable last60 model
weights. AdamW is explicitly restarted for the changed objective. SALICON uses
an explicit new513-parameter task head; none of the old results are replaced.
"""
import argparse
import copy
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
from torch.utils.data import DataLoader, Subset

from .models import build_model, parameter_counts
from .task_problems import Problem
from .optimization_losses import SmallSaliconHead, forward_features, pose_distribution_loss, balanced_edit_loss
from .frozen_resnet18 import assert_encoder_unchanged, encoder_state_sha256, file_sha256
from .continue_lgvq50 import structured_sha256, shutdown_loader
from .shared_idle_gpu import wait
from .train_easy100 import TASK, git, require_published_source, write_json, save_checkpoint

SOURCES = ('optimization_stage.py','optimization_losses.py','task_problems.py','models.py',
           'other_data.py','frozen_resnet18.py','frozen_visual_models.py','frozen_multimodal_models.py',
           'shared_idle_gpu.py','continue_lgvq50.py','train_easy100.py')
COUNTS={'lsp':(10428,1000),'openmoji':(5000,1000),'salicon':(10000,5000)}
NATIVE_SOURCES={
    'lsp':('experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/datasets.py',
           'experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/metrics.py'),
    'openmoji':('experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/metrics.py',),
    'salicon':('experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/datasets.py',
               'experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency/objectives.py')}


def moved(batch, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def objective(task, output, batch, settings):
    if task == 'lsp':
        return pose_distribution_loss(output,batch['heatmaps'],batch['visible'])
    if task == 'openmoji':
        return balanced_edit_loss(output,batch)
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import saliency_loss
    return saliency_loss(output,batch['density'],batch['fixation'],settings)[0]


def score(task, output, batch):
    if task == 'lsp':
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import PoseMetricAccumulator
        accumulator=PoseMetricAccumulator()
        accumulator.update(output,batch['keypoints'],batch['visible'],batch['torso_scale'],batch['head_scale'],224)
        return accumulator.compute()['pck_at_0.2_torso']
    if task == 'openmoji':
        from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
        accumulator=MetricAccumulator();accumulator.update(output,batch)
        return accumulator.compute()['overall']['changed_cell_accuracy']
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import density_from_logits,correlation_coefficient
    return float(correlation_coefficient(density_from_logits(output),batch['density']))


def cache_train(model, dataset, task, device, run, *, probe):
    """Only deterministic TRAIN data; never cache one realization of augmented full LSP."""
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=2,
                      multiprocessing_context='spawn',persistent_workers=True)
    pieces=[];columns={};ids=[];equivalence=[]
    model.eval()
    try:
        for step,batch in enumerate(loader):
            image=batch['image'].to(device)
            levels=model.encoder.features(image)
            kept=levels if task=='lsp' else (levels[-1],)
            pieces.append(tuple(x.cpu() for x in kept))
            for key,value in batch.items():
                if key!='image':columns.setdefault(key,[]).append(value)
            ids.extend(batch['sample_id'])
            if step==0:
                with torch.no_grad():
                    via=forward_features(model,task,levels,batch.get('tokens',None).to(device) if 'tokens' in batch else None)
                    direct=model(image,batch['tokens'].to(device)) if task=='openmoji' else model(image)
                for key in via if isinstance(via,dict) else (None,):
                    a,b=(via[key],direct[key]) if key else (via,direct)
                    torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
                equivalence.append(dict(batch=step,head_matches_full_forward=True,absolute_tolerance=1e-6))
            if step%25==0:write_json(run/'progress.json',dict(status='caching_TRAIN',samples=len(ids)))
    finally:shutdown_loader(loader)
    result={k:torch.cat(v) if torch.is_tensor(v[0]) else sum((list(x) for x in v),[]) for k,v in columns.items()}
    result['features']=tuple(torch.cat([piece[level] for piece in pieces]) for level in range(len(pieces[0])))
    if len(ids)!=len(dataset) or len(set(ids))!=len(ids):raise RuntimeError('Cache TRAIN identities incomplete')
    identity=dict(samples=len(ids),ids_SHA=structured_sha256(ids),features_SHA=structured_sha256(result['features']),
                  labels_SHA=structured_sha256({k:v for k,v in result.items() if k!='features'}),
                  FP32=True,deterministic_TRAIN_only=True,probe_only=probe,forward_equivalence=equivalence)
    write_json(run/'train_cache_identity.json',identity)
    return result,identity


def cached_batch(cache, indices, device):
    batch={k:v[indices].to(device) if torch.is_tensor(v) else [v[i] for i in indices]
           for k,v in cache.items() if k!='features'}
    features=tuple(x[indices].to(device) for x in cache['features'])
    return batch,features


def run(args):
    require_published_source()
    plan=json.loads(args.plan.read_text());spec=plan['tasks'][args.task]
    run=args.run_dir.resolve()
    category='smoke' if args.phase=='probe' else 'simulation'
    if not run.is_relative_to((TASK/'runs'/category).resolve()):raise ValueError('Run outside task')
    run.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='preflight',task=args.task,phase=args.phase,pid=os.getpid(),started_unix=time.time(),
        command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],git_commit=git('rev-parse','HEAD'),
        source_sha256={name:file_sha256(TASK/name) for name in SOURCES},profile_SHA=file_sha256(args.plan),
        native_source_sha256={name:file_sha256(Path(__file__).resolve().parents[3]/name) for name in NATIVE_SOURCES[args.task]},
        python=sys.version,torch=torch.__version__,numpy=np.__version__,
        seed=42,learning_rate=.001,schedule='constant',warmup_epochs=0,effective_batch=32,
        precision='FP32_TF32_disabled',loss_profile=spec['loss'],optimizer_restart=True,
        test_selected_checkpoint=False,test_evaluations=0,historical_TEST_observed=True,
        gpu_uuid=args.gpu_uuid,model_graph_changed=args.task=='salicon')
    write_json(run/'run_manifest.json',manifest)
    loaders=[];began=time.monotonic()
    try:
        torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        problem=Problem(args.task)
        if (len(problem.train),len(problem.test))!=COUNTS[args.task]:raise ValueError('Original full dataset counts differ')
        model=SmallSaliconHead() if args.task=='salicon' else build_model(args.task,backbone='resnet18_imagenet_frozen',
            **({'text_encoder':'byte_gru'} if args.task=='openmoji' else {}))
        parent=None;parent_sha=None
        if spec['parent']:
            parent=TASK/'runs/simulation'/spec['parent']/'last_checkpoint.pt'
            parent_sha=file_sha256(parent)
            if parent_sha!=spec['parent_sha256']:raise ValueError('Immutable last60 parent SHA differs')
            payload=torch.load(parent,map_location='cpu',weights_only=False)
            if payload['epoch']!=60 or payload['manifest']['data_identity']!=problem.identity:
                raise ValueError('Parent budget/data identity differs')
            model.load_state_dict(payload['model'],strict=True)
        frozen=encoder_state_sha256(model.encoder)
        counts=parameter_counts(model)
        if counts['total']>=15000000 or (args.task=='salicon' and counts['trainable']!=513):
            raise ValueError('Complete model/head parameter limit violated')
        initial=structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})
        manifest.update(data_identity=problem.identity,parameter_counts=counts,encoder_SHA_before=frozen,
            pretrained_provenance=model.encoder.pretrained_provenance,parent_SHA=parent_sha,
            parent_run=spec['parent'],parent_epoch=60 if parent else 0,initial_trainable_SHA=initial,
            training_scope='frozen_vision_train_GRU_and_head' if args.task=='openmoji' else 'frozen_vision_train_head',
            status='waiting_for_gpu',frozen_encoder_audits=[])
        if args.phase=='formal':
            gate=json.loads(args.probe_result.read_text())
            if (gate['status']!='complete' or not gate['gate_passed'] or gate['task']!=args.task
                or gate['profile_SHA']!=manifest['profile_SHA'] or gate['source_sha256']!=manifest['source_sha256']
                or gate['native_source_sha256']!=manifest['native_source_sha256']
                or gate['initial_trainable_SHA']!=initial or gate['parent_SHA']!=parent_sha or gate['test_evaluations']!=0):
                raise ValueError('Own matching TRAIN-only probe did not pass')
            manifest['probe_result_SHA']=file_sha256(args.probe_result)
        write_json(run/'run_manifest.json',manifest)
        if args.device=='cuda':manifest['gpu_admission']=wait(args)
        elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU validation must hide CUDA')
        device=torch.device(args.device);model.to(device)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=.001,weight_decay=.0001)
        if args.phase=='probe':
            dataset=copy.deepcopy(problem.train) if args.task=='lsp' else problem.train
            if args.task=='lsp':dataset.base.training=False
            ids=[r.sample_id for r in dataset.base.records] if args.task=='lsp' else (
                [r['sample_id'] for r in dataset.rows] if args.task=='openmoji' else [r.sample_id for r in dataset.base.records])
            indices=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('T19_probe42/'+ids[i]).encode()).hexdigest())[:32]
            cache,cache_identity=cache_train(model,Subset(dataset,indices),args.task,device,run,probe=True)
            batch,features=cached_batch(cache,list(range(32)),device)
            with torch.no_grad():baseline=score(args.task,forward_features(model,args.task,features,batch.get('tokens')),batch)
            history=[];model.train()
            for step in range(1,plan['probe_steps']+1):
                optimizer.zero_grad(set_to_none=True)
                output=forward_features(model,args.task,features,batch.get('tokens'))
                loss=objective(args.task,output,batch,problem.settings)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite probe loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1);optimizer.step()
                if step%50==0 or step==plan['probe_steps']:
                    with torch.no_grad():value=score(args.task,forward_features(model,args.task,features,batch.get('tokens')),batch)
                    history.append(dict(step=step,train_loss=float(loss.detach()),train_score=value))
                    write_json(run/'progress.json',dict(status='TRAIN_probe',**history[-1]))
            audit=assert_encoder_unchanged(model.encoder,frozen)
            final=history[-1]['train_score'];threshold=spec['probe_minimum_train_score']
            changed=initial!=structured_sha256({n:p for n,p in model.named_parameters() if p.requires_grad})
            passed=changed and (threshold is None or final>=threshold)
            write_json(run/'result.json',dict(status='complete',task=args.task,phase='probe',gate_passed=passed,
                baseline_TRAIN_score=baseline,final_TRAIN_score=final,history=history,
                initial_trainable_SHA=initial,parent_SHA=parent_sha,source_sha256=manifest['source_sha256'],
                native_source_sha256=manifest['native_source_sha256'],
                profile_SHA=manifest['profile_SHA'],test_evaluations=0,frozen_encoder_audit=audit,
                optimizer_steps=plan['probe_steps'],TRAIN_cache_identity=cache_identity,trainable_changed=changed))
            manifest['status']='complete';return
        cache=None
        if args.task!='lsp':
            if args.task=='salicon' and problem.settings.augmentation_enabled:raise ValueError('Cannot cache random augmentation')
            cache,identity=cache_train(model,problem.train,args.task,device,run,probe=False)
            manifest['TRAIN_cache_identity']=identity
        loader=DataLoader(problem.train if cache is None else range(len(problem.train)),batch_size=problem.batch_size,
            shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=2 if cache is None else 0,
            **(dict(multiprocessing_context='spawn',persistent_workers=True) if cache is None else {}))
        loaders.append(loader);history=[];start=61 if parent else 1
        if spec['final_epoch']-start+1!=plan['epochs_per_stage']:raise ValueError('Declared stage budget differs')
        manifest.update(status='training',first_epoch=start,final_epoch=spec['final_epoch'],
            augmentation_RNG='new_seed42_stage_not_uninterrupted_parent',optimizer_steps_before=0)
        write_json(run/'run_manifest.json',manifest)
        for epoch in range(start,spec['final_epoch']+1):
            model.train();optimizer.zero_grad(set_to_none=True);total=count=0;order=[]
            accumulation=32//problem.batch_size
            for step,item in enumerate(loader):
                if cache is None:
                    batch=moved(item,device);output=model(batch['image'])
                else:
                    batch,features=cached_batch(cache,item.tolist(),device)
                    output=forward_features(model,args.task,features,batch.get('tokens'))
                n=len(batch['sample_id']);loss=objective(args.task,output,batch,problem.settings)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite formal loss')
                group_start=(step//accumulation)*32;group_n=min(32,len(problem.train)-group_start)
                (loss*n/group_n).backward()
                if (step+1)%accumulation==0 or step+1==len(loader):
                    torch.nn.utils.clip_grad_norm_(trainable,1);optimizer.step();optimizer.zero_grad(set_to_none=True)
                total+=float(loss.detach())*n;count+=n;order.extend(batch['sample_id'])
                if (step+1)%25==0:write_json(run/'progress.json',dict(status='training',epoch=epoch,step=step+1,train_loss=total/count))
            if count!=len(problem.train) or len(set(order))!=count:raise RuntimeError('TRAIN epoch identity incomplete')
            audit=assert_encoder_unchanged(model.encoder,frozen)
            manifest['frozen_encoder_audits'].append(dict(epoch=epoch,**audit))
            entry=dict(epoch=epoch,train_loss=total/count,samples=count,sample_order_SHA=structured_sha256(order),
                loss_profile=spec['loss'],elapsed_seconds=time.monotonic()-began)
            history.append(entry);manifest['last_completed_epoch']=epoch
            save_checkpoint(run/'last_checkpoint.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch,manifest=manifest,metrics=entry))
            write_json(run/'history.json',history);write_json(run/'run_manifest.json',manifest)
            write_json(run/'progress.json',dict(status='training',**entry));print(entry,flush=True)
        steps=math.ceil(len(problem.train)/32)*plan['epochs_per_stage']
        if any(int(v['step'])!=steps for v in optimizer.state.values()):raise RuntimeError('Fixed stage optimizer count differs')
        last_SHA=file_sha256(run/'last_checkpoint.pt')
        model.load_state_dict(torch.load(run/'last_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True)
        raw=[]
        def hook(module,inputs,output):
            if args.task=='openmoji':raw.append(dict(edit_mask=output['edit_logits'].detach().cpu().ge(0),
                category=output['category_logits'].detach().cpu().argmax(1),task=output['task_logits'].detach().cpu().argmax(1)))
            elif args.task=='lsp':raw.append(output.detach().cpu())
        handle=model.register_forward_hook(hook) if args.task!='salicon' else model.head.register_forward_hook(
            lambda module,inputs,output:raw.append(output.detach().cpu()))
        test=DataLoader(problem.test,batch_size=problem.batch_size,shuffle=False,num_workers=2,
            multiprocessing_context='spawn',persistent_workers=True);loaders.append(test)
        try:metrics,records=problem.evaluate(model,test,device)
        finally:handle.remove()
        if args.task=='openmoji':
            masks=torch.cat([x['edit_mask'] for x in raw]);categories=torch.cat([x['category'] for x in raw]);tasks=torch.cat([x['task'] for x in raw])
            for i,row in enumerate(records):row.update(predicted_edit_mask=masks[i].tolist(),
                generated_category_grid=categories[i].tolist(),predicted_task_index=int(tasks[i]))
        else:save_checkpoint(run/'fixed_weight_maps.pt',dict(sample_ids=[r['sample_id'] for r in records],logits=torch.cat(raw)))
        expected=COUNTS[args.task][1]
        if len(records)!=expected or len({x['sample_id'] for x in records})!=expected:raise RuntimeError('TEST identity incomplete')
        if file_sha256(run/'last_checkpoint.pt')!=last_SHA or (parent and file_sha256(parent)!=parent_sha):
            raise RuntimeError('Last/parent checkpoint changed')
        write_json(run/'fixed_weight_predictions.json',records)
        write_json(run/'result.json',dict(status='complete',task=args.task,selected_epoch=spec['final_epoch'],
            selection='fixed_last_epoch',selection_uses_test=False,test_evaluations=1,metrics=metrics,
            checkpoint_SHA=last_SHA,parent_SHA=parent_sha,optimizer_steps=steps,parameter_counts=counts,
            frozen_encoder_audit=assert_encoder_unchanged(model.encoder,frozen),profile_SHA=manifest['profile_SHA'],
            source_sha256=manifest['source_sha256'],elapsed_seconds=time.monotonic()-began))
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['finished_unix']=time.time();write_json(run/'run_manifest.json',manifest)
        for loader in loaders:shutdown_loader(loader)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('lsp','openmoji','salicon'),required=True)
    parser.add_argument('--phase',choices=('probe','formal'),required=True)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--queue-status-path',type=Path,required=True)
    parser.add_argument('--probe-result',type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--max-jobs-per-gpu',type=int,default=2)
    run(parser.parse_args())
