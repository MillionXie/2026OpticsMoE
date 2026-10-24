"""Prespecified50/100-update I2T diagnostics; unchanged original data/model/loss.

Each budget starts independently from the same seed42 initialization. The rest
of the real150-batch sampler is consumed without any model forward/update, so
persistent workers shut down cleanly. This is explicitly sub-epoch training.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch
from . import evaluate_retrieval_budget as base
from .easy100_data import Easy100
from .models import parameter_counts
from .retrieval import cross_modal_loss
from .frozen_resnet18 import assert_encoder_unchanged,encoder_state_sha256,file_sha256
from .train_easy100 import TASK,git,require_published_source,save_checkpoint,write_json


def validate_plan(plan):
    expected=dict(profile='retrieval_subepoch_20261024',task='abo_image_to_text',seed=42,
        optimizer_updates=[50,100],updates_per_complete_epoch=150,device='cpu',batch_size=32,
        learning_rate=.001,weight_decay=.0001,gradient_clip=1.,schedule='constant',warmup_epochs=0,
        cache_source='runs/simulation/retrieval_fixed_budget_cpu_20261024/cache/easy100')
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed sub-epoch plan differs')


def train_prefix(model,optimizer,cache,tokens,title_skus,updates,loader):
    model.train();order=[];losses=[];complete_order=[];done=0
    try:
        for step,indices in enumerate(loader,1):
            indexes=indices.tolist();complete_order.extend(indexes)
            if step>updates:continue
            optimizer.zero_grad(set_to_none=True)
            loss=cross_modal_loss(base.cached_images(model,cache['features'][indices]),model.encode_text(tokens),
                [cache['skus'][i] for i in indexes],title_skus,direction='image_to_text')
            if not torch.isfinite(loss):raise ValueError('Nonfinite TRAIN loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step();order.extend(indexes);losses.append(float(loss.detach()));done+=1
        if done!=updates or sorted(complete_order)!=list(range(len(cache['features']))):
            raise ValueError('Budget or complete original sampler differs')
        return dict(optimizer_updates=done,optimized_sample_indices=order,complete_sampler_indices=complete_order,
            optimized_order_SHA=base.structured_sha256(order),complete_order_SHA=base.structured_sha256(complete_order),TRAIN_losses=losses)
    finally:base.shutdown_loader(loader)


def run(args):
    require_published_source()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU diagnostic must hide CUDA')
    plan=json.loads(args.plan.read_text());validate_plan(plan)
    suite=args.suite_dir.resolve()
    if not suite.is_relative_to((TASK/'runs/simulation').resolve()):raise ValueError('Suite outside T19')
    suite.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    easy=Easy100(args.data_root);model=base.build_budget_model('abo_image_to_text')
    frozen=encoder_state_sha256(model.encoder)
    cache={};receipts={}
    for split in ('train','test'):
        cache[split],receipts[split]=base.load_feature_cache(TASK/plan['cache_source']/split,data_identity=easy.identity,encoder_hash=frozen)
        expected_ids=[row['sample_id'] for row in (easy.train if split=='train' else easy.test)]
        if cache[split]['sample_ids']!=expected_ids:raise ValueError('Original ordered IDs differ')
    if len(cache['train']['features'])!=4800 or len(cache['test']['features'])!=2400:raise ValueError('Original counts differ')
    source={**base.source_identity(),'retrieval_subepoch.py':file_sha256(Path(__file__))}
    anchors=dict(profile_SHA=file_sha256(args.plan),source_sha256=source,cache_SHA={s:receipts[s]['cache_file_sha256'] for s in receipts})
    protocol=dict(plan=plan,**anchors,pid=os.getpid(),git_commit=git('rev-parse','HEAD'),command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],
        complete_model_counts=parameter_counts(model),pretrained_provenance=model.encoder.pretrained_provenance,
        data_identity=easy.identity,all_budgets_retained=True,historical_TEST_observed=True,status='running')
    write_json(suite/'protocol.json',protocol)
    old_initial=json.loads((TASK/'runs/simulation/retrieval_fixed_budget_cpu_20261024/abo_image_to_text_seed42_epoch1/run_manifest.json').read_text())['model_state_sha256_before']
    results=[]
    stages=[('probe',2),('formal',50),('formal',100)]
    for phase,updates in stages:
        folder=(TASK/'runs/smoke/retrieval_subepoch_probe_s42_20261024') if phase=='probe' else suite/f'abo_image_to_text_updates{updates}'
        folder.mkdir(parents=True,exist_ok=False);model=base.build_budget_model('abo_image_to_text')
        initial=encoder_state_sha256(model)
        if initial!=old_initial:raise ValueError('Independent original initialization differs')
        params=[p for p in model.parameters() if p.requires_grad];counts=parameter_counts(model)
        if counts['total']!=11240608 or counts['trainable']!=64096:raise ValueError('Full task parameter contract differs')
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
        manifest=dict(protocol)
        manifest.update(status='training',phase=phase,optimizer_update_budget=updates,
            completed_epochs=0,epoch_fraction=updates/150,model_SHA_before=initial,encoder_SHA_before=frozen,parameter_counts=counts,
            optimizer_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],test_evaluations=0)
        write_json(folder/'run_manifest.json',manifest)
        if phase=='formal':
            gate=json.loads((TASK/'runs/smoke/retrieval_subepoch_probe_s42_20261024/result.json').read_text())
            if gate['status']!='complete' or gate['test_evaluations']!=0 or gate['anchors']!=anchors:raise ValueError('Own TRAIN-only smoke differs')
        began=time.monotonic()
        try:
            history=train_prefix(model,optimizer,cache['train'],easy.title_tokens,easy.title_skus,updates,base.training_index_loader(4800))
            audit=assert_encoder_unchanged(model.encoder,frozen)
            steps=sorted({int(s['step']) for s in optimizer.state.values()})
            if steps!=[updates]:raise ValueError('Actual AdamW steps differ')
            manifest.update(status='trained',frozen_encoder_audit=audit,optimizer_steps=steps,model_SHA_after=encoder_state_sha256(model))
            write_json(folder/'sampler_and_training.json',history)
            if phase=='probe':
                if initial==manifest['model_SHA_after']:raise ValueError('Probe did not update task model')
                write_json(folder/'result.json',dict(status='complete',test_evaluations=0,anchors=anchors,optimizer_updates=updates,frozen_encoder_audit=audit))
                manifest['status']='complete';continue
            save_checkpoint(folder/'last_checkpoint.pt',dict(optimizer_updates=updates,completed_epochs=0,epoch_fraction=updates/150,
                model=model.state_dict(),optimizer=optimizer.state_dict(),manifest=manifest))
            last_SHA=file_sha256(folder/'last_checkpoint.pt')
            saved=torch.load(folder/'last_checkpoint.pt',map_location='cpu',weights_only=False)
            if saved['optimizer_updates']!=updates or base.structured_sha256(saved['optimizer'])!=base.structured_sha256(optimizer.state_dict()):raise ValueError('Saved fixed-last identity differs')
            model.load_state_dict(saved['model'],strict=True)
            if encoder_state_sha256(model)!=manifest['model_SHA_after']:raise ValueError('Fixed-last reload differs')
            predictions=base.evaluate_cached(model,cache['test'],tokens=easy.title_tokens,title_skus=easy.title_skus,direction='image_to_text')
            write_json(folder/'fixed_weight_predictions.json',predictions)
            result=dict(status='complete',optimizer_updates=updates,completed_epochs=0,epoch_fraction=updates/150,
                selection='prespecified_fixed_last_update_budget',selection_uses_test=False,historical_TEST_observed=True,test_evaluations=1,
                checkpoint_SHA=last_SHA,metrics={k:v for k,v in predictions.items() if isinstance(v,(int,float))},parameter_counts=counts,
                model_SHA_before=initial,frozen_encoder_audit=assert_encoder_unchanged(model.encoder,frozen),elapsed_seconds=time.monotonic()-began)
            if file_sha256(folder/'last_checkpoint.pt')!=last_SHA:raise ValueError('Fixed last changed during TEST')
            write_json(folder/'result.json',result);results.append(result);manifest.update(status='complete',test_evaluations=1)
            write_json(suite/'results.json',dict(status='complete' if len(results)==2 else 'running',all_fixed_budgets_retained=True,results=results))
        except BaseException as exc:manifest.update(status='failed',error=repr(exc));raise
        finally:write_json(folder/'run_manifest.json',manifest)
    protocol['status']='complete';write_json(suite/'protocol.json',protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--suite-dir',required=True,type=Path)
    parser.add_argument('--data-root',required=True,type=Path)
    run(parser.parse_args())
