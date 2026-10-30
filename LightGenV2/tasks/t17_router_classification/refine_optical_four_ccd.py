"""Validation-only refinement of four-expert optical LG; inference stays strict Top-2."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model_four import FourRouterClassification
from .train_optical_four_ccd import prepare_dataset,evaluate
from .optical_training_methods import augment_rgb_tiles,actual_top2_load_loss,dark_expert_coverage_loss,inverse_sqrt_class_weights,update_phase_ema
from .train import write
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def score(metrics):return metrics['balanced_accuracy'],-metrics['cross_entropy']


def save_predictions(path,ids,labels,p,q,dense):
    np.savez_compressed(path,ids=ids,labels=labels.cpu().numpy(),scores=p,predictions=p.argmax(1),route_power=q,dense_route_power=dense)


def main():
    parser=argparse.ArgumentParser()
    for name in ('profile','data-root','parent','out'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();gpu=require_non_a100_gpu();cfg=json.loads(a.profile.read_text())
    if (cfg['architecture']!='optical_four_top2_ccd' or cfg['experts']!=4 or cfg['OEO_stages']!=2
        or cfg['class_weight_power']!=.5):raise ValueError('inference/training contract')
    if json.loads((a.parent/'status.json').read_text())['state']!='complete':raise ValueError('Parent must finish its predefined budget')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    try:
        parent_sha=sha256(a.parent/'best_checkpoint.pt');parent=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        assert parent_sha==json.loads((a.parent/'test_selection_lock.json').read_text())['checkpoint_sha256']
        load,ids,identity,hashes=prepare_dataset(a.data_root,cfg)
        old=json.loads((a.parent/'metadata.json').read_text())
        for key in ('archive_sha256','manifest_sha256','classes','counts'):assert identity[key]==old['data'][key],key
        task=Path(__file__).resolve().parent;lg=task.parents[1]
        sources=[Path(__file__),task/'model_four.py',task/'train_optical_four_ccd.py',task/'optical_training_methods.py',
                 task/'train_linear_ccd_8class.py',task/'train.py',lg/'demo_check/pure_optical/models.py',
                 lg/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']
        metadata=dict(profile=cfg,profile_sha256=sha256(a.profile),command=sys.argv,pid=os.getpid(),gpu_uuid=gpu,
                      source_commit=os.environ.get('T17_SOURCE_COMMIT'),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      source_sha256={str(p):sha256(p) for p in sources},data=identity,parent_checkpoint_sha256=parent_sha,
                      parent_selected_epoch=parent['epoch'],parent_completed_epochs=60,
                      torch=torch.__version__,test_pixels_read=False,optimizer_reset=True)
        write(a.out/'metadata.json',metadata)
        x={};y={}
        for split in ('train','val'):x[split],y[split]=load(split)
        assert not (set(hashes['train'])&set(hashes['val']))
        model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda();model.load_state_dict(parent['model'])
        assert sum(p.numel() for p in model.parameters())==479364
        ema=copy.deepcopy(model).eval()
        for p in ema.parameters():p.requires_grad_(False)
        weights=inverse_sqrt_class_weights(y['train'],cfg['num_classes'])
        write(a.out/'training_class_weights.json',{'counts':torch.bincount(y['train'],minlength=cfg['num_classes']).tolist(),'weights':weights.tolist()})
        parent_val,*_=evaluate(model,x['val'],y['val'],cfg['batch'])
        for key in ('accuracy','balanced_accuracy','cross_entropy'):assert abs(parent_val[key]-parent['validation'][key])<1e-6
        best=score(parent_val);best_kind='parent';best_epoch=0;last_improvement=0;history=[];started=time.time()
        def checkpoint(chosen,epoch,kind,val):
            return dict(model={k:v.detach().cpu().clone() for k,v in chosen.state_dict().items()},
                        epoch=parent['epoch']+epoch,fine_tune_epoch=epoch,weights_kind=kind,validation=val,
                        config=metadata,parent_checkpoint_sha256=parent_sha)
        torch.save(checkpoint(model,0,'parent',parent_val),a.out/'best_checkpoint.pt')
        optimizer=torch.optim.Adam([{'params':[model.first_phase,model.global_phase],'lr':cfg['phase_lr']},
                                    {'params':[model.router_phase],'lr':cfg['router_lr']}],weight_decay=0)
        rng=np.random.default_rng(cfg['seed']);completed=0
        for epoch in range(1,cfg['refinement_epochs']+1):
            begin=time.time();factor=(cfg['minimum_lr_fraction']+(1-cfg['minimum_lr_fraction'])*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['refinement_epochs']-1))))*min(1.,epoch/3)
            for group,lr in zip(optimizer.param_groups,(cfg['phase_lr'],cfg['router_lr'])):group['lr']=lr*factor
            warm=epoch<=cfg['dense_warmup_epochs'];mix=cfg['initial_uniform_mix']*(1-(epoch-1)/cfg['dense_warmup_epochs']) if warm else 0.
            balance_weight=cfg['balance_loss_weight']*max(0.,1-(epoch-1)/cfg['balance_anneal_epochs'])
            coverage_weight=cfg['coverage_loss_weight']*max(0.,1-(epoch-1)/cfg['balance_anneal_epochs'])
            model.train();order=rng.permutation(len(y['train']));total=0.;supervised_sum=0.;balance_sum=0.;coverage_sum=0.
            gradient_sums=torch.zeros(6,device='cuda');batches=0
            for offset in range(0,len(order),cfg['batch']):
                idx=torch.as_tensor(order[offset:offset+cfg['batch']],device='cuda');images=augment_rgb_tiles(x['train'][idx],cfg['d4_probability'])
                optimizer.zero_grad(set_to_none=True)
                out=model(images,return_dense_route=True,training_route='dense_warmup' if warm else 'top2',training_uniform_mix=mix)
                ce=F.cross_entropy(out['logits'],y['train'][idx],weight=weights,label_smoothing=cfg['label_smoothing'])
                auxiliary=actual_top2_load_loss(out['dense_route_power']);coverage=dark_expert_coverage_loss(out['dense_route_power'])
                loss=ce+balance_weight*auxiliary+coverage_weight*coverage
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('phase gradient failure')
                gradient_sums+=torch.cat((model.first_phase.grad.abs().sum((-2,-1)),
                                          model.global_phase.grad.abs().sum().reshape(1),model.router_phase.grad.abs().sum().reshape(1)))
                batches+=1
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                update_phase_ema(ema,model,cfg['ema_decay'])
                total+=float(loss.detach())*len(idx);supervised_sum+=float(ce.detach())*len(idx);balance_sum+=float(auxiliary.detach())*len(idx)
                coverage_sum+=float(coverage.detach())*len(idx)
            online_val,*_=evaluate(model,x['val'],y['val'],cfg['batch']);ema_val,*_=evaluate(ema,x['val'],y['val'],cfg['batch'])
            kind,chosen,val=max([('online',model,online_val),('ema',ema,ema_val)],key=lambda item:score(item[2]))
            improved=score(val)>best
            if improved:
                best=score(val);best_kind=kind;best_epoch=epoch;last_improvement=epoch
                torch.save(checkpoint(chosen,epoch,kind,val),a.out/'best_checkpoint.pt')
            row=dict(fine_tune_epoch=epoch,training_objective=total/len(order),supervised_objective=supervised_sum/len(order),
                     unweighted_balance_loss=balance_sum/len(order),balance_coefficient=balance_weight,train_route='dense_warmup' if warm else 'top2',
                     unweighted_coverage_loss=coverage_sum/len(order),coverage_coefficient=coverage_weight,
                     uniform_mix=mix,phase_lr=optimizer.param_groups[0]['lr'],router_lr=optimizer.param_groups[1]['lr'],
                     mean_raw_gradient_L1_experts_then_global_router=(gradient_sums/batches).cpu().tolist(),
                     validation=online_val,validation_ema=ema_val,selected=improved,selected_kind=best_kind,selected_fine_tune_epoch=best_epoch)
            if epoch==1 or epoch%cfg['training_evaluate_every']==0:
                row['training'],*_=evaluate(model,x['train'],y['train'],cfg['batch'])
                row['training_ema'],*_=evaluate(ema,x['train'],y['train'],cfg['batch'])
            row['seconds']=time.time()-begin;history.append(row);write(a.out/'history.json',history)
            last=checkpoint(model,epoch,'online',online_val);last.update(ema={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},optimizer=optimizer.state_dict(),shuffle_rng=rng.bit_generator.state)
            torch.save(last,a.out/'last_checkpoint.pt');completed=epoch
            write(a.out/'status.json',dict(state='training',fine_tune_epoch=epoch,maximum=cfg['refinement_epochs'],best_kind=best_kind,best_epoch=best_epoch,best_validation_macro=best[0]))
            print(json.dumps(row),flush=True)
            if epoch>=cfg['minimum_refinement_epochs'] and epoch-last_improvement>=cfg['validation_patience']:break
        chosen=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'])
        lock=dict(checkpoint_sha256=sha256(a.out/'best_checkpoint.pt'),weights_kind=best_kind,selected_fine_tune_epoch=best_epoch,
                  parent_checkpoint_sha256=parent_sha,parent_selected_epoch=parent['epoch'],test_evaluations_before_lock=0,
                  selection='Validation macro recall then CE across parent, online and EMA; no TEST selection')
        write(a.out/'test_selection_lock.json',lock);metrics={}
        for split,key in [('train','train'),('val','validation')]:
            metrics[key],p,q,dense=evaluate(model,x[split],y[split],cfg['batch']);save_predictions(a.out/f'{split}_predictions.npz',ids[split],y[split],p,q,dense)
        if best_kind=='parent':
            assert all(torch.equal(chosen['model'][k],parent['model'][k]) for k in parent['model'])
            previous=json.loads((a.parent/'result.json').read_text());metrics['test']=previous['test'];metrics['test_clean']=previous['test_clean']
            for name in ('test_predictions.npz','test_overlap_audit.json'):shutil.copyfile(a.parent/name,a.out/name)
            test_count=0;test_source='parent predictions reused, identical model tensors'
        else:
            x['test'],y['test']=load('test');metrics['test'],p,q,dense=evaluate(model,x['test'],y['test'],cfg['batch'])
            save_predictions(a.out/'test_predictions.npz',ids['test'],y['test'],p,q,dense)
            seen=set(hashes['train'])|set(hashes['val']);clean=np.array([h not in seen for h in hashes['test']]);labels=y['test'].cpu().numpy();pred=p.argmax(1);classes=cfg['num_classes']
            cm=np.bincount(labels[clean]*classes+pred[clean],minlength=classes*classes).reshape(classes,classes)
            metrics['test_clean']=dict(accuracy=float((pred[clean]==labels[clean]).mean()),balanced_accuracy=float((cm.diagonal()/cm.sum(1)).mean()),cross_entropy=float(-np.log(p[clean,labels[clean]]).mean()),n=int(clean.sum()),confusion_matrix=cm.tolist())
            write(a.out/'test_overlap_audit.json',dict(removed_test_ids=ids['test'][~clean].tolist(),clean_test_ids=ids['test'][clean].tolist()))
            test_count=1;test_source='one new inference after validation weight lock'
        result=dict(**lock,**metrics,completed_base_epochs=60,completed_fine_tune_epochs=completed,total_executed_epoch_budget=60+completed,
                    effective_selected_weight_epoch=parent['epoch']+best_epoch,test_evaluations_this_run=test_count,test_source=test_source,
                    elapsed_seconds=time.time()-started,role='development_followup_optimized_training_recipe_not_equal_tuning_budget_architecture_only_test')
        write(a.out/'result.json',result);write(a.out/'status.json',dict(state='complete',**result));print(json.dumps(result),flush=True)
    except BaseException as error:
        write(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(error)));raise
    finally:
        torch.cuda.empty_cache();write(a.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
