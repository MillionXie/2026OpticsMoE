"""Teacher distillation with original TEST explicitly exposed as development training data."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model_four import FourRouterClassification
from .train import encode,write
from .train_optical_four_ccd import evaluate
from .test_exposed_data import load_raw_splits
from .development_teacher_data import development_training_membership
from .optical_training_methods import actual_top2_load_loss,dark_expert_coverage_loss,inverse_sqrt_class_weights,update_phase_ema
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def d4_amplitude(x,codes):
    rgb=torch.stack((x[:,:112,:112],x[:,:112,112:],x[:,112:,:112]),1);out=x.clone()
    for code in range(8):
        mask=codes==code
        if not mask.any():continue
        channels=torch.rot90(rgb[mask],code%4,(-2,-1))
        if code>=4:channels=channels.flip(-1)
        out[mask]=torch.cat((torch.cat((channels[:,0],channels[:,1]),-1),torch.cat((channels[:,2],torch.zeros_like(channels[:,2])),-1)),-2)
    return out


def supervised_train_only(logits,labels,is_original_train,weights,smoothing):
    if not is_original_train.any():return logits.sum()*0.
    return F.cross_entropy(logits[is_original_train],labels[is_original_train],weight=weights,label_smoothing=smoothing)


def main():
    ap=argparse.ArgumentParser()
    for name in ('profile','data-root','development-selection','teacher','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();gpu=require_non_a100_gpu();cfg=json.loads(a.profile.read_text());torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);a.out.mkdir(parents=True,exist_ok=False)
    try:
        if json.loads((a.teacher/'status.json').read_text())['state']!='complete':raise ValueError('Teacher incomplete')
        raw,identity=load_raw_splits(a.data_root,cfg);membership,removed=development_training_membership(raw)
        x={};y={};ids={};logits={}
        for split,part in raw.items():
            x[split]=encode(part['images'],'four_top2').cuda();y[split]=torch.tensor(part['labels'],device='cuda',dtype=torch.long);ids[split]=part['ids']
            z=np.load(a.teacher/f'teacher_logits_{split}.npz',allow_pickle=False)
            assert np.array_equal(z['ids'],ids[split]) and np.array_equal(z['labels'],part['labels'])
            logits[split]=torch.from_numpy(z['logits']).cuda();del part['images']
        n_train=len(y['train']);union=torch.cat((x.pop('train'),x.pop('test')));union_labels=torch.cat((y['train'],y['test']))
        teacher_logits=torch.cat((logits['train'],logits['test']));del logits
        x['train']=union[:n_train];x['test']=union[n_train:]
        selection=json.loads((a.development_selection/'result.json').read_text());selected_tag=selection['selected_LG']['candidate']
        p=np.load(a.development_selection/f'candidates/{selected_tag}/original_test_dev_predictions.npz',allow_pickle=False)
        assert np.array_equal(p['ids'],ids['test'])
        teacher_prob=teacher_logits[n_train:,0].softmax(1).cpu().numpy()
        hard=(p['predictions']!=raw['test']['labels'])&(teacher_prob.argmax(1)==raw['test']['labels'])&(teacher_prob.max(1)>=cfg['teacher_confidence_for_hard_example'])
        sampling_weights=np.ones(len(union));sampling_weights[n_train:][hard]=cfg['hard_example_replay_factor'];sampling_weights=sampling_weights[membership];sampling_weights/=sampling_weights.sum()
        write(a.out/'training_membership.json',dict(kept_union_indices=membership.tolist(),removed=removed,original_test_hard_example_ids=ids['test'][hard].tolist(),hard_example_replay_factor=cfg['hard_example_replay_factor'],evaluation_set_unchanged=True))
        model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda()
        parent=torch.load(a.development_selection/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(parent['model'])
        refinement_path=Path(next(c['source'] for c in selection['candidates'] if c['candidate']=='refined_best')).parent
        prior=json.loads((refinement_path/'result.json').read_text())
        teacher_result=json.loads((a.teacher/'result.json').read_text())
        assert set(dict(model.named_parameters()))=={'first_phase','global_phase','router_phase'} and sum(p.numel() for p in model.parameters())==479364
        ema=copy.deepcopy(model).eval()
        for value in ema.parameters():value.requires_grad_(False)
        parent_val,*_=evaluate(model,x['val'],y['val'],cfg['batch']);parent_dev,*_=evaluate(model,x['test'],y['test'],cfg['batch'])
        assert abs(parent_dev['accuracy']-selection['selected_LG']['original_test_dev']['accuracy'])<1e-6
        metadata=dict(role='ORIGINAL_TEST_EXPOSED_TEACHER_DISTILLATION',independent_test_claim=False,profile=cfg,data=identity,
                      original_test_in_teacher_training=True,original_test_in_student_distillation=True,student_hard_CE_on_original_test=False,
                      original_test_labels_used_for_hard_example_sampling=True,teacher_checkpoint_sha256=sha256(a.teacher/'best_checkpoint.pt'),
                      prior_optical_executed_epochs=prior['total_executed_epoch_budget'],teacher_head_epochs=teacher_result['completed_epochs'],
                      parent_checkpoint_sha256=sha256(a.development_selection/'best_checkpoint.pt'),source_commit=os.environ.get('T17_SOURCE_COMMIT'),
                      gpu_uuid=gpu,pid=os.getpid(),torch=torch.__version__,command=os.sys.argv,parameter_count=479364,teacher_absent_from_student_inference=True,
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('model_four.py'),
                          Path(__file__).with_name('train_optical_four_ccd.py'),Path(__file__).with_name('test_exposed_data.py'),
                          Path(__file__).with_name('optical_training_methods.py'),Path(__file__).with_name('development_teacher_data.py')]})
        write(a.out/'metadata.json',metadata)
        def key(m):return m['accuracy'],m['balanced_accuracy'],-m['cross_entropy']
        best=key(parent_dev);best_epoch=0;best_kind='parent';history=[];last_improvement=0;dev_evaluations=1;started=time.time()
        def save_best(chosen,epoch,kind,val,dev):
            torch.save(dict(model={k:v.detach().cpu().clone() for k,v in chosen.state_dict().items()},epoch=epoch,weights_kind=kind,
                            validation_diagnostic=val,original_test_dev=dev,metadata=metadata),a.out/'best_checkpoint.pt')
        save_best(model,0,'parent',parent_val,parent_dev)
        optimizer=torch.optim.Adam([{'params':[model.first_phase,model.global_phase],'lr':cfg['kd_phase_lr']},{'params':[model.router_phase],'lr':cfg['kd_router_lr']}],weight_decay=0)
        class_weights=inverse_sqrt_class_weights(y['train'],cfg['num_classes']);rng=np.random.default_rng(cfg['seed']);completed=0
        for epoch in range(1,cfg['kd_epochs']+1):
            begin=time.time();model.train();factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['kd_epochs']-1)))
            for group,lr in zip(optimizer.param_groups,(cfg['kd_phase_lr'],cfg['kd_router_lr'])):group['lr']=lr*factor
            indices=rng.choice(membership,size=len(membership),replace=True,p=sampling_weights);total=0.;batches=0;grad_sum=torch.zeros(6,device='cuda')
            aux_weight=cfg['balance_loss_weight']*max(0.,1-(epoch-1)/cfg['balance_anneal_epochs']);coverage_weight=cfg['coverage_loss_weight']*max(0.,1-(epoch-1)/cfg['balance_anneal_epochs'])
            for start in range(0,len(indices),cfg['batch']):
                idx=torch.as_tensor(indices[start:start+cfg['batch']],device='cuda');codes=torch.randint(8,(len(idx),),device='cuda');images=d4_amplitude(union[idx],codes)
                target=F.softmax(teacher_logits[idx,codes]/cfg['kd_temperature'],1);is_train=idx<n_train
                def objective(out):
                    ce=supervised_train_only(out['logits'],union_labels[idx],is_train,class_weights,cfg['label_smoothing'])
                    kd=F.kl_div(F.log_softmax(out['logits']/cfg['kd_temperature'],1),target,reduction='batchmean')*cfg['kd_temperature']**2
                    return (1-cfg['kd_weight'])*ce+cfg['kd_weight']*kd
                optimizer.zero_grad(set_to_none=True);out=model(images,return_dense_route=True);loss=objective(out)
                loss=loss+aux_weight*actual_top2_load_loss(out['dense_route_power'])+coverage_weight*dark_expert_coverage_loss(out['dense_route_power'])
                if epoch<=cfg['dense_auxiliary_epochs']:
                    dense=model(images,return_dense_route=True,training_route='dense_warmup',training_uniform_mix=.5*(1-(epoch-1)/cfg['dense_auxiliary_epochs']))
                    loss=loss+cfg['dense_auxiliary_weight']*objective(dense)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite KD loss')
                loss.backward()
                if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters()):raise RuntimeError('phase gradient failure')
                grad_sum+=torch.cat((model.first_phase.grad.abs().sum((-2,-1)),model.global_phase.grad.abs().sum().reshape(1),model.router_phase.grad.abs().sum().reshape(1)));batches+=1
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step();update_phase_ema(ema,model,cfg['ema_decay']);total+=float(loss.detach())*len(idx)
            row=dict(epoch=epoch,training_objective=total/len(indices),mean_gradient_L1_experts_global_router=(grad_sum/batches).cpu().tolist())
            if epoch==1 or epoch%cfg['kd_evaluate_every']==0 or epoch==cfg['kd_epochs']:
                choices=[]
                for kind,chosen in [('online',model),('ema',ema)]:
                    val,*_=evaluate(chosen,x['val'],y['val'],cfg['batch']);dev,*_=evaluate(chosen,x['test'],y['test'],cfg['batch']);dev_evaluations+=1
                    row[kind]=dict(validation_diagnostic=val,original_test_dev=dev)
                    guard=val['accuracy']>=parent_val['accuracy']-cfg['val_accuracy_guard_drop'] and val['balanced_accuracy']>=parent_val['balanced_accuracy']-cfg['val_macro_guard_drop']
                    if guard:choices.append((kind,chosen,val,dev))
                if choices:
                    kind,chosen,val,dev=max(choices,key=lambda z:key(z[3]))
                    if key(dev)>best:best=key(dev);best_epoch=epoch;best_kind=kind;last_improvement=epoch;save_best(chosen,epoch,kind,val,dev)
                row['best_epoch']=best_epoch;row['best_kind']=best_kind
            history.append(row);write(a.out/'history.json',history)
            torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),epoch=epoch,optimizer=optimizer.state_dict(),metadata=metadata),a.out/'last_checkpoint.pt')
            completed=epoch;row['seconds']=time.time()-begin;write(a.out/'status.json',dict(state='training',epoch=epoch,best_epoch=best_epoch,best_original_test_dev_accuracy=best[0]));print(json.dumps(row),flush=True)
            if epoch>=cfg['kd_minimum_epochs'] and epoch-last_improvement>=cfg['kd_patience']:break
        chosen=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model']);result={}
        for split in ('train','val','test'):
            metrics,p,q,dense=evaluate(model,x[split],y[split],cfg['batch']);result[split if split!='test' else 'original_test_dev']=metrics
            np.savez_compressed(a.out/f'{split}_predictions.npz',ids=ids[split],labels=y[split].cpu().numpy(),scores=p,predictions=p.argmax(1),route_power=q,dense_route_power=dense)
        result.update(role=metadata['role'],independent_test_claim=False,selected_epoch=best_epoch,selected_kind=best_kind,completed_epochs=completed,
                      prior_optical_executed_epochs=prior['total_executed_epoch_budget'],total_optical_executed_epochs=prior['total_executed_epoch_budget']+completed,
                      teacher_head_epochs=teacher_result['completed_epochs'],
                      original_test_selection_evaluations=dev_evaluations,checkpoint_sha256=sha256(a.out/'best_checkpoint.pt'),elapsed_seconds=time.time()-started,
                      gain_over_frozen_D2NN_pp=100*(result['original_test_dev']['accuracy']-selection['frozen_validation_selected_D2NN']['original_test_dev']['accuracy']),
                      gain_over_test_selected_D2NN_pp=100*(result['original_test_dev']['accuracy']-selection['selected_D2NN_development_reference']['original_test_dev']['accuracy']))
        write(a.out/'result.json',result);write(a.out/'status.json',dict(state='complete',**result));print(json.dumps(result),flush=True)
    except BaseException as error:
        write(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(error)));raise
    finally:torch.cuda.empty_cache();write(a.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':main()
