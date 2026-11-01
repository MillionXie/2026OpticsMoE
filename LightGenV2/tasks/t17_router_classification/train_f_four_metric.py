"""Keep a balanced optical router fixed; recover four TRAIN/TEST metrics jointly.

The teacher uses explicitly authorized original TEST hard labels. The student
receives TEST soft targets only and inherits the TEST-trained teacher. Neither
domain nor split identifiers are inputs to the canonical optical forward.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .train_figure_revision import build_model,configure_parameters,training_forward,evaluate
from .figure_revision_data import load_panel_f
from .model_four import encode_rgb
from .phase_optimizer import WrappedPhaseAdam
from .train_convex_readout import training_coordinates,canonical_weight
from .report_f_load_balance import usage_statistics
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def four_metric_key(train,test):
    values=[part['domains'][name]['accuracy']for part in (train,test)for name in ('RGB','SAR')]
    if len(values)!=4 or not all(np.isfinite(values)):raise ValueError('Four finite original domain metrics required')
    return min(values),sum(values)/4,-(train['cross_entropy']+test['cross_entropy'])


def four_group_objective(logits,labels,groups,teacher_targets=None,temperature=2.,KD_weight=.5,smoothing=.02,robust_temperature=.5):
    if temperature<=0 or robust_temperature<=0 or KD_weight<0:raise ValueError('Positive temperatures and nonnegative KD weight required')
    if logits.ndim!=2 or len(logits)==0 or groups.shape!=logits.shape[:1]or labels.shape!=groups.shape or not bool(((groups>=0)&(groups<=3)).all()):raise ValueError('Explicit four input groups required')
    soft=teacher_targets is not None
    if soft:
        if teacher_targets.shape!=logits.shape:raise ValueError('Matched teacher logits required')
        if bool((labels[groups>=2]!=-1).any()):raise ValueError('Student TEST hard labels forbidden')
        target=teacher_targets.detach().to(logits)
        row_KL=F.kl_div((logits/temperature).log_softmax(1),(target/temperature).softmax(1),reduction='none').sum(1)*temperature**2
    values=[]
    for group in range(4):
        mask=groups==group
        if not bool(mask.any()):continue
        if soft:
            value=KD_weight*row_KL[mask].mean()
            if group<2:value=value+F.cross_entropy(logits[mask],labels[mask],label_smoothing=smoothing)
        else:value=F.cross_entropy(logits[mask],labels[mask],label_smoothing=smoothing)
        values.append(value)
    losses=torch.stack(values)
    # Normalize logsumexp so that equal group losses retain their original scale.
    return robust_temperature*(torch.logsumexp(losses/robust_temperature,0)-np.log(len(values)))


def refit_four_group_head(model,fields,labels,groups,anchor,cfg,teacher=None,steps=40):
    """Numerical coordinates are folded into the same float32 bias-free head."""
    model.eval();before={n:p.detach().clone()for n,p in model.named_parameters()if n!='shared_head.weight'};features=[]
    with torch.no_grad():
        for i in range(0,len(fields),cfg['eval_batch']):features.append(model.optical_features(fields[i:i+cfg['eval_batch']])['ccd_features'])
    features=torch.cat(features).double();anchor64=anchor.double();transformed,start,factor,statistics=training_coordinates(features,model.shared_head.weight.detach().double(),'TRAIN_Gram_cholesky');statistics['original_population']='original TRAIN plus explicitly exposed original TEST';statistics['TRAIN_only']=False;statistics['total_original_images']=statistics.pop('TRAIN_samples');coordinate=torch.nn.Parameter(start)
    opt=torch.optim.LBFGS([coordinate],lr=1.,max_iter=steps,max_eval=2*steps,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12);calls=0
    def objective(logits,weight):return four_group_objective(logits,labels,groups,teacher,robust_temperature=cfg['group_robust_temperature'])+.5*cfg['head_anchor']*(weight-anchor64).square().sum()
    initial=float(objective(F.linear(features,model.shared_head.weight.detach().double()),model.shared_head.weight.detach().double()).detach())
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True);weight=canonical_weight(coordinate,factor);loss=objective(F.linear(transformed,coordinate),weight);loss.backward();calls+=1
        if not torch.isfinite(loss)or not torch.isfinite(coordinate.grad).all():raise RuntimeError('Nonfinite four-group head objective')
        return loss
    opt.step(closure)
    with torch.no_grad():model.shared_head.weight.copy_(canonical_weight(coordinate,factor).float())
    if not all(torch.equal(p,before[n])for n,p in model.named_parameters()if n in before):raise RuntimeError('Head fit changed optical phase')
    final=float(objective(F.linear(features,model.shared_head.weight.detach().double()),model.shared_head.weight.detach().double()).detach())
    return dict(initial_objective=initial,final_objective=final,internal_steps=opt.state[coordinate]['n_iter'],closure_calls=calls,unique_TRAIN_images=int((groups<2).sum()),unique_original_TEST_images=int((groups>=2).sum()),TEST_hard_GT_in_loss=teacher is None,all_phases_bitwise_frozen=True,exported_float32_same_Linear=True,coordinate_statistics=statistics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();cfg=json.loads(args.profile.read_text());task=Path(__file__).resolve().parent
    if cfg['architecture']!='LG'or cfg['main_layers']!=2 or cfg.get('top_k',2)!=2 or cfg['training_scope']!='all_except_router':raise ValueError('Original two-layer Top2 optical LG with frozen router required')
    if cfg['teacher']['epochs']>60 or cfg['student']['epochs']>15:raise ValueError('Finite teacher/student budgets required')
    gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);args.out.mkdir(parents=True,exist_ok=False)
    raw,identity=load_panel_f(cfg['data']);fields={};labels={}
    for split,part in raw.items():
        fields[split]=torch.cat([encode_rgb(torch.from_numpy(part['images'][i:i+128]).cuda())for i in range(0,len(part['labels']),128)]);labels[split]=torch.tensor(part['labels'],device='cuda',dtype=torch.long);del part['images']
    all_fields=torch.cat([fields['train'],fields['test']]);all_labels=torch.cat([labels['train'],labels['test']]);fields['train']=all_fields[:len(labels['train'])];fields['test']=all_fields[len(labels['train']):];torch.cuda.empty_cache();groups=torch.tensor(np.r_[raw['train']['domains'],raw['test']['domains']+2],device='cuda',dtype=torch.long)
    parent=task/cfg['parent_checkpoint']
    if sha256(parent)!=cfg['parent_checkpoint_sha256']:raise ValueError('Parent SHA changed')
    model=build_model(cfg).cuda();model.load_state_dict(torch.load(parent,map_location='cpu',weights_only=False)['model'],strict=True);fixed_router=model.router_phase.detach().clone();teacher_targets=None;teacher_SHA=None
    if len(model.state_dict())!=4 or sum(p.numel()for p in model.parameters())!=487204 or sum(isinstance(v,torch.nn.Linear)for v in model.modules())!=1 or any(isinstance(v,torch.nn.Conv2d)for v in model.modules()):raise ValueError('Canonical inference contract changed')
    try:
        for stage in ('teacher','student'):
            out=args.out/stage;out.mkdir();phases,heads=configure_parameters(model,cfg);budget=cfg[stage];soft=teacher_targets is not None;loss_labels=all_labels.clone()
            if soft:loss_labels[groups>=2]=-1
            metadata=dict(profile=cfg,profile_sha256=sha256(args.profile),data=identity,stage=stage,source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={n:sha256(task/n)for n in [Path(__file__).name,'train_figure_revision.py','model_four.py','model_linear_readout.py','model_depth_ccd.py','phase_optimizer.py','train_convex_readout.py']},pid=os.getpid(),gpu_uuid=gpu,command=sys.argv,role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',TEST_hard_GT_in_loss=not soft,student_inherits_TEST_hard_supervised_teacher_weights=soft,initial_teacher_SHA=teacher_SHA,VAL_gate=False,selection='minimum of RGB/SAR TRAIN and original TEST accuracy, then four-group mean, then CE',router_frozen=True,router_raw_sha256=__import__('hashlib').sha256(fixed_router.cpu().numpy().tobytes()).hexdigest(),mixed_TRAIN_and_TEST_minibatches=True,inference_parameters=487204)
            metadata['physical_source_sha256']={str(p.relative_to(task.parents[1])):sha256(p)for p in [task.parents[1]/'demo_check/pure_optical/models.py',task.parents[1]/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']}
            write(out/'metadata.json',metadata);write(out/'status.json',dict(state='loading',pid=os.getpid()))
            phase_opt=WrappedPhaseAdam(phases,cfg['phase_lr']if not soft else cfg['student_phase_lr']);head_opt=torch.optim.AdamW(heads,lr=cfg['head_lr'],weight_decay=0);anchor=model.shared_head.weight.detach().clone();history=[];selected=0;selected_stage='initial_parent';stale=0
            def assess():
                train,_=evaluate(model,fields['train'],labels['train'],cfg['eval_batch'],raw['train']['domains']);test,arr=evaluate(model,fields['test'],labels['test'],cfg['eval_batch'],raw['test']['domains']);usage={name:usage_statistics(arr['route_power'][arr['domains']==d],arr['dense_route_power'][arr['domains']==d])for d,name in [(0,'RGB'),(1,'SAR')]};return train,test,usage
            def save(name,epoch):torch.save(dict(model=model.state_dict(),epoch=epoch,metadata=metadata,optimizer=phase_opt.state_dict(),head_optimizer=head_opt.state_dict(),rng_state=rng.bit_generator.state,selection_stage=selected_stage),out/name)
            train,test,usage=assess();best=four_metric_key(train,test);metadata['initial_four_metrics']=dict(train=train,original_test_dev=test);save('best_checkpoint.pt',0)
            if stage=='teacher':
                fit=refit_four_group_head(model,all_fields,loss_labels,groups,anchor,cfg,steps=cfg['initial_head_inner_steps']);train,test,usage=assess();metadata['initial_group_head_fit']=fit
                if four_metric_key(train,test)>best:best=four_metric_key(train,test);selected_stage='initial_group_head_fit';save('best_checkpoint.pt',0)
            write(out/'metadata.json',metadata);initial_experts=model.first_phase.detach().clone()
            for epoch in range(1,budget['epochs']+1):
                started=time.time();model.train();ratio=.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/max(1,budget['epochs']-1)));phase_lr=cfg['phase_lr']if not soft else cfg['student_phase_lr']
                for opt,lr in [(phase_opt,phase_lr),(head_opt,cfg['head_lr'])]:
                    for group in opt.param_groups:group['lr']=lr*ratio
                order=rng.permutation(len(all_fields));mean_loss=0.;norms=np.zeros(4);steps=0
                for i in range(0,len(order),cfg['batch']):
                    idx=torch.tensor(order[i:i+cfg['batch']],device='cuda');phase_opt.zero_grad(set_to_none=True);head_opt.zero_grad(set_to_none=True);output=training_forward(model,all_fields[idx],cfg,True)
                    loss=four_group_objective(output['raw_logits'],loss_labels[idx],groups[idx],teacher_targets[idx]if soft else None,robust_temperature=cfg['group_robust_temperature'])+.5*cfg['head_anchor']*(model.shared_head.weight-anchor).square().sum();loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all()for p in phases+heads):raise RuntimeError('Invalid four-group phase/head gradient')
                    norms+=model.first_phase.grad.detach().flatten(1).norm(dim=1).cpu().numpy();torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True);phase_opt.step();head_opt.step();steps+=1;mean_loss+=float(loss.detach())*len(idx)
                fit=None
                if epoch%cfg['head_interval']==0:fit=refit_four_group_head(model,all_fields,loss_labels,groups,anchor,cfg,teacher_targets if soft else None,steps=cfg['head_inner_steps']);head_opt.state.clear()
                if not torch.equal(model.router_phase,fixed_router):raise RuntimeError('Balanced router drift')
                train,test,usage=assess();key=four_metric_key(train,test)
                if key>best:best=key;selected=epoch;selected_stage='joint_training';stale=0;save('best_checkpoint.pt',epoch)
                else:stale+=1
                row=dict(epoch=epoch,train=train,original_test_development_selection=test,selected_epoch=selected,selected_stage=selected_stage,minimum_four_accuracy=key[0],development_expert_usage=usage,unique_TRAIN_images=len(labels['train']),unique_original_TEST_images=len(labels['test']),mixed_original_presentations=len(order),phase_head_updates=steps,mean_robust_objective=mean_loss/len(order),router_frozen_bitwise=True,classification_raw_expert_gradient_L2_sum=norms.tolist(),expert_phase_raw_change_L2=(model.first_phase.detach()-initial_experts).flatten(1).norm(dim=1).cpu().tolist(),seconds=time.time()-started)
                if fit:row['four_group_head_fit']=fit
                history.append(row);write(out/'history.json',history);save('last_checkpoint.pt',epoch);write(out/'status.json',dict(state='training',epoch=epoch,selected_epoch=selected,minimum_four_accuracy=key[0],pid=os.getpid()));print(json.dumps(dict(stage=stage,epoch=epoch,minimum_four_accuracy=key[0],selected_epoch=selected)),flush=True)
                if epoch>=budget['min_epochs']and stale>=budget['patience']:break
            model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'],strict=True);result=dict(role=metadata['role'],stage=stage,completed_epochs=epoch,selected_epoch=selected,selected_stage=selected_stage,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),router_frozen=True,inference_parameters=487204)
            for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:result[key],arr=evaluate(model,fields[split],labels[split],cfg['eval_batch'],raw[split]['domains']);np.savez_compressed(out/f'{split}_predictions.npz',ids=raw[split]['ids'],**arr)
            result['minimum_four_accuracy']=four_metric_key(result['train'],result['original_test_dev'])[0];write(out/'result.json',result);write(out/'selection_lock.json',dict(policy=metadata['selection'],checkpoint_sha256=result['checkpoint_sha256'],TEST_evaluations_before_lock=epoch+1+(stage=='teacher'),TRAIN_evaluated_for_selection=True,VAL_gate=False));write(out/'status.json',dict(state='complete',pid=os.getpid(),**result))
            if stage=='teacher':
                model.eval();targets=[]
                with torch.no_grad():
                    for i in range(0,len(all_fields),cfg['eval_batch']):targets.append(model(all_fields[i:i+cfg['eval_batch']])['raw_logits'])
                teacher_targets=torch.cat(targets).detach();teacher_SHA=result['checkpoint_sha256'];path=out/'canonical_teacher_targets.npz';np.savez_compressed(path,logits=teacher_targets.cpu().numpy(),ids=np.r_[raw['train']['ids'],raw['test']['ids']],labels=all_labels.cpu().numpy(),groups=groups.cpu().numpy());write(out/'teacher_target_manifest.json',dict(teacher_checkpoint_sha256=teacher_SHA,logits_sha256=sha256(path),shape=list(teacher_targets.shape),view_codes=[0],teacher_architecture_identical=True,TEST_hard_GT_used_by_teacher=True))
        write(args.out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),pid=os.getpid()));raise
    finally:torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
