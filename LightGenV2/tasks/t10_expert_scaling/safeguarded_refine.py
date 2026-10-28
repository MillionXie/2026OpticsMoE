"""Finite deterministic full-objective L-BFGS with bounded Armijo trials.

Both clean-view objective-pool choice and optimizer differ from prior stochastic
D4 Adam stages. This is adaptive TEST-exposed fitting, not a mechanism-isolated
or equal-budget comparison. Existing source checkpoints remain immutable.
"""
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

from .expert_tail_schedule import save,sha
from .checkpoint_test_selection import model_hash


def direction(gradient, history, scales, radius=.25):
    """Two-loop inverse-Hessian action; supports NumPy and Torch vectors."""
    q=gradient*1; alphas=[]
    for s,y,rho in reversed(history):
        alpha=rho*float((s*q).sum());alphas.append(alpha);q=q-alpha*y
    factor=1.
    if history:
        s,y,_=history[-1];factor=max(1e-6,min(1e6,float((s*y).sum())/max(float((y*y).sum()),1e-30)))
    r=q*scales*factor
    for (s,y,rho),alpha in zip(history,reversed(alphas)):
        r=r+s*(alpha-rho*float((y*r).sum()))
    p=-r;fallback=False
    if not math.isfinite(float((gradient*p).sum())) or float((gradient*p).sum())>=0:
        p=-gradient*scales;fallback=True
    norm=math.sqrt(float((p*p).sum()))
    if norm>radius:p=p*(radius/norm)
    return p,dict(initial_inverse_scale=factor,fallback_to_scaled_gradient=fallback,
        direction_norm=math.sqrt(float((p*p).sum())),directional_derivative=float((gradient*p).sum()))


def armijo_accept(old,new,slope,alpha):
    return all(math.isfinite(v) for v in [old,new,slope,alpha]) and slope<0 and alpha>0 and new<=old-max(1e-7,-1e-4*alpha*slope)


def pool_indices(labels,seed):
    import numpy as np
    y=np.asarray(labels);assert y.shape==(4248,)
    rng=np.random.default_rng(seed+2400810019)
    train=np.concatenate([rng.permutation(np.flatnonzero((y==k)&(np.arange(4248)<3496)))[:94] for k in range(8)])
    indices=np.concatenate([train,np.arange(3496,4248)]).astype(np.int64);rng.shuffle(indices)
    assert len(indices)==len(set(indices))==1504 and sum(indices>=3496)==752
    return indices


def attach_test_split(splits):
    # train.data deliberately defers TEST. Auxiliary.configure has already
    # explicitly assembled the authorized TRAIN+TEST pool in this stage.
    x,y,ids=splits['train']
    assert len(y)==len(ids)==4248 and len(splits['original_train'][1])==3496
    splits['test']=(x[3496:],y[3496:],ids[3496:])
    assert len(splits['test'][1])==752


def validate(c):
    assert c['model_experts'] in [25,49] and c['seed'] in [27,37]
    assert c['expected_parameter_count']=={25:5634176,49:11078432}[c['model_experts']]
    assert c['max_steps']==8 and c['max_objective_passes']==24 and c['max_line_trials']==4
    assert c['check_steps']==[2,4,8] and c['history_size']==3 and c['direction_radius']==.25
    assert c['fixed_lambda_kd']==.3203496288685833 and c['temperature']==2 and c['parent_weight']==.5
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['gpu_index'] in [1,3] and c['minimum_gpu_free_mib']>=6000
    assert c['gpu_uuid']=={1:'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',3:'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}[c['gpu_index']]


def run(c,out,profile):
    import numpy as np
    import torch
    from torch.nn.utils import parameters_to_vector,vector_to_parameters
    from . import reheat_refine as base
    from .test_exposed_distill import Auxiliary
    from .train import evaluate,objective
    from .diagnose_topk import classification_margin_loss,save_route_audit
    validate(c);commit=os.environ['T10_SAFEGUARDED_SOURCE_COMMIT']
    os.environ['T10_REHEAT_SOURCE_COMMIT']=os.environ['T10_DISTILL_SOURCE_COMMIT']=commit
    assert os.environ['CUDA_VISIBLE_DEVICES']==c['gpu_uuid']
    gpu=subprocess.check_output(['nvidia-smi','-i',str(c['gpu_index']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert gpu[0].strip()==c['gpu_uuid'] and int(gpu[1])>=6000
    commit,hashes=base.sources(profile,c)
    assert sha(Path(c['parent_cache'])/'receipt.json')==c['locked_parent_receipt_sha256']
    assert sha(Path(c['parent_cache'])/'parent_scores.npz')==c['locked_parent_scores_sha256']
    assert sha(Path(c['target_cache'])/'teacher_manifest.json')==c['locked_teacher_manifest_sha256']
    aux=Auxiliary(c,False);aux.weight=c['fixed_lambda_kd'];aux.calibration_view0=True;aux.start_round(0)
    model,metadata,protocol,splits,_,_=base.prepare(c,'student');aux.configure(splits);attach_test_split(splits)
    source_hash=model_hash(model.state_dict());assert source_hash==c['source_tensor_sha256']
    x,y,ids=splits['train'];indices=pool_indices(y.numpy(),c['seed'])
    assert set(ids[indices]).isdisjoint(splits['val'][2])
    tc=protocol['training_proposal'];assert tc['phase_noise_std']==0
    assert {k:tc[k] for k in ['label_smoothing','capture_loss_weight','phase_tv_weight','router_balance_weight']}==dict(label_smoothing=.02,capture_loss_weight=.005,phase_tv_weight=.02,router_balance_weight=.01)
    params=list(model.parameters());shapes=[p.shape for p in params]
    scales=torch.cat([torch.full_like(p,.1 if n=='router_phase' else .5 if n.startswith('global') else 1.).reshape(-1) for n,p in model.named_parameters()])
    weights=torch.ones(8,device='cuda');started=time.time();last_progress=0.
    counts=dict(objective_passes=0,gradient_passes=0,objective_model_inputs=0,gradient_model_inputs=0,
        TEST_gradient_inputs=0,original_TRAIN_gradient_inputs=0,check_TEST_inputs=0,check_VAL_inputs=0)
    out.mkdir(parents=True,exist_ok=False)
    metadata=copy.deepcopy(metadata);metadata.update(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,
        source_checkpoint_sha256=c['incumbent_sha256'],source_model_tensor_sha256=source_hash,
        profile_sha256=sha(profile),configuration=c,auxiliary_identity=aux.identity,protocol=protocol,
        host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_method='Bounded Armijo L-BFGS; no Adam or EMA',
        metric_role='TEST-exposed fitting; clean fixed1504 pool and optimizer change together; adaptive seed budgets; no equal-budget/independent-generalization claim.')
    metadata['arguments'].update(out=str(out),epochs=8,optimizer='safeguarded_LBFGS')
    save(out/'metadata.json',metadata)
    save(out/'pool.json',dict(indices=indices.tolist(),ids=ids[indices].tolist(),labels=y[indices].tolist(),
        origins=np.where(indices<3496,'train','test').tolist(),view_codes=[0]*len(indices),
        selection='All752 TEST +94 uniform originalTRAIN per class; class/origin RNG independent of predictions; fixed across all line trials.'))
    source=Path(c['incumbent_run']);prior=json.loads((source/'result.json').read_text())
    assert prior['test_correct']==c['baseline_test_correct']
    assert sha(source/'selected_test_predictions.npz')==c['incumbent_predictions_sha256']
    shutil.copy2(source/'best_checkpoint.pt',out/'best_checkpoint.pt');shutil.copy2(source/'selected_test_predictions.npz',out/'selected_test_predictions.npz')
    best=dict(origin='incumbent',step=0,test_correct=prior['test_correct'],test=prior['test'],model_tensor_sha256=source_hash,
        checkpoint_sha256=c['incumbent_sha256'],state_field='model',predictions_sha256=c['incumbent_predictions_sha256'])
    save(out/'best_development_result.json',best);best_key=(best['test_correct'],-best['test']['macro_nll'])
    history=[];memory=[];bad=0;reason='fixed_step_budget';checked=set();elapsed_offset=0.
    resume_objective=None;resume_gradient_norm=None
    if c.get('resume_run'):
        old=Path(c['resume_run']);resume_path=old/'last_checkpoint.pt'
        assert sha(resume_path)==c['resume_checkpoint_sha256'] and sha(old/'pool.json')==c['resume_pool_sha256']
        assert json.loads((old/'status.json').read_text())['error']=="KeyError('test')"
        assert json.loads((old/'best_development_result.json').read_text())['origin']=='incumbent'
        value=torch.load(resume_path,map_location='cpu',weights_only=False)
        assert value['metadata']['source_commit']==c['resume_previous_source_commit']
        assert value['round']==value['epoch']==c['resume_accepted_steps']==2
        assert model_hash(value['model'])==c['resume_model_tensor_sha256']
        for key in ['seed','model_experts','data_sha256','max_steps','max_objective_passes','max_line_trials','history_size','direction_radius','fixed_lambda_kd','parent_checkpoint_sha256']:
            assert value['metadata']['configuration'][key]==c[key],key
        previous_pool=json.loads((old/'pool.json').read_text())
        assert previous_pool==json.loads((out/'pool.json').read_text())
        history=json.loads((old/'history.json').read_text())
        assert [r['step'] for r in history]==[1,2] and history[-1]['model_tensor_sha256']==c['resume_model_tensor_sha256']
        model.load_state_dict(value['model']);counts=copy.deepcopy(value['cost'])
        assert counts==history[-1]['cost'] and counts['objective_passes']==5 and counts['gradient_passes']==3
        memory=[(s.cuda(),v.cuda(),rho) for s,v,rho in value['lbfgs_history']]
        assert len(memory)<=3 and all(s.numel()==v.numel()==c['expected_parameter_count'] and torch.isfinite(s).all() and torch.isfinite(v).all() and rho>0 for s,v,rho in memory)
        assert all(abs(float((s*v).sum())*rho-1)<1e-4 for s,v,rho in memory)
        resume_objective=value['objective'];resume_gradient_norm=history[-1]['gradient_norm'];elapsed_offset=history[-1]['elapsed_seconds']
        metadata['resume']=dict(previous_run=str(old),previous_source_commit=value['metadata']['source_commit'],
            checkpoint_sha256=c['resume_checkpoint_sha256'],model_tensor_sha256=c['resume_model_tensor_sha256'],
            accepted_steps_restored=2,LBFGS_history_pairs_restored=len(memory),previous_cost=copy.deepcopy(counts),
            gradient_recomputed_once_and_charged_to_original_budget=True,previous_elapsed_seconds=elapsed_offset)
        save(out/'metadata.json',metadata);save(out/'history.json',history)
        del value

    def objective_pass(backward,phase):
        nonlocal last_progress
        if counts['objective_passes']>=c['max_objective_passes']:raise RuntimeError('Objective pass budget exhausted')
        model.train();model.zero_grad(set_to_none=True);aux.start_batch(torch.from_numpy(indices),0)
        total=0.;counts['objective_passes']+=1
        if backward:counts['gradient_passes']+=1
        with torch.set_grad_enabled(backward):
            for begin in range(0,len(indices),2):
                rows=torch.from_numpy(indices[begin:begin+2]);labels=y[rows].cuda()
                output=model(aux.images(x,rows,0));p=output['probabilities']
                loss=(objective(model,output,labels,weights,tc)+.2*classification_margin_loss(p,labels))*len(rows)/len(indices)
                loss=loss+aux.loss(p,rows,0)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite full objective')
                if backward:loss.backward()
                total+=float(loss.detach());counts['objective_model_inputs']+=len(rows)
                if backward:
                    counts['gradient_model_inputs']+=len(rows);counts['TEST_gradient_inputs']+=int((rows>=3496).sum());counts['original_TRAIN_gradient_inputs']+=int((rows<3496).sum())
                if time.time()-last_progress>30:
                    save(out/'status.json',dict(state=phase,pid=os.getpid(),completed=begin+len(rows),total=len(indices),timestamp=time.time(),cost=counts));last_progress=time.time()
        if backward:
            grad=torch.cat([p.grad.detach().reshape(-1) for p in params]);assert torch.isfinite(grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0 for p in params)
            return total,grad
        return total,None

    def checkpoint(step,loss):
        torch.save(dict(model={n:p.detach().cpu() for n,p in model.state_dict().items()},epoch=step,round=step,
            metadata=metadata,objective=loss,current_gradient=gradient.detach().cpu(),
            lbfgs_history=[(s.cpu(),v.cpu(),rho) for s,v,rho in memory],cost=counts),out/'last_checkpoint.pt')

    def check(step):
        nonlocal best,best_key,bad
        assert step not in checked;checked.add(step)
        lock=dict(step=step,checkpoint_sha256=sha(out/'last_checkpoint.pt'),state_field='model',model_tensor_sha256=model_hash(model.state_dict()))
        with (out/'development_locks.jsonl').open('a') as f:f.write(json.dumps(lock)+'\n')
        test,scores=evaluate(model,splits['test'],4);val,_=evaluate(model,splits['val'],4)
        counts['check_TEST_inputs']+=752;counts['check_VAL_inputs']+=752
        assert model_hash(model.state_dict())==lock['model_tensor_sha256']
        name=f'test_step{step}.npz';np.savez_compressed(out/name,ids=splits['test'][2],labels=splits['test'][1].numpy(),scores=scores.numpy(),predictions=scores.argmax(1).numpy())
        correct=int(np.trace(test['confusion_matrix']));key=(correct,-test['macro_nll'])
        decision=dict(**lock,test_correct=correct,test=test,val=val,predictions=name,predictions_sha256=sha(out/name),selected=key>best_key,origin='refinement')
        if key>best_key:
            best_key=key;best=decision;shutil.copy2(out/'last_checkpoint.pt',out/'best_checkpoint.pt');shutil.copy2(out/name,out/'selected_test_predictions.npz');save(out/'best_development_result.json',best)
        with (out/'development_decisions.jsonl').open('a') as f:f.write(json.dumps(decision)+'\n')
        bad=bad+1 if prior['val']['accuracy']-val['accuracy']>.05 else 0
        return decision

    loss,gradient=objective_pass(True,'resume_full_gradient' if history else 'initial_full_gradient')
    if history:
        assert abs(loss-resume_objective)<1e-5 and abs(float(gradient.norm())-resume_gradient_norm)<1e-5
        metadata['resume'].update(recomputed_objective=loss,recomputed_gradient_norm=float(gradient.norm()),
            objective_absolute_error=abs(loss-resume_objective),gradient_norm_absolute_error=abs(float(gradient.norm())-resume_gradient_norm))
        counts['resume_recomputed_gradient_passes']=1;save(out/'metadata.json',metadata)
    checkpoint(len(history),loss)
    if history:
        history[-1]['development_check']=check(len(history));save(out/'history.json',history)
    for step in range(len(history)+1,c['max_steps']+1):
        if counts['objective_passes']+2>c['max_objective_passes']:reason='objective_pass_budget';break
        old_vector=parameters_to_vector([p.detach() for p in params]).clone();old_gradient=gradient.clone();old_loss=loss
        p,info=direction(gradient,memory,scales,c['direction_radius']);slope=info['directional_derivative']
        trials=[];accepted=False
        for trial in range(c['max_line_trials']):
            if counts['objective_passes']+2>c['max_objective_passes']:break
            alpha=.5**trial
            vector_to_parameters(old_vector+alpha*p,params)
            value,_=objective_pass(False,f'step{step}_line{trial+1}')
            ok=armijo_accept(old_loss,value,slope,alpha);trials.append(dict(alpha=alpha,loss=value,accepted=ok))
            if ok:accepted=True;break
        if not accepted:
            vector_to_parameters(old_vector,params);reason='line_search_no_accepted_descent';save(out/'last_line_failure.json',dict(step=step,old_loss=old_loss,direction=info,trials=trials,source_restored_to_previous_step=True));break
        loss,gradient=objective_pass(True,f'step{step}_accepted_gradient')
        assert abs(loss-value)<1e-5 and armijo_accept(old_loss,loss,slope,alpha)
        s=parameters_to_vector([v.detach() for v in params])-old_vector;delta=gradient-old_gradient;curvature=float((s*delta).sum())
        if curvature>1e-12:
            memory.append((s.detach().clone(),delta.detach().clone(),1/curvature));memory=memory[-c['history_size']:]
        row=dict(step=step,objective_before=old_loss,objective_after=loss,direction=info,trials=trials,
            accepted_alpha=alpha,actual_step_norm=float(s.norm()),curvature=curvature,history_pairs=len(memory),
            gradient_norm=float(gradient.norm()),gradients={n:dict(norm=float(p.grad.norm()),finite=bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters()},
            model_tensor_sha256=model_hash(model.state_dict()),elapsed_seconds=elapsed_offset+time.time()-started,cost=copy.deepcopy(counts))
        history.append(row);checkpoint(step,loss);save(out/'history.json',history)
        if step in c['check_steps']:
            row['development_check']=check(step);save(out/'history.json',history)
            if bad>=2:reason='preset_VAL_degradation';break
        save(out/'status.json',dict(state='optimizing',pid=os.getpid(),accepted_steps=len(history),best_test_correct=best['test_correct'],timestamp=time.time(),cost=counts))
    if history and len(history) not in checked:check(len(history))
    save(out/'history.json',history)
    selected=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);del selected
    assert model_hash(model.state_dict())==best['model_tensor_sha256']
    train,tscores=evaluate(model,splits['original_train'],4);val,_=evaluate(model,splits['val'],4)
    np.savez_compressed(out/'selected_original_train_scores.npz',ids=splits['original_train'][2],labels=splits['original_train'][1].numpy(),scores=tscores.numpy())
    for name in ['original_train','val','test']:save_route_audit(model,splits[name],out/f'selected_{name}_routes.npz')
    union=np.asarray(train['confusion_matrix'])+np.asarray(best['test']['confusion_matrix'])
    assert sha(source/'best_checkpoint.pt')==c['incumbent_sha256']
    result=dict(source_commit=commit,accepted_steps=len(history),selection=best,test=best['test'],test_correct=best['test_correct'],
        original_train=train,val=val,union_correct=int(np.trace(union)),union_confusion_matrix=union.tolist(),
        checkpoint_sha256=sha(out/'best_checkpoint.pt'),last_checkpoint_sha256=sha(out/'last_checkpoint.pt'),
        selected_model_tensor_sha256=best['model_tensor_sha256'],source_best_disk_SHA_unchanged=True,cost=counts,
        final_full_model_inputs=4248,final_router_only_inputs=5000,stop_reason=reason,elapsed_seconds=elapsed_offset+time.time()-started,
        original_equal_budget_target_established=False,test_exposed_teacher=True,test_images_enter_student_gradient=True,
        metric_role='Adaptive TEST-exposed fit; uniform fixed752TRAIN subset and all752TEST, clean view, safeguarded optimizer; no independent/equal-budget claim.')
    save(out/'result.json',result);save(out/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time(),test_correct=best['test_correct'],accepted_steps=len(history),stop_reason=reason))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();c=json.loads(a.profile.read_text());c['_profile']=str(a.profile)
    try:run(c,a.out,a.profile)
    except BaseException as error:
        if a.out.exists():save(a.out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error),timestamp=time.time()))
        raise


if __name__=='__main__':main()
