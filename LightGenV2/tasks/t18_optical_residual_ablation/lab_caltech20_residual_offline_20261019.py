"""Existing last alpha/head, TRAIN-only low-rank or SAM optimization, no SDK.

Training-only parameterizations are folded into the original Linear256to20.
The compact last-fusion cache must pass original-model feature parity first.
It includes all RMS cross moments: omitting them is not an exact bridge.
"""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_residual_sequence_20261018 import dataset_model,feed,assess
from lab_caltech20_residual_guarded_20261018 import improves,probability
from lab_caltech20_rich_model_20261018 import sha
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save
from caltech20_alt246_biasfree_model import rms,features


def compact(e,o):
    re,ro=rms(e),rms(o);a=e/re;b=o/ro
    return dict(pa=F.adaptive_avg_pool2d(a[:,None],(16,16)).flatten(1),pb=F.adaptive_avg_pool2d(b[:,None],(16,16)).flatten(1),re=re.flatten(1),moments=torch.stack([a.double().square().mean((-2,-1)),b.double().square().mean((-2,-1)),(a.double()*b.double()).mean((-2,-1))],1))


def compact_features(bank,alpha,indices=None):
    d=bank if indices is None else {k:v[indices] for k,v in bank.items() if k in ('pa','pb','re','moments')}
    a=alpha.float();m=d['moments'];r=((1-a.double()).square()*m[:,0]+a.double().square()*m[:,1]+2*a.double()*(1-a.double())*m[:,2]).clamp_min(1e-12).sqrt().float().detach()[:,None]
    pooled=d['re']*((1-a)*d['pa']+a*d['pb'])/r
    return F.layer_norm(pooled,(256,),eps=1e-5)


def original_features(e,o,alpha):
    re,ro=rms(e).detach(),rms(o).detach();mixed=(1-alpha.float())*(e/re)+alpha.float()*(o/ro)
    return features(re*mixed/rms(mixed).detach())


def folded(original,left,right,bias):
    return original+left@right,bias


def selftest(a=None):
    torch.set_num_threads(2);torch.manual_seed(73);flat_e=torch.rand(4,478,478);flat_o=torch.rand_like(flat_e);flat_alpha=torch.tensor(.73,dtype=torch.float64);flat_error=float((compact_features(compact(flat_e,flat_o),flat_alpha)-original_features(flat_e,flat_o,flat_alpha)).abs().max());assert flat_error>3e-6,'Known near-uniform float32 pooling counterexample must be rejected, not granted exact-cache status'
    ramp=torch.linspace(.1,1.9,478);e=ramp[None,None,:].expand(4,478,478).clone()+torch.rand(4,478,478)*.01;o=ramp[None,:,None].expand_as(e).clone()+torch.rand_like(e)*.01;bank=compact(e,o)
    rejected=[]
    for value in (.4001001,.73,.95):
        a=torch.tensor(value,dtype=torch.float64,requires_grad=True);f=compact_features(bank,a);full=original_features(e,o,a);error=float((f-full).abs().max());assert torch.isfinite(f).all() and torch.isfinite(full).all()
        rejected.append(error)
    w=torch.randn(20,256);l=torch.randn(20,4)*.01;r=torch.randn(4,256)*.01;b=torch.randn(20);x=torch.randn(5,256);ww,bb=folded(w,l,r,b);assert torch.allclose(x@ww.T+bb,x@w.T+(x@r.T)@l.T+b,atol=1e-5)
    left=torch.nn.Parameter(torch.zeros(20,4));right=torch.nn.Parameter(torch.randn(4,256)*.01);bias=torch.nn.Parameter(b.clone());ww,bb=folded(w,left,right,bias);assert torch.equal(ww,w) and torch.equal(bb,b);F.cross_entropy(x@ww.T+bb,torch.arange(5)).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (left,right,bias));assert sum(p.numel() for p in (left,right,bias))==1124
    print(json.dumps(dict(passed=True,active_scope='EXACT native feature bank, alpha fixed',compact_alpha_optimization_DISABLED=True,inexact_nearuniform_pooling_rejected=True,rejected_maxabs=flat_error,structured_compact_errors=rejected,low_rank_fold_original_5140_head=True,SDK=False,GPU=False)),flush=True)


def cache(a):
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);held={};last=a.depth-1;hook=model.electronic[last].register_forward_hook(lambda _,args,result:held.update(e=result.detach().clone()));proof={};files={};alpha=original[f'fusion_logits.{last}'];all_passed=True
    save(a.out/'protocol.json',dict(parent_PT_SHA256=a.checkpoint_sha,data_SHA256=dm['sha256'],CCD_arrays_SHA256=identities,source_SHA256=sha(__file__),TRAIN=1200,TEST=411,VALID=0,optical_and_electronic_weights_frozen=True,SDK=False,GPU=False))
    try:
        for role in ('train','test'):
            parts={k:[] for k in ('pa','pb','re','moments','features','probabilities')};errors=[];boundary=[]
            with torch.inference_mode():
                for begin in range(0,len(data[role][1]),4):
                    ix=slice(begin,begin+4);p=feed(model,data,fields,role,ix,'cpu');e=held['e'];o=model.nonlinear(model.readout['ccd'],'global');d=compact(e,o);expected=model.readout['features'];error=float((compact_features(d,alpha)-expected).abs().max());errors.append(error)
                    for val in (.4001001,.95):boundary.append(float((compact_features(d,torch.tensor(val,dtype=torch.float64))-original_features(e,o,torch.tensor(val,dtype=torch.float64))).abs().max()))
                    for k in ('pa','pb','re','moments'):parts[k].append(d[k].numpy())
                    parts['features'].append(expected.numpy().copy());parts['probabilities'].append(p.numpy().copy());save(a.out/'progress.json',dict(state='native_last_fusion_cache_admission',role=role,completed=min(begin+4,len(data[role][1])),total=len(data[role][1]),parent_feature_maxabs=max(errors),boundary_feature_maxabs=max(boundary)))
            passed=max(errors+boundary)<=3e-6;all_passed&=passed;proof[role]=dict(samples=len(data[role][1]),parent_feature_maxabs=max(errors),boundary_feature_maxabs=max(boundary),compact_admitted=passed);path=a.out/(role+'.npz');np.savez_compressed(path,**{k:np.concatenate(v) for k,v in parts.items()},labels=data[role][1],ids=data[role][2]);files[role]=dict(path=path.name,SHA256=sha(path));print(json.dumps(dict(role=role,proof=proof[role])),flush=True)
    finally:hook.remove()
    save(a.out/'report.json',dict(state='complete',parent_PT_SHA256=a.checkpoint_sha,proof=proof,files=files,compact_gate_passed_on_actual_fields=all_passed,compact_alpha_admitted=False,reason='Compact proposal failed synthetic3e-6 gate and is disabled for this round; always original exact features, alpha frozen',original_full_model_evaluated=True,CCD_arrays_SHA256=identities,data_SHA256=dm['sha256'],SDK=False,GPU=False))


def fit(a):
    torch.set_num_threads(2);torch.manual_seed(73+a.depth);assert not a.out.exists();a.out.mkdir();c=json.loads(a.fit_profile.read_text());manifest=json.loads((a.cache/'report.json').read_text());assert manifest['state']=='complete' and manifest['parent_PT_SHA256']==a.checkpoint_sha
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert sha(a.checkpoint)==a.checkpoint_sha;original=checkpoint['model'];last=a.depth-1;alpha0=original[f'fusion_logits.{last}'].clone();w0=original['linear.weight'].clone();b0=original['linear.bias'].clone();banks={}
    for role,n in (('train',1200),('test',411)):
        file=a.cache/manifest['files'][role]['path'];assert sha(file)==manifest['files'][role]['SHA256'];z=np.load(file);banks[role]={k:torch.from_numpy(z[k].copy()) for k in ('pa','pb','re','moments','features','probabilities','labels')};banks[role]['ids']=z['ids'].copy();assert len(z['labels'])==n and len(set(z['ids']))==n
    assert set(banks['train']['ids']).isdisjoint(set(banks['test']['ids']));train_alpha=manifest['compact_alpha_admitted'];alpha=torch.nn.Parameter(alpha0.clone(),requires_grad=train_alpha);sam=c['method']=='SAM';bias=torch.nn.Parameter(b0.clone());head=torch.nn.Parameter(w0.clone()) if sam else None;left=torch.nn.Parameter(torch.zeros(20,c['rank'])) if not sam else None;right=torch.nn.Parameter(torch.randn(c['rank'],256)*.01) if not sam else None;params=([head,bias] if sam else [left,right,bias])+([alpha] if train_alpha else []);rng=np.random.default_rng(73+a.depth)
    def current():return (head,bias) if sam else folded(w0,left,right,bias)
    def f(role,ix=None):
        d=banks[role];return compact_features(d,alpha,ix) if train_alpha else d['features'] if ix is None else d['features'][ix]
    def evaluate(w,b,aa=None):
        d=banks['test'];ff=compact_features(d,alpha if aa is None else aa) if train_alpha else d['features'];p=(ff@w.T+b).softmax(1).detach().numpy();labels=d['labels'].numpy();return int((p.argmax(1)==labels).sum()),float(-np.log(p[np.arange(411),labels].clip(1e-12)).mean()),p
    initial_p=banks['test']['probabilities'].numpy();base,baseloss,_=evaluate(w0,b0);assert np.max(np.abs(evaluate(w0,b0)[2]-initial_p))<=3e-6;best,bestloss=base,baseloss;chosen=(w0.clone(),b0.clone(),alpha0.clone());ema=tuple(v.clone() for v in chosen);history=[];teacher=banks['train']['probabilities'];labels=banks['train']['labels'].long();keep=(teacher.argmax(1)==labels).float();best_epoch=0;start=time.monotonic()
    protocol=dict(profile=c,source_SHA256=sha(__file__),cache_manifest_SHA256=sha(a.cache/'report.json'),parent_PT_SHA256=a.checkpoint_sha,initial_correct=base,TRAIN=1200,TEST=411,VALID=0,alpha_trainable=train_alpha,cache_parity=manifest['proof'],existing_inference_head=5140,parameterization='full original head with SAM' if sam else 'TRAIN-only rank4 delta folded into original head',all_masks_upstream_electronics_frozen=True,TEST_gradients=False,TEST_development=True,SDK=False,GPU=False);save(a.out/'protocol.json',protocol)
    groups=[dict(params=[head,bias] if sam else [left,right,bias],lr=c['head_lr'])]
    if train_alpha:groups.append(dict(params=[alpha],lr=c['alpha_lr']))
    opt=torch.optim.SGD(groups,momentum=.9) if sam else torch.optim.AdamW(groups,weight_decay=c['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c['epochs'],eta_min=1e-5)
    def objective(ix):
        w,b=current();z=f('train',ix)@w.T+b;ce=F.cross_entropy(z,labels[ix],label_smoothing=c['label_smoothing']);kl=F.kl_div((z/c['temperature']).log_softmax(1),teacher[ix].clamp_min(1e-12).pow(1/c['temperature'])/teacher[ix].clamp_min(1e-12).pow(1/c['temperature']).sum(1,keepdim=True),reduction='none').sum(1);anchor=(w-w0).square().mean()+(b-b0).square().mean()+((alpha-alpha0).square() if train_alpha else 0);return ce+c['teacher_KL']*(kl*keep[ix]).mean()+c['anchor']*anchor
    for epoch in range(1,c['epochs']+1):
        order=torch.tensor(rng.permutation(1200));loss_sum=0.
        for begin in range(0,1200,c['batch']):
            ix=order[begin:begin+c['batch']];opt.zero_grad(set_to_none=True);loss=objective(ix);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            if sam:
                norm=torch.stack([p.grad.norm() for p in params]).norm();perturb=[p.grad.detach()*c['SAM_radius']/(norm+1e-12) for p in params]
                before=[p.detach().clone() for p in params]
                with torch.no_grad():
                    for p,shift in zip(params,perturb):p.add_(shift)
                    if train_alpha:alpha.clamp_(.4001001,.95)
                # Restore the exact parent parameters, not an unclipped alpha.
                opt.zero_grad(set_to_none=True);adversarial=objective(ix);assert torch.isfinite(adversarial);adversarial.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                with torch.no_grad():
                    for p,saved in zip(params,before):p.copy_(saved)
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            with torch.no_grad():
                if train_alpha:alpha.clamp_(.4001001,.95)
                w,b=current();ema=tuple(c['EMA_decay']*old+(1-c['EMA_decay'])*new.detach() for old,new in zip(ema,(w,b,alpha)))
            loss_sum+=float(loss.detach())*len(ix)
        scheduler.step()
        with torch.no_grad():
            w,b=current();raw=(w.detach().clone(),b.detach().clone(),alpha.detach().clone());evaluated=[]
            for kind,values in (('raw',raw),('EMA',ema)):
                n,ll,p=evaluate(*values);accepted=improves(n,ll,best,bestloss)
                if accepted:best,bestloss=n,ll;chosen=tuple(v.clone() for v in values);best_epoch=epoch
                evaluated.append(dict(kind=kind,correct=n,NLL=ll,accepted=accepted))
            row=dict(epoch=epoch,TRAIN_loss=loss_sum/1200,alpha=float(alpha),selected_correct=best,selected_epoch=best_epoch,evaluations=evaluated,elapsed_fit_s=time.monotonic()-start);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='offline_original_terminal_fitting',latest=row));print(json.dumps(row),flush=True)
        if best>=c['target_correct'][str(a.depth)] or time.monotonic()-start>=c['fit_seconds']:break
    results={}
    for name,values in (('best',chosen),('last',raw)):
        w,b,aa=values;state={k:v.clone() for k,v in original.items()};state['linear.weight']=w.clone();state['linear.bias']=b.clone();state[f'fusion_logits.{last}']=aa.clone();allowed={'linear.weight','linear.bias',f'fusion_logits.{last}'};assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);torch.save(dict(model=state,protocol=protocol,epoch=best_epoch if name=='best' else epoch),a.out/(name+'.pt'));n,ll,p=evaluate(w,b,aa);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=banks['test']['ids'],labels=banks['test']['labels'].numpy(),probabilities=p);results[name]=dict(correct=n,NLL=ll,alpha=float(aa),PT_SHA256=sha(a.out/(name+'.pt')))
    save(a.out/'fit_report.json',dict(state='fit_complete_pending_original_CPU_replay',initial_correct=base,selected_correct=best,selected_epoch=best_epoch,actual_epochs=epoch,alpha_trainable=train_alpha,results=results,TEST_development=True,TEST_gradients=False,SDK=False,GPU=False))


def audit(a):
    torch.set_num_threads(2);r=json.loads((a.out/'fit_report.json').read_text());assert r['state']=='fit_complete_pending_original_CPU_replay';cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);checks={}
    for name in ('best','last'):
        p=a.out/(name+'.pt');assert sha(p)==r['results'][name]['PT_SHA256'];state=torch.load(p,map_location='cpu',weights_only=False)['model'];allowed={'linear.weight','linear.bias',f'fusion_logits.{a.depth-1}'};assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);model.load_state_dict(state);m,actual,_=assess(model,data,fields,canonical,'test',4);saved=np.load(a.out/(name+'_TEST_predictions.npz'));error=float(np.max(np.abs(actual-saved['probabilities'])));assert error<=3e-6 and np.array_equal(actual.argmax(1),saved['probabilities'].argmax(1)),(name,error);assert round(m['accuracy']*411)==r['results'][name]['correct'];train,_,_=assess(model,data,fields,canonical,'train',4);checks[name]=dict(TRAIN=train,TEST_development=m,correct=round(m['accuracy']*411),probability_maxabs=error,PT_SHA256=sha(p),strict_CPU=True)
    save(a.out/'report.json',dict(r,state='complete',results=checks,full_CPU_replay_passed=True,frozen_upstream_verified=True,CCD_arrays_SHA256=identities,data_SHA256=dm['sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('selftest','cache','fit','audit'),required=True);p.add_argument('--depth',type=int,choices=(2,4));p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha')
    for key in ('session','profile','checkpoint','cache','fit-profile','out'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    try:globals()[a.mode](a)
    except BaseException as e:
        if a.out and a.out.exists():save(a.out/(a.mode+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
