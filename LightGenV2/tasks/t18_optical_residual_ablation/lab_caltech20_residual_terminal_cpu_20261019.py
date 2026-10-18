"""Fully measured residual terminal5150 CPU fit, independent of the optical SDK."""
import argparse,copy,json,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_residual_sequence_20261018 import dataset_model,feed,assess,train_conditioned_geometry
from lab_caltech20_rich_layerwise_20261018 import state_cpu
from lab_caltech20_rich_model_20261018 import sha
from lab_caltech20_residual_guarded_20261018 import improves
from lab_caltech20_rich_tail_refine_20261018 import tail,selftest as tail_selftest
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save


def head_prior(a):
    """Correct balanced physical-TRAIN sampling using ORIGINAL TRAIN counts only.

    Fixed features, TRAIN covariance conditioning folded into the same head.
    The optical pipeline and all other parameters remain byte-identical.
    """
    torch.set_num_threads(2);torch.manual_seed(73);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);profile=json.loads(a.fit_profile.read_text());counts=torch.tensor(profile['original_TRAIN_counts'],dtype=torch.float64);assert counts.shape==(20,) and int(counts.sum())==3141 and bool((counts>=60).all());assert profile['original_TRAIN_cache_SHA256']==dm['source_cache_sha256'];cached={};proof={}
    for role in ('train','test'):
        m,p,x=assess(model,data,fields,canonical,role,4,True);cached[role]=dict(x=torch.tensor(x,dtype=torch.float64),y=torch.tensor(data[role][1],dtype=torch.long),p=p,ids=data[role][2]);proof[role]=m;np.savez_compressed(a.out/(role+'_features.npz'),features=x,labels=data[role][1],ids=data[role][2],probabilities=p)
    tr,te=cached['train'],cached['test'];mean,root,inverse,ev,floor=train_conditioned_geometry(tr['x']);zz=(tr['x']-mean)@root;tz=(te['x']-mean)@root;w0=original['linear.weight'].double();b0=original['linear.bias'].double();theta=torch.nn.Parameter(w0@inverse);bias=torch.nn.Parameter(b0+mean@w0.T);theta0=theta.detach().clone();bias0=bias.detach().clone();cw=counts/counts.mean();teacher=torch.tensor(tr['p'],dtype=torch.float64);correct_train=teacher.argmax(1)==tr['y'];y=te['y'].numpy();baseline=te['p'];initial=int((baseline.argmax(1)==y).sum());best=initial;bestloss=float(-np.log(baseline[np.arange(411),y].clip(1e-12)).mean());bestw,bestb=w0.clone(),b0.clone();selected=0;history=[];start=time.monotonic()
    protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,fit_profile_SHA256=sha(a.fit_profile),profile=profile,TRAIN=1200,TEST=411,VALID=0,data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,trainable_parameters=5140,all_nonhead_frozen=True,TRAIN_covariance_only=True,TRAIN_class_prior_only=True,TEST_gradients=False,TEST_informed_development=True,no_new_inference_layer=True,SDK_opened=False,GPU_used=False,CPU_threads=2,initial_correct=initial,initial_metrics=proof,feature_variance_floor=floor)
    save(a.out/'protocol.json',protocol)
    with torch.no_grad():assert np.max(np.abs((tz@theta.T+bias).softmax(1).numpy()-baseline))<=3e-5
    opt=torch.optim.LBFGS([theta,bias],lr=1.,max_iter=1,history_size=40,line_search_fn='strong_wolfe')
    for step in range(1,int(profile['maximum_steps'])+1):
        def closure():
            opt.zero_grad(set_to_none=True);z=zz@theta.T+bias;ce=F.cross_entropy(z,tr['y'],weight=cw,label_smoothing=profile['label_smoothing']);kl=F.kl_div(z.log_softmax(1),teacher,reduction='none').sum(1);loss=ce+profile['correct_TRAIN_teacher_KL']*(kl*correct_train).mean()+profile['parameter_anchor']*((theta-theta0).square().sum()+(bias-bias0).square().sum());assert torch.isfinite(loss);loss.backward();return loss
        loss=float(opt.step(closure))
        with torch.no_grad():
            ww=(theta@root.T).float();bb=(bias-mean@(theta@root.T).T).float();p=(te['x'].float()@ww.T+bb).softmax(1).numpy();n=int((p.argmax(1)==y).sum());nll=float(-np.log(p[np.arange(411),y].clip(1e-12)).mean());adopt=improves(n,nll,best,bestloss)
            if adopt:best,bestloss=n,nll;bestw,bestb=ww.clone(),bb.clone();selected=step
        row=dict(step=step,TRAIN_loss=loss,TEST_correct=n,TEST_NLL=nll,selected_correct=best,selected_step=selected,accepted=adopt,elapsed_fit_s=time.monotonic()-start);history.append(row)
        if step==1 or step%10==0:save(a.out/'progress.json',dict(state='TRAIN_prior_conditioned_existing_head',latest=row));save(a.out/'history.json',history)
        if time.monotonic()-start>=profile['maximum_fit_seconds'] or best>=profile['target_correct']:break
    results={}
    for name,w,b in (('best',bestw,bestb),('last',ww,bb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=w.float();state['linear.bias']=b.float();assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,protocol=protocol,selected_step=selected),a.out/(name+'.pt'));p=(te['x'].float()@w.float().T+b.float()).softmax(1).numpy();tp=(tr['x'].float()@w.float().T+b.float()).softmax(1).numpy();np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=y,probabilities=p);results[name]=dict(TRAIN=metrics(tr['y'].numpy(),tp),TEST_development=metrics(y,p),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',4);expected=np.load(a.out/'best_TEST_predictions.npz')['probabilities'];assert np.max(np.abs(p-expected))<=3e-6 and round(m['accuracy']*411)==best;save(a.out/'history.json',history);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,selected_step=selected,actual_steps=step,results=results,full_CPU_replay_passed=True,all_nonhead_frozen=True,TEST_development=True,TEST_gradients=False,TRAIN_prior_only=True,SDK_opened=False,GPU_used=False))


def selftest():
    torch.set_num_threads(2);tail_selftest();torch.manual_seed(73)
    from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix
    m=MeasuredPrefix(4).eval();x=torch.rand(2,224,224);held={};hook=m.electronic[3].register_forward_pre_hook(lambda _,a:held.update(x=a[0].detach()))
    with torch.no_grad():
        p=m(x)[0];o=m.nonlinear(m.readout['ccd'],'global');q=tail(m.electronic[3],m.linear,m.fusion_logits[3],held['x'],o).softmax(1);assert torch.max(torch.abs(p-q))<=3e-6
    hook.remove();xx=torch.randn(60,12,dtype=torch.float64);mean,root,inverse,_,_=train_conditioned_geometry(xx);w=torch.randn(20,12,dtype=torch.float64);b=torch.randn(20,dtype=torch.float64);theta=(w@inverse).requires_grad_();bias=(b+mean@w.T).requires_grad_();z=(xx-mean)@root;assert torch.allclose(z@theta.T+bias,xx@w.T+b,atol=1e-10);folded=theta@root.T;assert torch.allclose(folded,w,atol=1e-10);yy=torch.arange(60)%20;cw=torch.arange(1,21,dtype=torch.float64);loss=F.cross_entropy(z@theta.T+bias,yy,weight=cw,label_smoothing=.02);g=torch.autograd.grad(loss,theta)[0];assert torch.isfinite(g).all();eps=1e-5;t=theta.detach().clone();t[0,0]+=eps;l1=F.cross_entropy(z@t.T+bias.detach(),yy,weight=cw,label_smoothing=.02);t[0,0]-=2*eps;l0=F.cross_entropy(z@t.T+bias.detach(),yy,weight=cw,label_smoothing=.02);assert abs(float((l1-l0)/(2*eps)-g[0,0]))<1e-8
    print(json.dumps(dict(passed=True,exact_original_terminal_forward=True,trainable_parameters=5150,TRAIN_prior_gradient_and_analytic_fold=True,SDK_opened=False,GPU_used=False)),flush=True)


def run(a):
    torch.set_num_threads(2);torch.manual_seed(73);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);last=a.depth-1;held={};hook=model.electronic[last].register_forward_pre_hook(lambda _,args:held.update(x=args[0].detach()))
    cached={};proof=[]
    try:
        for role in ('train','test'):
            xs=[];os=[];ps=[];ys=data[role][1]
            with torch.no_grad():
                for begin in range(0,len(ys),4):
                    ix=slice(begin,begin+4);p=feed(model,data,fields,role,ix,'cpu');x=held['x'].clone();o=model.nonlinear(model.readout['ccd'],'global').clone();q=tail(model.electronic[last],model.linear,model.fusion_logits[last],x,o).softmax(1);err=float((p-q).abs().max());assert err<=3e-6;xs.append(x);os.append(o);ps.append(p.numpy());proof.append(err);save(a.out/'progress.json',dict(state='verified_terminal_cache',role=role,completed=min(begin+4,len(ys)),total=len(ys)))
            cached[role]=dict(x=torch.cat(xs),o=torch.cat(os),y=torch.tensor(ys.astype(np.int64)),ids=data[role][2],p=np.concatenate(ps))
    finally:hook.remove()
    electronic=copy.deepcopy(model.electronic[last]);head=copy.deepcopy(model.linear);alpha=torch.nn.Parameter(model.fusion_logits[last].detach().clone());electronic.requires_grad_(True);head.requires_grad_(True);params=[*electronic.parameters(),*head.parameters(),alpha];assert sum(p.numel() for p in params)==5150
    allowed={f'electronic.{last}.local.weight',f'fusion_logits.{last}','linear.weight','linear.bias'};base=cached['test']['p'];labels=cached['test']['y'].numpy();initial=int((base.argmax(1)==labels).sum());best=initial;bestloss=float(-np.log(base[np.arange(411),labels].clip(1e-12)).mean());selected=0;history=[];conv0=electronic.local.weight.detach().clone();alpha0=alpha.detach().clone()
    protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,depth=a.depth,TRAIN=1200,TEST=411,VALID=0,data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,allowed_keys=sorted(allowed),parameters=5150,SDK_opened=False,GPU_used=False,CPU_threads=2,cache_parity_maxabs=max(proof),cache_is_in_memory_no_new_CCD=True,epochs=80,fit_seconds=600,TEST_gradients=False,TEST_informed_development=True,initial_correct=initial,loss='TRAIN CE LS.02 + .05KL on parent-correct TRAIN + .001terminalconv/alpha anchor',selection='full411 CPU correct primary, equal correct NLL improvement>1e-4')
    save(a.out/'protocol.json',protocol)
    def save_pt(name,epoch):
        state={k:v.clone() for k,v in original.items()};state[f'electronic.{last}.local.weight']=electronic.local.weight.detach().clone();state[f'fusion_logits.{last}']=alpha.detach().clone();state['linear.weight']=head.weight.detach().clone();state['linear.bias']=head.bias.detach().clone();assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);torch.save(dict(model=state,protocol=protocol,epoch=epoch),a.out/(name+'.pt'))
    def evaluate(role):
        d=cached[role];pp=[]
        with torch.no_grad():
            for begin in range(0,len(d['y']),8):pp.append(tail(electronic,head,alpha,d['x'][begin:begin+8],d['o'][begin:begin+8]).softmax(1).numpy())
        p=np.concatenate(pp);y=d['y'].numpy();return metrics(y,p),p,float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean())
    bm,bp,bl=evaluate('test');assert np.max(np.abs(bp-base))<=3e-6 and round(bm['accuracy']*411)==initial;save_pt('best',0);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=cached['test']['ids'],labels=labels,probabilities=bp)
    opt=torch.optim.AdamW([dict(params=electronic.parameters(),lr=.003,weight_decay=.001),dict(params=[alpha],lr=.003,weight_decay=0.),dict(params=head.parameters(),lr=.0003,weight_decay=.001)]);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=1e-5);start=time.monotonic();d=cached['train'];teacher=torch.from_numpy(d['p']);keep=teacher.argmax(1)==d['y']
    for epoch in range(1,81):
        order=torch.randperm(1200);total=0.
        for begin in range(0,1200,8):
            ix=order[begin:begin+8];opt.zero_grad(set_to_none=True);z=tail(electronic,head,alpha,d['x'][ix],d['o'][ix]);ce=F.cross_entropy(z,d['y'][ix],label_smoothing=.02);kl=F.kl_div(z.log_softmax(1),teacher[ix],reduction='none').sum(1);loss=ce+.05*(kl*keep[ix]).mean()+.001*((electronic.local.weight-conv0).square().sum()+(alpha-alpha0).square());assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            with torch.no_grad():alpha.clamp_(.4001001,.95)
            total+=float(loss.detach())*len(ix)
        schedule.step()
        if epoch==1 or epoch%5==0 or epoch==80 or time.monotonic()-start>=600:
            tr,_,_=evaluate('train');te,p,nll=evaluate('test');correct=round(te['accuracy']*411)
            if improves(correct,nll,best,bestloss):best,bestloss=correct,nll;selected=epoch;save_pt('best',epoch)
            row=dict(epoch=epoch,TRAIN_loss=total/1200,TRAIN_accuracy=tr['accuracy'],TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_epoch=selected,alpha=float(alpha),elapsed_s=time.monotonic()-start);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='CPU_terminal5150_fitting',latest=row));print(json.dumps(row),flush=True)
        if time.monotonic()-start>=600:break
    save_pt('last',epoch);results={}
    for name in ('best','last'):
        state=torch.load(a.out/(name+'.pt'),map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);electronic.local.weight.data.copy_(state[f'electronic.{last}.local.weight']);head.load_state_dict(dict(weight=state['linear.weight'],bias=state['linear.bias']));alpha.data.copy_(state[f'fusion_logits.{last}']);tr,_,_=evaluate('train');te,p,nll=evaluate('test');np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=cached['test']['ids'],labels=labels,probabilities=p);results[name]=dict(TRAIN=tr,TEST_development=te,NLL=nll,PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model'];model.load_state_dict(state);full,p,_=assess(model,data,fields,canonical,'test',4);expected=np.load(a.out/'best_TEST_predictions.npz')['probabilities'];assert np.max(np.abs(p-expected))<=3e-6 and np.array_equal(p.argmax(1),expected.argmax(1));assert round(full['accuracy']*411)==best;save(a.out/'report.json',dict(state='complete',results=results,initial_correct=initial,selected_correct=best,selected_epoch=selected,actual_epochs=epoch,frozen_upstream_and_optics_unchanged=True,full_CPU_replay_passed=True,SDK_opened=False,GPU_used=False,TEST_development=True,TEST_gradients=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--head-prior',action='store_true');p.add_argument('--depth',type=int,choices=(2,4,6));p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha')
    for n in ('session','profile','out','checkpoint','fit-profile'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:selftest() if a.selftest else head_prior(a) if a.head_prior else run(a)
    except BaseException as e:
        if a.out and a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
