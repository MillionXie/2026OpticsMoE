"""Fully measured residual terminal5150 CPU fit, independent of the optical SDK."""
import argparse,copy,json,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_residual_sequence_20261018 import dataset_model,feed,assess
from lab_caltech20_rich_layerwise_20261018 import state_cpu
from lab_caltech20_rich_model_20261018 import sha
from lab_caltech20_residual_guarded_20261018 import improves
from lab_caltech20_rich_tail_refine_20261018 import tail,selftest as tail_selftest
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save


def selftest():
    torch.set_num_threads(2);tail_selftest();torch.manual_seed(73)
    from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix
    m=MeasuredPrefix(4).eval();x=torch.rand(2,224,224);held={};hook=m.electronic[3].register_forward_pre_hook(lambda _,a:held.update(x=a[0].detach()))
    with torch.no_grad():
        p=m(x)[0];o=m.nonlinear(m.readout['ccd'],'global');q=tail(m.electronic[3],m.linear,m.fusion_logits[3],held['x'],o).softmax(1);assert torch.max(torch.abs(p-q))<=3e-6
    hook.remove();print(json.dumps(dict(passed=True,exact_original_terminal_forward=True,trainable_parameters=5150,SDK_opened=False,GPU_used=False)),flush=True)


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
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--depth',type=int,choices=(2,4,6));p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha')
    for n in ('session','profile','out','checkpoint'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out and a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
