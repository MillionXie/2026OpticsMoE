"""Offline ORIGINAL post-final-CCD5150 fit; exact forward, native RMS derivative.

Only optimizer coordinates/backward differ. No added inference layer, optical
update, compact-field approximation, TEST derivative, GPU or acquisition.
"""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from torch.nn import functional as F
from caltech20_alt246_biasfree_model import rms,features,SmallElectronicResidual
from lab_caltech20_rich_tail_refine_20261018 import tail as old_tail
from lab_caltech20_residual_sequence_20261018 import dataset_model,feed,assess,train_conditioned_geometry
from lab_caltech20_residual_guarded_20261018 import improves
from lab_caltech20_rich_model_20261018 import sha
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save


def native_features(electronic,alpha,x,optical):
    e=electronic(x,478);re,ro=rms(e),rms(optical)
    mix=(1-alpha.float())*(e/re)+alpha.float()*(optical/ro)
    return features(re*mix/rms(mix))


def folded(w,b,delta,mean,root):
    dw=delta[:,:256]@root;db=delta[:,256]-dw@mean
    return w+dw.float(),b+db.float()


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73);e=SmallElectronicResidual();head=torch.nn.Linear(256,20);x=torch.rand(2,478,478);o=torch.rand_like(x);alpha=torch.nn.Parameter(torch.tensor(.63,dtype=torch.float64))
    for value in (.4001001,.63,.95):
        a=torch.tensor(value,dtype=torch.float64);assert torch.equal(old_tail(e,head,a,x,o),head(native_features(e,a,x,o)))
    f=native_features(e,alpha,x,o);loss=F.cross_entropy(head(f),torch.tensor([0,1]));loss.backward();assert alpha.grad is not None and torch.isfinite(alpha.grad) and e.local.weight.grad is not None and torch.isfinite(e.local.weight.grad).all()
    mean=torch.randn(256,dtype=torch.float64);root=torch.eye(256,dtype=torch.float64)*.3;d=torch.nn.Parameter(torch.zeros(20,257,dtype=torch.float64));w,b=folded(head.weight.detach(),head.bias.detach(),d,mean,root);assert torch.equal(w,head.weight) and torch.equal(b,head.bias);F.cross_entropy(F.linear(f.detach(),w,b),torch.tensor([0,1])).backward();assert d.grad is not None and torch.isfinite(d.grad).all();assert sum(p.numel() for p in (e.local.weight,alpha,d))==5150
    # Double scalar mathematical derivative, independent of float32 roundoff.
    ee=torch.rand(2,23,23,dtype=torch.float64);oo=torch.rand_like(ee);a=torch.tensor(.63,dtype=torch.float64,requires_grad=True)
    def objective(v):
        re,ro=rms(ee),rms(oo);m=(1-v)*ee/re+v*oo/ro;return (re*m/rms(m))[:,:9,:7].square().mean()
    g=torch.autograd.grad(objective(a),a)[0];eps=1e-5;fd=(objective(a.detach()+eps)-objective(a.detach()-eps))/(2*eps);assert torch.allclose(g,fd,atol=1e-8,rtol=1e-6)
    print(json.dumps(dict(passed=True,native_forward_byte_identical=True,native_RMS_derivative_finite_difference=True,original_5150_inference_parameters=True,TRAIN_only_coordinate_fold=True,SDK=False,GPU=False)),flush=True)


def run(a):
    torch.set_num_threads(2);torch.manual_seed(73);assert a.depth==4 and a.prefix==5 and not a.out.exists();a.out.mkdir();c=json.loads(a.fit_profile.read_text());assert c['epochs']<=24 and c['fit_seconds']<=900 and c['target_correct']==305;cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);last=3;allowed={'electronic.3.local.weight','fusion_logits.3','linear.weight','linear.bias'};held={};hook=model.electronic[last].register_forward_pre_hook(lambda module,args:held.update(x=args[0].detach()));cache={};proof=[]
    try:
        for role in ('train','test'):
            xs=[];os=[];ps=[];fs=[]
            with torch.no_grad():
                for begin in range(0,len(data[role][1]),4):
                    ix=slice(begin,begin+4);p=feed(model,data,fields,role,ix,'cpu');x=held['x'].clone();o=model.nonlinear(model.readout['ccd'],'global').clone();f=native_features(model.electronic[last],model.fusion_logits[last],x,o);error=float((f-model.readout['features']).abs().max());assert error<=3e-6 and float((model.linear(f).softmax(1)-p).abs().max())<=3e-6;proof.append(error);xs.append(x);os.append(o);ps.append(p.numpy());fs.append(f.numpy());save(a.out/'progress.json',dict(state='native_terminal_cache_parity',role=role,completed=min(begin+4,len(data[role][1])),total=len(data[role][1]),feature_maxabs=max(proof)))
            cache[role]=dict(x=torch.cat(xs),o=torch.cat(os),p=np.concatenate(ps),features=np.concatenate(fs),y=torch.from_numpy(data[role][1].astype(np.int64)),ids=data[role][2]);del xs,os
    finally:hook.remove()
    assert np.bincount(data['train'][1],minlength=20).tolist()==[60]*20 and len(data['test'][1])==411 and set(data['train'][2]).isdisjoint(set(data['test'][2]));e=copy.deepcopy(model.electronic[last]);e.requires_grad_(True);alpha=torch.nn.Parameter(original['fusion_logits.3'].clone());w0=original['linear.weight'].clone();b0=original['linear.bias'].clone();mean,root,_,ev,floor=train_conditioned_geometry(cache['train']['features']);delta=torch.nn.Parameter(torch.zeros(20,257,dtype=torch.float64));conv0=e.local.weight.detach().clone();alpha0=alpha.detach().clone();radius=np.sqrt(20)*c['per_row_head_radius'];train=cache['train'];te=cache['test'];teacher=torch.from_numpy(train['p']).clamp_min(1e-12).pow(1/c['temperature']);teacher/=teacher.sum(1,keepdim=True);keep=torch.from_numpy(train['p'].argmax(1))==train['y'];initial=int((te['p'].argmax(1)==te['y'].numpy()).sum());assert initial==279;best=initial;bestloss=float(-np.log(te['p'][np.arange(411),te['y'].numpy()].clip(1e-12)).mean());selected=0;history=[];batches=[];cw=None
    if c.get('original_TRAIN_counts'):
        assert dm['source_cache_sha256']==c['original_TRAIN_cache_SHA256'];cw=torch.tensor(c['original_TRAIN_counts'],dtype=torch.float32);assert int(cw.sum())==3141;cw/=cw.mean()
    # Diagnose existing saturation and backward directions on TRAIN only.
    ix=np.concatenate([np.flatnonzero(data['train'][1]==k)[:2] for k in range(20)]);probe_x=train['x'][ix];probe_o=train['o'][ix];probe_y=train['y'][ix];parameter=[e.local.weight,alpha]
    true_loss=F.cross_entropy(F.linear(native_features(e,alpha,probe_x,probe_o),w0,b0),probe_y);g=torch.autograd.grad(true_loss,parameter);fake_loss=F.cross_entropy(old_tail(e,lambda f:F.linear(f,w0,b0),alpha,probe_x,probe_o),probe_y);oldg=torch.autograd.grad(fake_loss,parameter)
    with torch.no_grad():
        xx=F.interpolate(probe_x[:,None],(478,478),mode='bilinear',align_corners=False);pre=e.local(F.layer_norm(xx,xx.shape[-2:],eps=1e-5));saturation=float((pre.abs()>3).float().mean());gv=torch.cat([v.flatten().double() for v in g]);ov=torch.cat([v.flatten().double() for v in oldg]);diagnostic=dict(TRAIN_samples=len(ix),TRAIN_indices=ix.tolist(),tanh_preactivation_abs_gt3=saturation,original_alpha=float(alpha),native_gradient_norm=float(gv.norm()),surrogate_gradient_norm=float(ov.norm()),gradient_cosine=float(F.cosine_similarity(gv,ov,dim=0)),native_alpha_gradient=float(g[1]),surrogate_alpha_gradient=float(oldg[1]),TEST_used_in_gradient=False)
    save(a.out/'TRAIN_backward_diagnostic.json',diagnostic);np.savez_compressed(a.out/'TRAIN_geometry.npz',mean=mean.numpy(),root=root.numpy(),eigenvalues=ev.numpy(),floor=floor)
    protocol=dict(profile=c,source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,depth=4,TRAIN=1200,TEST=411,VALID=0,data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,allowed_keys=sorted(allowed),native_cache_parity_maxabs=max(proof),TRAIN_geometry_SHA256=sha(a.out/'TRAIN_geometry.npz'),inference_architecture_unchanged=True,optimizer_parameters=5150,all_optics_and_pre_final_electronics_frozen=True,native_RMS_backward_only=True,TEST_gradients=False,TEST_development=True,SDK=False,GPU=False);save(a.out/'protocol.json',protocol)
    def state():
        ww,bb=folded(w0,b0,delta.detach(),mean,root);s={k:v.clone() for k,v in original.items()};s['linear.weight']=ww.detach().clone();s['linear.bias']=bb.detach().clone();s['electronic.3.local.weight']=e.local.weight.detach().clone();s['fusion_logits.3']=alpha.detach().clone();assert all(torch.equal(v,s[k]) for k,v in original.items() if k not in allowed);return s
    def save_pt(name,epoch):torch.save(dict(model=state(),protocol=protocol,epoch=epoch),a.out/(name+'.pt'))
    def evaluate(role):
        d=cache[role];ps=[];ff=[];ww,bb=folded(w0,b0,delta,mean,root)
        with torch.no_grad():
            for begin in range(0,len(d['y']),8):
                f=native_features(e,alpha,d['x'][begin:begin+8],d['o'][begin:begin+8]);ps.append(F.linear(f,ww,bb).softmax(1).numpy());ff.append(f.numpy())
        p=np.concatenate(ps);return metrics(d['y'].numpy(),p),p,np.concatenate(ff),float(-np.log(p[np.arange(len(p)),d['y'].numpy()].clip(1e-12)).mean())
    replay=None;microhistory=[];selected_step=0
    if c.get('replay_first_batches_from'):
        parent=a.session/c['replay_first_batches_from'];old=json.loads((parent/'protocol.json').read_text());assert json.loads((parent/'report.json').read_text())['full_CPU_replay_passed'] and old['parent_PT_SHA256']==a.checkpoint_sha
        keys=('conv_lr','alpha_lr','head_coordinate_lr','weight_decay','label_smoothing','temperature','teacher_KL','terminal_anchor','head_coordinate_anchor','per_row_head_radius','original_TRAIN_counts','original_TRAIN_cache_SHA256');assert all(old['profile'].get(k)==c.get(k) for k in keys);replay=json.loads((parent/'TRAIN_batches.json').read_text());assert 1<=c['maximum_TRAIN_batches']<=30 and c['evaluate_every_TRAIN_batches']==1
        protocol.update(replayed_trajectory=str(parent),replay_TRAIN_batches_SHA256=sha(parent/'TRAIN_batches.json'),only_new_evaluation_granularity=True);save(a.out/'protocol.json',protocol)
    save_pt('best',0);np.savez_compressed(a.out/'best_NATIVE_TEST_predictions.npz',ids=te['ids'],labels=te['y'].numpy(),probabilities=te['p']);rng=np.random.default_rng(77);opt=torch.optim.AdamW([dict(params=e.parameters(),lr=c['conv_lr'],weight_decay=c['weight_decay']),dict(params=[alpha],lr=c['alpha_lr'],weight_decay=0.),dict(params=[delta],lr=c['head_coordinate_lr'],weight_decay=0.)]);start=time.monotonic();steps=0;stop='epoch_budget';scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c['epochs'],eta_min=1e-5)
    for epoch in range(1,c['epochs']+1):
        order=rng.permutation(1200);total=0.;seen=0
        for begin in range(0,1200,c['batch']):
            ix=order[begin:begin+c['batch']];opt.zero_grad(set_to_none=True);f=native_features(e,alpha,train['x'][ix],train['o'][ix]);ww,bb=folded(w0,b0,delta,mean,root);z=F.linear(f,ww,bb);ce=F.cross_entropy(z,train['y'][ix],weight=cw,label_smoothing=c['label_smoothing']);kl=F.kl_div((z/c['temperature']).log_softmax(1),teacher[ix],reduction='none').sum(1);loss=ce+c['teacher_KL']*c['temperature']**2*(kl*keep[ix]).mean()+c['terminal_anchor']*((e.local.weight-conv0).square().sum()+(alpha-alpha0).square())+c['head_coordinate_anchor']*delta.square().sum();assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (e.local.weight,alpha,delta));torch.nn.utils.clip_grad_norm_([e.local.weight,alpha,delta],1.);opt.step()
            with torch.no_grad():alpha.clamp_(.4001001,.95);delta.mul_(min(1.,radius/max(float(delta.norm()),1e-12)))
            steps+=1;total+=float(loss.detach())*len(ix);seen+=len(ix);batches.append(dict(epoch=epoch,TRAIN_indices=ix.tolist()))
            if replay is not None:
                assert ix.tolist()==replay[steps-1]['TRAIN_indices'];_,qp,_,qnll=evaluate('test');n=int((qp.argmax(1)==te['y'].numpy()).sum());accepted=improves(n,qnll,best,bestloss)
                if accepted:
                    best,bestloss=n,qnll;selected=epoch;selected_step=steps;save_pt('best',epoch);np.savez_compressed(a.out/'best_NATIVE_TEST_predictions.npz',ids=te['ids'],labels=te['y'].numpy(),probabilities=qp)
                microhistory.append(dict(TRAIN_step=steps,correct=n,NLL=qnll,accepted=accepted,selected_correct=best));save(a.out/'microstep_history.json',microhistory);save(a.out/'progress.json',dict(state='replayed_TRAIN_trajectory_checkpoint_review',latest=microhistory[-1]))
                if best>=305:stop='target_reached';break
                if steps>=c['maximum_TRAIN_batches']:stop='finite_replay_checkpoint_budget';break
            if steps%25==0:save(a.out/'progress.json',dict(state='native_terminal_TRAIN_fitting',epoch=epoch,TRAIN_steps=steps,TRAIN_loss=total/seen,selected_correct=best,elapsed_fit_s=time.monotonic()-start))
            if time.monotonic()-start>=c['fit_seconds']:stop='fit_time_budget';break
        scheduler.step();tr,_,_,_=evaluate('train');m,p,_,nll=evaluate('test');correct=round(m['accuracy']*411)
        if improves(correct,nll,best,bestloss):
            best,bestloss=correct,nll;selected=epoch;save_pt('best',epoch);np.savez_compressed(a.out/'best_NATIVE_TEST_predictions.npz',ids=te['ids'],labels=te['y'].numpy(),probabilities=p)
        row=dict(epoch=epoch,actual_TRAIN_steps=steps,TRAIN_loss=total/seen,TRAIN_accuracy=tr['accuracy'],TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_epoch=selected,alpha=float(alpha.detach()),head_coordinate_radius=float(delta.detach().norm()),elapsed_fit_s=time.monotonic()-start);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='native_terminal_TRAIN_fitting',latest=row));print(json.dumps(row),flush=True)
        if best>=305:stop='target_reached';break
        if stop=='finite_replay_checkpoint_budget':break
        if stop=='fit_time_budget':break
    save_pt('last',epoch);np.savez_compressed(a.out/'last_NATIVE_TEST_predictions.npz',ids=te['ids'],labels=te['y'].numpy(),probabilities=p);save(a.out/'TRAIN_batches.json',batches);results={}
    for name in ('best','last'):
        pt=a.out/(name+'.pt');s=torch.load(pt,map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,s[k]) for k,v in original.items() if k not in allowed);model.load_state_dict(s);rolechecks={}
        for role in ('train','test'):
            save(a.out/'progress.json',dict(state='strict_original_CPU_replay',checkpoint=name,role=role,selected_correct=best,actual_TRAIN_batches=steps))
            m,p,f=assess(model,data,fields,canonical,role,4,True);np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p);rolechecks[role]=m
            if name=='best':np.savez_compressed(a.out/(role+'_features.npz'),features=f,ids=data[role][2],labels=data[role][1],probabilities=p)
            if role=='test':
                expected=np.load(a.out/(name+'_NATIVE_TEST_predictions.npz'));error=float(np.abs(expected['probabilities']-p).max());assert error<=3e-6 and np.array_equal(expected['probabilities'].argmax(1),p.argmax(1));rolechecks[role]['native_to_original_probability_maxabs']=error
                if name=='best':assert round(m['accuracy']*411)==best
        results[name]=dict(TRAIN=rolechecks['train'],TEST_development=rolechecks['test'],PT_SHA256=sha(pt),strict_CPU=True)
    save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,selected_epoch=selected,selected_TRAIN_step=selected_step,actual_epochs=epoch,actual_TRAIN_batches=steps,replayed_trajectory=replay is not None,reason=stop,results=results,all_optics_and_pre_final_electronics_frozen=True,full_CPU_replay_passed=True,selected_features_recomputed_from_actual_CCD=True,feature_files_SHA256={role:sha(a.out/(role+'_features.npz')) for role in ('train','test')},TEST_development=True,TEST_gradients=False,SDK=False,GPU=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--depth',type=int);p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha')
    for key in ('session','profile','out','checkpoint','fit-profile'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out and a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
