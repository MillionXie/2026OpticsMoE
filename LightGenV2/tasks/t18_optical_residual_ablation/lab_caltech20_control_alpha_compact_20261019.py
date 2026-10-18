"""Exact compact training replay of the EXISTING final bounded fusion and head.

Both images in final fusion are fixed when terminal convolution is frozen.
Pooling is linear; their two pooled fields and three second moments suffice.
This is a training-time algebraic cache, NOT a new deployed model or new CCD.
"""
import argparse,json,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,BOUNDED
from lab_caltech20_control_head_refine_20261018 import read,existing_head,evaluate
from lab_caltech20_rich_tail_refine_20261018 import rms,features,metrics
from lab_caltech10_20261015 import save,sha
HERE=Path(__file__).resolve().parent


def compact_features(raw,d):
    alpha=.8501+(.95-.8501)*raw.sigmoid()
    norm=((1-alpha).square()*d['moments'][:,0]+alpha.square()*d['moments'][:,1]+2*alpha*(1-alpha)*d['moments'][:,2]).sqrt().clamp_min(1e-6).detach()
    pooled=((1-alpha)*d['e']+alpha*d['o'])*(d['re']/norm)[:,None]
    return F.layer_norm(pooled,(256,),eps=1e-5)


def compact_fields(e,o):
    re,ro=rms(e).detach(),rms(o).detach();u,v=e/re,o/ro
    pool=lambda x:F.adaptive_avg_pool2d(x[:,None],(16,16)).flatten(1)
    mean=lambda x:x.sum((-2,-1))/(x.shape[-2]*x.shape[-1])
    return dict(e=pool(u),o=pool(v),re=re.flatten(),moments=torch.stack([mean(u*u),mean(v*v),mean(u*v)],1))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73);e=torch.rand(3,478,478);o=torch.rand_like(e);d=compact_fields(e,o);a=torch.tensor(.7,requires_grad=True);b=a.detach().clone().requires_grad_();aa=.8501+.0999*a.sigmoid();mix=(1-aa)*(e/rms(e))+aa*(o/rms(o));actual=features(rms(e)*mix/rms(mix).detach());cached=compact_features(b,d);err=float((actual-cached).abs().max());assert err<=3e-6;probe=torch.randn_like(actual);ga=torch.autograd.grad((actual*probe).sum(),a)[0];gb=torch.autograd.grad((cached*probe).sum(),b)[0];assert torch.allclose(ga,gb,atol=2e-5,rtol=1e-4)
    print(json.dumps(dict(passed=True,original_final_fusion_forward_gradient_equivalent=True,max_feature_error=err,SDK=False,GPU=False)),flush=True)


def export(a):
    assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2);r=read(a.parent/'report.json');warm=a.parent/'best.pt';assert r['state']=='complete' and sha(warm)==r['results']['best']['PT_SHA256'];state=torch.load(warm,map_location='cpu',weights_only=False)['model'];contract=read(a.arm/'contract.json');capture=read(a.arm/'capture_report.json');spec=contract['model'];model,_=load_checkpoint(a.arm.parent/spec['file'],spec['sha256'],6,0.,alpha_schema=BOUNDED);base=model.state_dict();allowed={'linear.weight','linear.bias','fusion_logits.5','electronic.5.local.weight'};assert all(torch.equal(v,state[k]) for k,v in base.items() if k not in allowed);model.load_state_dict(state,strict=True);assets=HERE/'runs/hardware/caltech20_rich20_L2_20261018/assets';dm=read(assets/'data_manifest.json');assert sha(assets/dm['file'])==dm['sha256']==contract['data_SHA256'];held={};hook=model.electronic[5].register_forward_pre_hook(lambda m,args:held.update(x=args[0]));files={};proof=[];arrays={}
    try:
        with np.load(assets/dm['file'],allow_pickle=False) as identity,torch.inference_mode():
            for role,n in [('train',1000),('test',411)]:
                fields={}
                for stage in ['router']+[str(k) for k in range(6)]:
                    p=a.arm/'capture'/role/stage/'ccd.npy';arrays[role+'/'+stage]=sha(p);assert arrays[role+'/'+stage]==capture['arrays'][role+'/'+stage]['sha256'];fields[stage]=np.load(p,mmap_mode='r')
                images=torch.from_numpy(identity[role+'_images'].transpose(0,3,1,2).copy());parts={};prob=[];error=0.
                for begin in range(0,n,8):
                    def measured(stage,amplitude,phase,ids):return np.asarray(fields[stage][begin:begin+8]).copy()
                    actual=model(encode_images(images[begin:begin+8]),callback=measured,ids=identity[role+'_ids'][begin:begin+8])[0];e=model.electronic[5](held['x'],478);o=model.nonlinear(model.readout['ccd'],'global');d=compact_fields(e,o);cf=compact_features(state['fusion_logits.5'],d);error=max(error,float((cf-model.readout['features']).abs().max()));pred=existing_head(state)(cf).softmax(1);assert float((pred-actual).abs().max())<=3e-6
                    for k,v in d.items():parts.setdefault(k,[]).append(v.numpy().copy())
                    prob.append(actual.numpy());save(a.out/'progress.json',dict(state='exporting_exact_final_fusion_cache',role=role,completed=min(begin+8,n),total=n,SDK=False,GPU=False))
                assert error<=3e-6;pp=np.concatenate(prob);correct=int((pp.argmax(1)==identity[role+'_labels']).sum());assert correct==round(n*r['results']['best']['TRAIN' if role=='train' else 'TEST_development']['accuracy']);path=a.out/(role+'.npz');np.savez_compressed(path,**{k:np.concatenate(v) for k,v in parts.items()},probabilities=pp,labels=identity[role+'_labels'],ids=identity[role+'_ids']);files[role]=sha(path);proof.append(dict(role=role,n=n,correct=correct,compact_feature_error=error))
    finally:hook.remove()
    save(a.out/'report.json',dict(state='complete',passed=True,files_SHA256=files,proof=proof,parent_PT_SHA256=sha(warm),CCD_arrays_SHA256=arrays,all_captured_upstream_frozen=True,terminal_conv_frozen=True,new_CCD=0,SDK=False,GPU=False,source_SHA256=sha(__file__)));print(json.dumps(dict(state='complete',proof=proof)),flush=True)


def fit(a):
    assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(17);cfg=read(a.profile);proof=read(a.cache/'report.json');assert proof['passed'];pr=read(a.parent/'report.json');warm=a.parent/'best.pt';assert sha(warm)==pr['results']['best']['PT_SHA256']==proof['parent_PT_SHA256'];state=torch.load(warm,map_location='cpu',weights_only=False)['model'];head=existing_head(state);raw=torch.nn.Parameter(state['fusion_logits.5'].clone());data={}
    for role,n in [('train',1000),('test',411)]:
        p=a.cache/(role+'.npz');assert sha(p)==proof['files_SHA256'][role]
        with np.load(p,allow_pickle=False) as z:data[role]={k:torch.from_numpy(z[k].copy()) for k in z.files}
        assert len(data[role]['ids'])==n
    assert not set(data['train']['ids'].tolist())&set(data['test']['ids'].tolist());assert torch.bincount(data['train']['labels'],minlength=20).tolist()==[50]*20
    def assess(role):
        with torch.no_grad():p=head(compact_features(raw,data[role])).softmax(1).numpy()
        return metrics(data[role]['labels'].numpy(),p),p
    baseline,bp=assess('test');tr0,tp0=assess('train');assert baseline['accuracy']==pr['results']['best']['TEST_development']['accuracy'];assert float(np.abs(bp-data['test']['probabilities'].numpy()).max())<=3e-6;initial=best=round(411*baseline['accuracy']);best_nll=float(-np.log(bp[np.arange(411),data['test']['labels'].numpy()].clip(1e-12)).mean());teacher=torch.from_numpy(tp0);selected=0;history=[];allowed={'linear.weight','linear.bias','fusion_logits.5'};git=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe';protocol=dict(config=cfg,source_SHA256=sha(__file__),profile_SHA256=sha(a.profile),actual_git=subprocess.check_output([str(git) if git.exists() else 'git','rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),parent_PT_SHA256=sha(warm),cache_SHA256=proof['files_SHA256'],allowed_keys=sorted(allowed),trainable_parameters=5141,all_captured_upstream_frozen=True,terminal_conv_frozen=True,TRAIN=1000,TEST=411,VALID=0,TRAIN_gradients_only=True,TEST_development=True,SDK=False,GPU=False,new_layers=0,new_CCD=0);save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',probabilities=bp,labels=data['test']['labels'].numpy(),ids=data['test']['ids'].numpy())
    def persist(kind,epoch):
        z={k:v.clone() for k,v in state.items()};z['fusion_logits.5']=raw.detach().clone()
        for k,v in head.state_dict().items():z['linear.'+k]=v.detach().clone()
        assert all(torch.equal(v,z[k]) for k,v in state.items() if k not in allowed);torch.save(dict(model=z,selected=dict(epoch=epoch),protocol=protocol),a.out/(kind+'.pt'))
    persist('best',0);optim=torch.optim.AdamW([dict(params=head.parameters(),lr=cfg['head_lr']),dict(params=[raw],lr=cfg['alpha_lr'],weight_decay=0.)],weight_decay=cfg['weight_decay']);cw=torch.tensor(cfg['original_TRAIN_class_counts'],dtype=torch.float32);cw=cw/cw.mean();started=time.perf_counter();d=data['train']
    for epoch in range(1,cfg['epochs']+1):
        order=torch.randperm(1000);loss_total=0.
        for begin in range(0,1000,64):
            ix=order[begin:begin+64];part={k:v[ix] for k,v in d.items()};yy=part['labels'];optim.zero_grad();logits=head(compact_features(raw,part));loss=F.cross_entropy(logits,yy,weight=cw,label_smoothing=.02);keep=teacher[ix].argmax(1)==yy
            if keep.any():loss=loss+cfg['parent_KL']*F.kl_div(logits[keep].log_softmax(1),teacher[ix][keep],reduction='batchmean')
            assert torch.isfinite(loss);loss.backward();assert raw.grad is not None and torch.isfinite(raw.grad);optim.step();loss_total+=float(loss.detach())*len(ix)
        tr,_=assess('train');te,p=assess('test');n=round(te['accuracy']*411);nll=float(-np.log(p[np.arange(411),data['test']['labels'].numpy()].clip(1e-12)).mean())
        if n>best or (n==best and nll<best_nll-.0001):best,best_nll,selected=n,nll,epoch;persist('best',epoch)
        row=dict(epoch=epoch,TRAIN_loss=loss_total/1000,TRAIN_accuracy=tr['accuracy'],TEST_correct=n,best_correct=best,effective_alpha=float((.8501+.0999*raw.sigmoid()).detach()),elapsed_s=time.perf_counter()-started);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',row)
        if best>=cfg['target_min_correct'] or time.perf_counter()-started>=cfg['maximum_seconds']:break
    persist('last',epoch);results={}
    for kind in ('best','last'):
        p=a.out/(kind+'.pt');s=torch.load(p,map_location='cpu',weights_only=False)['model'];head.load_state_dict(existing_head(s).state_dict());raw.data.copy_(s['fusion_logits.5']);tr,trp=assess('train');te,tep=assess('test')
        for role,pp in [('train',trp),('test',tep)]:np.savez_compressed(a.out/(kind+'_'+role.upper()+'_predictions.npz'),probabilities=pp,labels=data[role]['labels'].numpy(),ids=data[role]['ids'].numpy())
        results[kind]=dict(TRAIN=tr,TEST_development=te,PT_SHA256=sha(p),strict_CPU=True,all_optical_and_captured_upstream_unchanged=True)
    assert best>=initial;save(a.out/'report.json',dict(state='complete',depth=6,baseline_correct=initial,best_correct=best,best_NLL=best_nll,selected_epoch=selected,actual_epochs=epoch,results=results,target_min_correct=cfg['target_min_correct'],target_met=best>=cfg['target_min_correct'],only_existing_post_final_CCD_changed=True,TRAIN=1000,TEST=411,VALID=0,TEST_development=True,SDK=False,GPU=False));print(json.dumps(dict(state='complete',baseline_correct=initial,best_correct=best,selected_epoch=selected)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--export',action='store_true')
    for n in ('arm','parent','out','cache','profile'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        try:export(a) if a.export else fit(a)
        except BaseException as e:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
            raise
