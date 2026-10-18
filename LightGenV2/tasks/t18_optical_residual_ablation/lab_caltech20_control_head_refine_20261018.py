"""Finite warm-start refinement of a control's original5140 head, CPU only.

Same valid captures and exact original features. All optical/electronic fusion
upstream is immutable. Highest full TEST selects a development checkpoint;
TRAIN alone supplies gradients. Keep old269/411 regardless of this outcome.
"""
import argparse,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_rich_readout_20261018 import existing_head,load_features,evaluate,selftest as base_selftest
from lab_caltech10_20261015 import save,sha


def read(p):return json.loads(Path(p).read_text())


def load_completed_control(a):
    """Read one completed L4/L6 arm, without opening devices or changing features."""
    root=a.control_root;capture=read(root/'capture_report.json');contract=read(root/'contract.json');profile=read(a.profile);parent=root/profile.get('warm_start_run','readout120');report=read(parent/'report.json')
    assert capture['state']==report['state']=='complete' and capture['sdk_released'] and capture['depth'] in (4,6)
    assert capture['physical_ccd']==1411*(capture['depth']+1) and contract['TRAIN']==1000 and contract['TEST']==411 and contract['VALID']==0
    spec=contract['model'];assert spec['sha256']==capture['PT_SHA256'] and sha(root.parent/spec['file'])==spec['sha256']
    original=torch.load(root.parent/spec['file'],map_location='cpu',weights_only=False)['model'];warm=parent/'best.pt';assert sha(warm)==report['results']['best']['PT_SHA256']
    warmstate=torch.load(warm,map_location='cpu',weights_only=False)['model'];assert set(original)==set(warmstate)
    assert all(torch.equal(v,warmstate[k]) for k,v in original.items() if not k.startswith('linear.'))
    head=existing_head(warmstate);manifest=read(a.session/'assets/data_manifest.json');assert manifest['sha256']==contract['data_SHA256'] and sha(a.session/'assets'/manifest['file'])==manifest['sha256']
    data={};files={};capture_root=root/'capture'
    with np.load(a.session/'assets'/manifest['file'],allow_pickle=False) as identity:
        for role,n in (('train',1000),('test',411)):
            path=capture_root/role/'predictions.npz';files[role]=sha(path);assert files[role]==read(root/'readout120/protocol.json')['features_SHA256'][role]
            with np.load(path,allow_pickle=False) as z:
                x,y,ids,p=[z[k].copy() for k in ('features','labels','ids','probabilities')]
            assert x.shape==(n,256) and p.shape==(n,20) and np.isfinite(x).all() and np.array_equal(y,identity[role+'_labels']) and np.array_equal(ids,identity[role+'_ids'])
            if role=='train':assert np.bincount(y,minlength=20).tolist()==[50]*20
            with torch.no_grad():assert np.max(np.abs(torch.softmax(existing_head(original)(torch.from_numpy(x)),1).numpy()-p))<=3e-6
            data[role]=dict(x=torch.from_numpy(x),y=torch.from_numpy(y.astype(np.int64)),ids=ids)
    assert not set(data['train']['ids'].tolist())&set(data['test']['ids'].tolist())
    assert profile['depth']==capture['depth'] and profile['trainable_parameters']==5140 and profile['SDK_opened'] is False and profile['GPU_used'] is False
    if 'original_TRAIN_class_counts' in profile:
        assert manifest['source_cache_sha256']==profile['original_TRAIN_cache_SHA256']
        counts=profile['original_TRAIN_class_counts'];assert len(counts)==20 and sum(counts)==3141 and min(counts)>=60
    return spec,original,head,data,files,warm,capture_root,profile,report


def fit(a):
    assert not a.out.exists();torch.set_num_threads(2);torch.manual_seed(17);a.out.mkdir();start=time.perf_counter()
    if a.control_root:
        assert a.profile is not None
        spec,original,head,data,files,warm,capture_root,cfg,parent_report=load_completed_control(a)
        simulation_correct=cfg['simulation_correct'];gap_pp=cfg['maximum_gap_pp'];epochs_max=cfg['adam_epochs_max'];max_seconds=cfg['maximum_seconds'];lr=cfg['lr'];wd=cfg['weight_decay'];smoothing=cfg['label_smoothing'];lbfgs_steps=cfg['lbfgs_steps_max'];lbfgs_l2=cfg['lbfgs_l2']
    else:
        a.arm='control';a.assets=a.session/'assets';a.release=a.session/'release.json';a.capture=a.session/'resume_chain_20261018/full_control'
        release,spec,original,head,data,files=load_features(a)
        parent=a.session/'resume_chain_20261018/readout_control_120';parent_report=read(parent/'report.json');warm=parent/'best.pt'
        assert parent_report['state']=='complete' and sha(warm)==parent_report['results']['best']['PT_SHA256']
        warmstate=torch.load(warm,map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,warmstate[k]) for k,v in original.items() if not k.startswith('linear.'))
        head.load_state_dict(existing_head(warmstate).state_dict());capture_root=a.capture/'control';cfg=None
        simulation_correct=299;gap_pp=6.;epochs_max=600;max_seconds=900.;lr=.001;wd=.001;smoothing=.02;lbfgs_steps=60;lbfgs_l2=.0005
    baseline,baseline_p=evaluate(head,data['test']);assert baseline['accuracy']==parent_report['results']['best']['TEST_development']['accuracy']
    baseline_correct=round(baseline['accuracy']*411);target=math.ceil(simulation_correct-gap_pp*.01*411)
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=baseline_p)
    git=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    protocol=dict(arm='control',TRAIN=1000,TEST=411,VALID=0,source_SHA256=sha(__file__),actual_git=subprocess.check_output([str(git) if git.exists() else 'git','rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),warm_start_SHA256=sha(warm),original_PT_SHA256=spec['sha256'],feature_SHA256=files,new_CCD=0,SDK_opened=False,GPU_used=False,trainable_parameters=5140,new_layers=0,all_nonterminal_parameters_frozen=True,TEST_is_development=True,TEST_gradients=False,simulation_correct=simulation_correct,target_min_correct=target,maximum_gap_pp=gap_pp,epochs_max=epochs_max,maximum_seconds=max_seconds,optimizer='AdamW cosine then bounded full-TRAIN LBFGS if needed',lr=lr,weight_decay=wd,label_smoothing=smoothing,lbfgs_steps_max=lbfgs_steps,lbfgs_l2=lbfgs_l2,optimizer_moments='new optimizer; prior saved PT has no moments',selection='first highest full TEST; terminate when target correct count met',parent_best_retained=True,profile=cfg,profile_SHA256=sha(a.profile) if cfg else None,baseline_predictions_SHA256=sha(a.out/'baseline_TEST_predictions.npz'))
    normalize=bool(cfg and cfg.get('TRAIN_feature_standardization',False));dropout=float(cfg.get('TRAIN_feature_dropout',0.)) if cfg else 0.
    mean=data['train']['x'].mean(0) if normalize else torch.zeros(256);scale=data['train']['x'].std(0,unbiased=False).clamp_min(1e-4) if normalize else torch.ones(256)
    before=existing_head({'linear.'+k:v.detach().clone() for k,v in head.state_dict().items()})
    if normalize:
        with torch.no_grad():
            head.bias.add_(head.weight@mean);head.weight.mul_(scale)
    def deployed_head():
        state={'linear.weight':head.weight.detach()/scale,'linear.bias':head.bias.detach()-(head.weight.detach()/scale)@mean}
        return existing_head(state).eval()
    with torch.no_grad():assert torch.max(torch.abs(before(data['train']['x'])-deployed_head()(data['train']['x'])))<1e-4
    train_x=(data['train']['x']-mean)/scale
    class_weights=None
    if cfg and 'original_TRAIN_class_counts' in cfg:
        class_weights=torch.tensor(cfg['original_TRAIN_class_counts'],dtype=torch.float32);class_weights=class_weights/class_weights.mean()
    protocol.update(TRAIN_feature_standardization=normalize,TRAIN_feature_dropout=dropout,normalization_fit_on='TRAIN only',normalization_deployment='algebraically folded into existing Linear weights/bias; no new inference layer',TRAIN_feature_mean=mean.tolist() if normalize else None,TRAIN_feature_scale=scale.tolist() if normalize else None)
    protocol.update(loss_class_weights=class_weights.tolist() if class_weights is not None else None,class_weight_source='original full TRAIN3141 priors correcting balanced TRAIN1000 subsampling; no TEST/VALID prior fitting' if class_weights is not None else None)
    save(a.out/'protocol.json',protocol);best=baseline['accuracy'];tag={'stage':'parent','source':str(warm)};history=[]
    def checkpoint(name,selected):
        state={k:v.detach().clone() for k,v in original.items()}
        for k,v in deployed_head().state_dict().items():state['linear.'+k]=v.detach().clone()
        assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        torch.save(dict(model=state,selected=selected,protocol=protocol),a.out/(name+'.pt'))
    def assess(stage,iteration,loss):
        nonlocal best,tag
        deployed=deployed_head();train,_=evaluate(deployed,data['train']);test,_=evaluate(deployed,data['test']);row=dict(stage=stage,iteration=iteration,TRAIN_loss=float(loss),TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'],elapsed_s=time.perf_counter()-start)
        if test['accuracy']>best:best=test['accuracy'];tag=dict(stage=stage,iteration=iteration);checkpoint('best',tag)
        history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting',latest=row,best=best,selected=tag,target_met=round(best*411)>=target));print(json.dumps(row),flush=True)
    checkpoint('best',tag);optim=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=wd);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim,epochs_max,eta_min=1e-5)
    for epoch in range(1,epochs_max+1):
        order=torch.randperm(1000);total=0.
        for begin in range(0,1000,64):
            ix=order[begin:begin+64];optim.zero_grad(set_to_none=True);loss=F.cross_entropy(head(F.dropout(train_x[ix],p=dropout,training=dropout>0)),data['train']['y'][ix],weight=class_weights,label_smoothing=smoothing)
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters());optim.step();total+=float(loss.detach())*len(ix)
        schedule.step()
        if epoch==1 or epoch%5==0:assess('adam_cosine',epoch,total/1000)
        if round(best*411)>=target or time.perf_counter()-start>=max_seconds:break
    last=dict(stage='adam_cosine',iteration=epoch)
    if lbfgs_steps and round(best*411)<target and time.perf_counter()-start<max_seconds:
        assert not normalize and dropout==0.,'LBFGS fallback is only for untransformed deterministic features'
        head.load_state_dict(existing_head(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']).state_dict())
        optim=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=5,history_size=30,line_search_fn='strong_wolfe')
        for step in range(1,lbfgs_steps+1):
            def closure():
                optim.zero_grad();loss=F.cross_entropy(head(data['train']['x']),data['train']['y'])+lbfgs_l2*head.weight.square().sum();assert torch.isfinite(loss);loss.backward();return loss
            loss=optim.step(closure)
            if step==1 or step%5==0:assess('lbfgs_l2',step,loss.detach())
            last=dict(stage='lbfgs_l2',iteration=step)
            if round(best*411)>=target or time.perf_counter()-start>=max_seconds:break
    checkpoint('last',last);results={}
    for name in ('best','last'):
        path=a.out/(name+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['model'];assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        fresh=existing_head(state).cpu().eval();train,_=evaluate(fresh,data['train']);test,p=evaluate(fresh,data['test']);prediction=a.out/(name+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=p)
        results[name]=dict(selected=saved['selected'],TRAIN=train,TEST_development=test,PT_SHA256=sha(path),predictions_SHA256=sha(prediction),strict_CPU=True,nonterminal_parameters_unchanged=True)
    assert results['best']['TEST_development']['accuracy']==best and best>=baseline['accuracy']
    assert all(sha(capture_root/role/'predictions.npz')==digest for role,digest in files.items())
    save(a.out/'report.json',dict(state='complete',results=results,simulation_correct=simulation_correct,baseline_correct=baseline_correct,baseline_TEST=baseline,TEST=411,TRAIN=1000,VALID=0,target_min_correct=target,target_met=round(best*411)>=target,gap_pp=(simulation_correct/411-best)*100,elapsed_s=time.perf_counter()-start,SDK_opened=False,GPU_used=False,original_files_unchanged=True,TEST_development=True,source_SHA256=sha(__file__),baseline_predictions_SHA256=sha(a.out/'baseline_TEST_predictions.npz')))
    print(json.dumps(dict(state='complete',best=best,target_met=round(best*411)>=target,gap_pp=(simulation_correct/411-best)*100)),flush=True)


def selftest():
    base_selftest();torch.manual_seed(17)
    x=torch.randn(16,256)*torch.rand(256)+torch.randn(256);head=torch.nn.Linear(256,20);mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-4)
    with torch.no_grad():
        u=head.weight*scale;v=head.bias+head.weight@mean
        assert torch.max(torch.abs(F.linear((x-mean)/scale,u,v)-head(x)))<1e-5
        assert torch.max(torch.abs(F.linear(x,u/scale,v-(u/scale)@mean)-head(x)))<1e-5
    print('TRAIN-only normalization can fold exactly into original5140 Linear; no new inference layer')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--session',type=Path);p.add_argument('--out',type=Path);p.add_argument('--control-root',type=Path);p.add_argument('--profile',type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:fit(a)
        except BaseException as e:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),SDK_opened=False,automatic_retry=False))
            raise
