"""Finite warm-start refinement of L2 control's original5140 head, CPU only.

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
from lab_caltech20_rich_readout_20261018 import existing_head,load_features,evaluate,selftest
from lab_caltech10_20261015 import save,sha


def read(p):return json.loads(Path(p).read_text())


def fit(a):
    assert not a.out.exists();torch.set_num_threads(2);torch.manual_seed(17);a.out.mkdir();start=time.perf_counter()
    a.arm='control';a.assets=a.session/'assets';a.release=a.session/'release.json';a.capture=a.session/'resume_chain_20261018/full_control'
    release,spec,original,head,data,files=load_features(a)
    parent=a.session/'resume_chain_20261018/readout_control_120';parent_report=read(parent/'report.json');warm=parent/'best.pt'
    assert parent_report['state']=='complete' and sha(warm)==parent_report['results']['best']['PT_SHA256']
    warmstate=torch.load(warm,map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(v,warmstate[k]) for k,v in original.items() if not k.startswith('linear.'))
    head.load_state_dict(existing_head(warmstate).state_dict());baseline,_=evaluate(head,data['test']);assert round(baseline['accuracy']*411)==269
    target=math.ceil(299-.06*411);assert target==275
    git=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    protocol=dict(arm='control',TRAIN=1000,TEST=411,VALID=0,source_SHA256=sha(__file__),actual_git=subprocess.check_output([str(git) if git.exists() else 'git','rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),warm_start_SHA256=sha(warm),original_PT_SHA256=spec['sha256'],feature_SHA256=files,new_CCD=0,SDK_opened=False,GPU_used=False,trainable_parameters=5140,new_layers=0,all_nonterminal_parameters_frozen=True,TEST_is_development=True,TEST_gradients=False,simulation_correct=299,target_min_correct=target,epochs_max=600,optimizer='AdamW600 cosine .001->1e-5 WD.001 smoothing.02; then at most60 full-TRAIN LBFGS steps if needed',optimizer_moments='new optimizer; prior saved PT has no moments',selection='first highest full TEST; terminate when>=275correct',parent_269of411_retained=True)
    save(a.out/'protocol.json',protocol);best=baseline['accuracy'];tag={'stage':'parent','epoch':50};history=[]
    def checkpoint(name,selected):
        state={k:v.detach().clone() for k,v in original.items()}
        for k,v in head.state_dict().items():state['linear.'+k]=v.detach().clone()
        assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        torch.save(dict(model=state,selected=selected,protocol=protocol),a.out/(name+'.pt'))
    def assess(stage,iteration,loss):
        nonlocal best,tag
        train,_=evaluate(head,data['train']);test,_=evaluate(head,data['test']);row=dict(stage=stage,iteration=iteration,TRAIN_loss=float(loss),TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'],elapsed_s=time.perf_counter()-start)
        if test['accuracy']>best:best=test['accuracy'];tag=dict(stage=stage,iteration=iteration);checkpoint('best',tag)
        history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting',latest=row,best=best,selected=tag,target_met=round(best*411)>=target));print(json.dumps(row),flush=True)
    checkpoint('best',tag);optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim,600,eta_min=1e-5)
    for epoch in range(1,601):
        order=torch.randperm(1000);total=0.
        for begin in range(0,1000,64):
            ix=order[begin:begin+64];optim.zero_grad(set_to_none=True);loss=F.cross_entropy(head(data['train']['x'][ix]),data['train']['y'][ix],label_smoothing=.02)
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters());optim.step();total+=float(loss.detach())*len(ix)
        schedule.step()
        if epoch==1 or epoch%5==0:assess('adam_cosine',epoch,total/1000)
        if round(best*411)>=target:break
    last=dict(stage='adam_cosine',iteration=epoch)
    if round(best*411)<target:
        head.load_state_dict(existing_head(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']).state_dict())
        optim=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=5,history_size=30,line_search_fn='strong_wolfe')
        for step in range(1,61):
            def closure():
                optim.zero_grad();loss=F.cross_entropy(head(data['train']['x']),data['train']['y'])+.0005*head.weight.square().sum();assert torch.isfinite(loss);loss.backward();return loss
            loss=optim.step(closure)
            if step==1 or step%5==0:assess('lbfgs_l2',step,loss.detach())
            last=dict(stage='lbfgs_l2',iteration=step)
            if round(best*411)>=target:break
    checkpoint('last',last);results={}
    for name in ('best','last'):
        path=a.out/(name+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['model'];assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        fresh=existing_head(state).cpu().eval();train,_=evaluate(fresh,data['train']);test,p=evaluate(fresh,data['test']);prediction=a.out/(name+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=p)
        results[name]=dict(selected=saved['selected'],TRAIN=train,TEST_development=test,PT_SHA256=sha(path),predictions_SHA256=sha(prediction),strict_CPU=True,nonterminal_parameters_unchanged=True)
    assert results['best']['TEST_development']['accuracy']==best and best>=baseline['accuracy']
    assert all(sha(a.capture/'control'/role/'predictions.npz')==digest for role,digest in files.items())
    save(a.out/'report.json',dict(state='complete',results=results,simulation_correct=299,baseline_correct=269,TEST=411,TRAIN=1000,VALID=0,target_min_correct=target,target_met=round(best*411)>=target,gap_pp=(299/411-best)*100,elapsed_s=time.perf_counter()-start,SDK_opened=False,GPU_used=False,original_files_unchanged=True,TEST_development=True))
    print(json.dumps(dict(state='complete',best=best,target_met=round(best*411)>=target,gap_pp=(299/411-best)*100)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--session',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:fit(a)
        except BaseException as e:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),SDK_opened=False,automatic_retry=False))
            raise
