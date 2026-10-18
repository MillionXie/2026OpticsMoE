"""Finite, warm-started refinement of the original5140 head on unchanged CCDs.

Two justified optimizer comparisons: longer AdamW with cosine decay, then
full-TRAIN L-BFGS for the same convex linear classification problem. TEST is
used only for explicitly authorized development selection, never gradients.
No SDK, optical forward, GPU, additional layer, changed feature normalization,
new acquisition or update of nonterminal model tensors is permitted here.
"""
import argparse,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_rich_readout_20261018 import existing_head,load_features,evaluate
from lab_caltech10_20261015 import save,sha


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def unchanged(original,actual):
    return all(torch.equal(v,actual[k]) for k,v in original.items() if not k.startswith('linear.'))


def refine(a):
    cfg=read(a.profile);assert cfg['arm']=='residual' and cfg['trainable_parameters']==5140
    assert cfg['train_samples']==1000 and cfg['test_samples']==411 and cfg['valid_samples']==0
    assert not a.out.exists();a.out.mkdir(parents=True);started=time.perf_counter()
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
    a.arm='residual';a.assets=a.session/'assets';a.release=a.session/'release.json';a.capture=a.session/'resume_chain_20261018/capture_residual'
    release,spec,original,head,data,files=load_features(a)
    parent=a.session/'resume_chain_20261018/readout_residual_120';parent_report=read(parent/'report.json');warm=parent/'best.pt'
    assert parent_report['state']=='complete' and sha(warm)==parent_report['results']['best']['PT_SHA256']
    saved=torch.load(warm,map_location='cpu',weights_only=False);assert unchanged(original,saved['model'])
    head.load_state_dict(existing_head(saved['model']).state_dict(),strict=True)
    baseline,baseline_p=evaluate(head,data['test']);train0,_=evaluate(head,data['train'])
    assert baseline['accuracy']==parent_report['results']['best']['TEST_development']['accuracy']
    bundled=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_8um/.git-tools/MinGit_2_56_0/cmd/git.exe'
    if not bundled.exists():bundled=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    git=str(bundled) if bundled.exists() else shutil.which('git')
    protocol=dict(config=cfg,profile_sha256=sha(a.profile),source_sha256=sha(__file__),actual_git=subprocess.check_output([git,'rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),warm_start_PT_SHA256=sha(warm),original_PT_SHA256=spec['sha256'],release_sha256=sha(a.release),features_sha256=files,TEST_is_development_not_independent=True,TEST_gradients=False,VALID_samples=0,SDK_opened=False,GPU_used=False,new_network_layers=0,trainable_parameters=5140,optimizer_moments='New optimizers; previous run did not save moments. Weights warm-start from accepted best, not random initialization',nonterminal_parameters_frozen=True,original_best_is_retained=True)
    save(a.out/'protocol.json',protocol);history=[];best_score=baseline['accuracy'];best_tag=dict(stage='parent',epoch=110);global_step=0
    def checkpoint(kind,tag):
        state={k:v.detach().clone() for k,v in original.items()}
        for key,value in head.state_dict().items():state['linear.'+key]=value.detach().clone()
        assert unchanged(original,state)
        torch.save(dict(model=state,selected=tag,protocol=protocol,only_existing_linear_adapted=True),a.out/(kind+'.pt'))
    checkpoint('best',best_tag)
    def assess(stage,iteration,loss):
        nonlocal best_score,best_tag
        train,_=evaluate(head,data['train']);test,_=evaluate(head,data['test']);tag=dict(stage=stage,iteration=iteration)
        row=dict(**tag,TRAIN_loss=float(loss),TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'],elapsed_s=time.perf_counter()-started,optimizer_steps=global_step)
        if test['accuracy']>best_score:
            best_score=test['accuracy'];best_tag=tag;checkpoint('best',tag)
        history.append(row);save(a.out/'history.json',history)
        save(a.out/'progress.json',dict(state='refining',latest=row,best=best_tag,best_TEST_development_accuracy=best_score,target_min_correct=325,target_met=round(best_score*411)>=325))
        print(json.dumps(row),flush=True)
    assess('warm_start',0,F.cross_entropy(head(data['train']['x']),data['train']['y']).detach())
    adam,lbfgs=cfg['stages'];assert adam['epochs']==600 and lbfgs['steps']==120
    optimizer=torch.optim.AdamW(head.parameters(),lr=adam['lr'],weight_decay=adam['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=adam['epochs'],eta_min=adam['min_lr'])
    stage_summaries=[]
    for epoch in range(1,adam['epochs']+1):
        order=torch.randperm(1000);loss_total=0.
        for start in range(0,1000,adam['batch_size']):
            ix=order[start:start+adam['batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(head(data['train']['x'][ix]),data['train']['y'][ix]);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step();global_step+=1;loss_total+=float(loss.detach())*len(ix)
        scheduler.step()
        if epoch%adam['evaluate_every']==0:assess(adam['name'],epoch,loss_total/1000)
    stage_summaries.append(dict(stage=adam['name'],best=best_tag,best_TEST_development=best_score))
    # Same linear model, full-TRAIN objective, initialized from accepted global best.
    state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model'];head.load_state_dict(existing_head(state).state_dict())
    optimizer=torch.optim.LBFGS(head.parameters(),lr=lbfgs['lr'],max_iter=lbfgs['inner_iterations'],history_size=lbfgs['history_size'],line_search_fn='strong_wolfe')
    for step in range(1,lbfgs['steps']+1):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(head(data['train']['x']),data['train']['y'])+.5*lbfgs['l2_sum']*head.weight.square().sum()
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters());return loss
        loss=optimizer.step(closure);global_step+=1
        if step%lbfgs['evaluate_every']==0:assess(lbfgs['name'],step,loss.detach())
    checkpoint('last',dict(stage=lbfgs['name'],iteration=lbfgs['steps']))
    stage_summaries.append(dict(stage=lbfgs['name'],best=best_tag,best_TEST_development=best_score))
    results={}
    for kind in ('best','last'):
        path=a.out/(kind+'.pt');value=torch.load(path,map_location='cpu',weights_only=False);assert unchanged(original,value['model'])
        fresh=existing_head(value['model']).cpu().eval();train,_=evaluate(fresh,data['train']);test,probability=evaluate(fresh,data['test'])
        prediction=a.out/(kind+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=probability)
        results[kind]=dict(selected=value['selected'],TRAIN=train,TEST_development=test,PT_SHA256=sha(path),predictions_SHA256=sha(prediction),fresh_CPU_reload_verified=True,all_nonterminal_parameters_exactly_unchanged=True)
    assert results['best']['TEST_development']['accuracy']==best_score and best_score>=baseline['accuracy']
    assert all(sha(a.capture/'residual'/role/'predictions.npz')==digest for role,digest in files.items())
    report=dict(state='complete',arm='residual',baseline_TRAIN=train0,baseline_TEST_development=baseline,results=results,stage_summaries=stage_summaries,elapsed_s=time.perf_counter()-started,target_min_correct=325,target_met=round(best_score*411)>=325,simulation_correct=329,TEST_samples=411,simulation_gap_pp=(329/411-best_score)*100,SDK_opened=False,GPU_used=False,TRAIN_samples=1000,VALID_samples=0,TEST_is_development_not_independent=True,original_files_SHA256_unchanged=True,profile_sha256=sha(a.profile),source_sha256=sha(__file__))
    save(a.out/'report.json',report);print(json.dumps(dict(state='complete',baseline_TEST=baseline['accuracy'],best_TEST_development=best_score,selected=best_tag,target_met=report['target_met'],gap_pp=report['simulation_gap_pp'],elapsed_s=report['elapsed_s'])),flush=True)


def selftest():
    torch.set_num_threads(2);torch.manual_seed(17);layer=torch.nn.Linear(256,20)
    x=torch.randn(50,256);y=torch.arange(50)%20;before=F.cross_entropy(layer(x),y).detach()
    optimizer=torch.optim.LBFGS(layer.parameters(),lr=1.,max_iter=5,line_search_fn='strong_wolfe')
    def closure():optimizer.zero_grad();loss=F.cross_entropy(layer(x),y);loss.backward();return loss
    optimizer.step(closure);assert F.cross_entropy(layer(x),y).detach()<before
    assert sum(p.numel() for p in layer.parameters())==5140
    print('Existing5140 convex head LBFGS descent passed; no real data/SDK/GPU')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    for name in ('session','out','profile'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:refine(a)
        except BaseException as error:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),SDK_opened=False,automatic_retry=False))
            raise
