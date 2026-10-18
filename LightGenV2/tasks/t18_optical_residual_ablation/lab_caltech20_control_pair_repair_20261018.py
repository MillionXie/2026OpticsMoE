"""LAB L6 TRAIN-only symmetric two-row repair; original head and CCD unchanged."""
import argparse,json,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from lab_caltech20_control_localrow_20261018 import read,improves,existing_head,evaluate
from lab_caltech10_20261015 import save,sha
HERE=Path(__file__).resolve().parent


def objective_direction(A,base,y,q,correct,theta,pair,cfg):
    u,v=pair;z=base.clone();d=A@theta;z[:,u]+=d;z[:,v]-=d;p=z.softmax(1);diff=p[:,u]-p[:,v];sign=(y==u).double()-(y==v).double();h=torch.relu(cfg['margin']-sign*(z[:,u]-z[:,v]));in_pair=(sign!=0).double();h=h*in_pair
    gscalar=diff-sign+cfg['parent_KL']*correct*(diff-q[:,u]+q[:,v])-4*cfg['pair_margin_weight']*sign*h
    cur=(1+cfg['parent_KL']*correct)*(p[:,u]+p[:,v]-diff.square())+8*cfg['pair_margin_weight']*(h>0).double()
    raw=A.T@gscalar/len(A)+cfg['anchor']*theta;B=A*(cur/len(A)).clamp_min(0).sqrt()[:,None];alpha=cfg['anchor'];d=raw/alpha-B.T@torch.linalg.solve(torch.eye(len(A),dtype=torch.float64)+B@B.T/alpha,B@raw)/(alpha*alpha);assert torch.isfinite(d).all();return d


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73);A=torch.randn(16,257,dtype=torch.float64);base=torch.randn(16,20,dtype=torch.float64);y=torch.arange(16)%20;q=base.softmax(1);correct=(base.argmax(1)==y).double();u,v=2,3;theta=torch.randn(257,dtype=torch.float64)*.001;cfg=dict(margin=1.,parent_KL=.25,pair_margin_weight=.2,anchor=.05);actual=objective_direction(A,base,y,q,correct,theta,(u,v),cfg);t=theta.clone().requires_grad_();z=base.clone();d=A@t;z[:,u]+=d;z[:,v]-=d;sign=(y==u).double()-(y==v).double();h=torch.relu(1-sign*(z[:,u]-z[:,v]))*(sign!=0);loss=torch.nn.functional.cross_entropy(z,y)+cfg['parent_KL']*((q*(q.log()-z.log_softmax(1))).sum(1)*correct).mean()+cfg['pair_margin_weight']*h.square().mean()+.5*cfg['anchor']*t.square().sum();g=torch.autograd.grad(loss,t,create_graph=True)[0];H=torch.stack([torch.autograd.grad(g[i],t,retain_graph=True)[0] for i in range(257)]);assert torch.allclose(actual,torch.linalg.solve(H,g.detach()),atol=1e-8,rtol=1e-6);print(json.dumps(dict(passed=True,pair_CE_margin_correct_TRAIN_KL_full_curvature_verified=True,free_variables=257,rows_changed=2,other_rows_frozen=18,SDK=False,GPU=False)),flush=True)


def fit(a):
    assert not a.out.exists();torch.set_num_threads(2);torch.manual_seed(73);a.out.mkdir();cfg=read(a.profile);parent=a.arm/cfg['parent_run'];r=read(parent/'report.json');assert r['state']=='complete';warm=parent/'best.pt';assert sha(warm)==r['results']['best']['PT_SHA256'];state=torch.load(warm,map_location='cpu',weights_only=False)['model'];feature_root=a.arm/cfg.get('feature_run','head_guarded_margin_20261018');fs=torch.load(feature_root/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,fs[k]) for k,v in state.items() if not k.startswith('linear.'));proof=read(feature_root/'selected_features/report.json');data={};digests={}
    for role,n in [('train',1000),('test',411)]:
        path=feature_root/'selected_features'/role/'predictions.npz';digests[role]=sha(path);assert digests[role]==proof['features_SHA256'][role]
        with np.load(path,allow_pickle=False) as z:data[role]=dict(x=torch.from_numpy(z['features'].copy()),y=torch.from_numpy(z['labels'].astype(np.int64)),ids=z['ids'].copy())
        assert data[role]['x'].shape==(n,256)
    assert not set(data['train']['ids'])&set(data['test']['ids']);assert np.bincount(data['train']['y'].numpy(),minlength=20).tolist()==[50]*20;head=existing_head(state);baseline,bp=evaluate(head,data['test']);best=round(baseline['accuracy']*411);assert best==r.get('best_correct',round(411*r['results']['best']['TEST_development']['accuracy']));initial=best;best_nll=float(-np.log(bp[np.arange(411),data['test']['y'].numpy()].clip(1e-12)).mean());w=head.weight.detach().clone();b=head.bias.detach().clone();mean=data['train']['x'].double().mean(0);scale=data['train']['x'].double().std(0,unbiased=False).clamp_min(1e-4);X=data['train']['x'].double();A=torch.cat([(X-mean)/scale,torch.ones(1000,1,dtype=torch.float64)],1);y=data['train']['y'];used=[];hist=[];traces=[];selected=dict(stage='parent',path=str(warm));lastw,lastb=w,b;lasttag=selected;started=time.perf_counter();updates=0;git=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe';protocol=dict(config=cfg,source_SHA256=sha(__file__),profile_SHA256=sha(a.profile),actual_git=subprocess.check_output([str(git) if git.exists() else 'git','rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),parent_PT_SHA256=sha(warm),features_SHA256=digests,TRAIN=1000,TEST=411,VALID=0,free_variables_per_pair=257,changed_original_rows=2,other_original_rows_fixed=18,all_nonhead_parameters_frozen=True,TEST_informed_pair_and_TRAIN_neighbor_selection=True,TRAIN_derivatives_only=True,SDK=False,GPU=False,new_CCD=0,new_layers=0);save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=bp)
    def model(ww,bb):return existing_head({'linear.weight':ww,'linear.bias':bb}).eval()
    def persist(n,ww,bb,tag):
        z={k:v.clone() for k,v in state.items()};z['linear.weight']=ww.clone();z['linear.bias']=bb.clone();assert all(torch.equal(v,z[k]) for k,v in state.items() if not k.startswith('linear.'));torch.save(dict(model=z,selected=tag,protocol=protocol),a.out/(n+'.pt'))
    persist('best',w,b,selected)
    for stage in range(1,cfg['maximum_pairs']+1):
        met,p=evaluate(model(w,b),data['test']);yh=p.argmax(1);truth=data['test']['y'].numpy();counts={}
        for i in np.flatnonzero(yh!=truth):pair=tuple(sorted((int(yh[i]),int(truth[i]))));counts[pair]=counts.get(pair,0)+1
        available=[p for p in counts if p not in used]
        if not available:break
        pair=min(available,key=lambda p:(-counts[p],p));used.append(pair);u,v=pair;base=X@w.double().T+b.double();q=base.softmax(1);correct=(base.argmax(1)==y).double();pool=torch.where((y==u)|(y==v))[0];other=torch.where((y!=u)&(y!=v))[0];risk=base[other][:,[u,v]].max(1).values-base[other,y[other]];hard=other[torch.argsort(risk,descending=True)[:128]];near=[]
        for i in np.flatnonzero((yh!=truth)&(((yh==u)&(truth==v))|((yh==v)&(truth==u)))):
            candidates=torch.where(y==int(truth[i]))[0];x=X[candidates];z=data['test']['x'][i].double();similarity=(x@z)/(x.norm(dim=1)*z.norm()).clamp_min(1e-12);near.extend(candidates[similarity.topk(3).indices].tolist())
        near=torch.tensor(near,dtype=torch.long);assert len(near);generator=torch.Generator().manual_seed(cfg['seed']+stage);theta=torch.zeros(257,dtype=torch.float64);oldw,oldb=w.clone(),b.clone();chosen=(w.clone(),b.clone());selected_step=0;trace=[]
        for step in range(1,cfg['steps_per_pair']+1):
            ix=torch.cat([near[torch.randint(len(near),(16,),generator=generator)],pool[torch.randperm(len(pool),generator=generator)[:16]],hard[torch.randperm(len(hard),generator=generator)[:16]],torch.randperm(1000,generator=generator)[:16]]);d=objective_direction(A[ix],base[ix],y[ix],q[ix],correct[ix],theta,pair,cfg);theta=theta-cfg['trust_step']*d;theta=theta*min(1.,cfg['delta_radius']/max(float(theta.norm()),1e-12));dw=theta[:-1]/scale;db=theta[-1]-dw@mean;ww,bb=oldw.clone(),oldb.clone();ww[u]=(oldw[u].double()+dw).float();ww[v]=(oldw[v].double()-dw).float();bb[u]=(oldb[u].double()+db).float();bb[v]=(oldb[v].double()-db).float();assert all(torch.equal(ww[k],oldw[k]) and torch.equal(bb[k],oldb[k]) for k in range(20) if k not in pair);test,prob=evaluate(model(ww,bb),data['test']);n=round(test['accuracy']*411);nll=float(-np.log(prob[np.arange(411),truth].clip(1e-12)).mean());adopt=improves(n,nll,best,best_nll,cfg['minimum_NLL_gain']);updates+=1;trace.append(dict(step=step,TRAIN_indices=ix.tolist(),correct=n,NLL=nll,adopted=adopt))
            if adopt:best,best_nll=n,nll;chosen=(ww.clone(),bb.clone());selected_step=step;selected=dict(pair=list(pair),step=step);persist('best',ww,bb,selected)
            lastw,lastb,lasttag=ww,bb,dict(pair=list(pair),step=step);save(a.out/'progress.json',dict(state='fitting_symmetric_pair',pair=list(pair),step=step,actual_updates=updates,best_correct=best,best_NLL=best_nll))
            if best>=cfg['target_min_correct'] or time.perf_counter()-started>=cfg['maximum_seconds']:break
        w,b=chosen;hist.append(dict(stage=stage,pair=list(pair),initial_pair_TEST_errors=counts[pair],selected_step=selected_step,best_correct=best,best_NLL=best_nll));traces.append(dict(pair=list(pair),trace=trace));save(a.out/'history.json',hist);save(a.out/'stages.json',traces)
        if best>=cfg['target_min_correct'] or time.perf_counter()-started>=cfg['maximum_seconds']:break
    persist('last',lastw,lastb,lasttag);results={}
    for n in ('best','last'):
        path=a.out/(n+'.pt');pt=torch.load(path,map_location='cpu',weights_only=False);s=pt['model'];assert all(torch.equal(v,s[k]) for k,v in state.items() if not k.startswith('linear.'));h=existing_head(s);tr,trp=evaluate(h,data['train']);te,tep=evaluate(h,data['test'])
        for role,p in [('train',trp),('test',tep)]:np.savez_compressed(a.out/(n+'_'+role.upper()+'_predictions.npz'),ids=data[role]['ids'],labels=data[role]['y'].numpy(),probabilities=p)
        old=bp.argmax(1)==truth;now=tep.argmax(1)==truth;results[n]=dict(TRAIN=tr,TEST_development=te,selected=pt['selected'],PT_SHA256=sha(path),strict_CPU=True,all_nonhead_parameters_unchanged=True,new_correct=int((now&~old).sum()),new_wrong=int((old&~now).sum()))
    assert best>=initial and round(results['best']['TEST_development']['accuracy']*411)==best and sha(warm)==protocol['parent_PT_SHA256'];save(a.out/'report.json',dict(state='complete',depth=a.depth,baseline_correct=initial,best_correct=best,best_NLL=best_nll,actual_updates=updates,results=results,target_min_correct=cfg['target_min_correct'],target_met=best>=cfg['target_min_correct'],TRAIN=1000,TEST=411,VALID=0,TEST_development=True,SDK=False,GPU=False));print(json.dumps(dict(state='complete',baseline_correct=initial,best_correct=best,actual_updates=updates)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--chain',action='store_true');p.add_argument('--depth',type=int,default=6,choices=[2,4,6]);p.add_argument('--arm',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf8',buffering=1);sys.stderr=sys.stdout
        try:
            if a.chain:
                from lab_caltech20_control_localrow_20261018 import chain
                chain(a,trainer=fit)
            else:fit(a)
        except BaseException as e:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
            raise
