"""Conditional same5140-head fit after completed optical robustness, no new graph."""
import argparse,copy,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_rich_relation as relation
import caltech20_rich_existing_head_TESTreview as earlier_heads
import caltech20_rich_robust as robust
import caltech20_rich_robust_precise as precise
import audit_caltech20_rich_pair50_alpha as mathcheck
HERE,ROOT,s,engine=robust.HERE,robust.ROOT,robust.s,robust.engine
base,kd,resources=robust.base,robust.kd,robust.resources
controls=robust.controls
FILES=('caltech20_rich_robust_head.py','caltech20_rich_robust_head_profile.json','check_caltech20_rich_robust_head.py','audit_caltech20_rich_robust_head.py','CALTECH20_RICH_ROBUST_HEAD_PROTOCOL_20261018.md')

def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_robust_head_cpu_20261018'

def sources():
    z={**relation.sources(),**earlier_heads.sources()}
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)

def claim(p,phase):s.save(p,dict(identity(),phase=phase,pid=os.getpid(),automatic_retry=False),exclusive=True)

def load_np(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def core_digest(state):return s.digest({n:v for n,v in state.items() if not n.startswith('linear.')})

def ready():
    choices={}
    for mod,rc in [(robust,1),(precise,0),(relation,0)]:
        ex=s.read(mod.OUT/'campaign_exit.json');started=s.read(mod.OUT.with_name(mod.OUT.name+'.started.json'))
        assert ex['actual_exit'] and ex['returncode']==rc and not Path('/proc',str(ex['pid']),'cmdline').exists() and not Path('/proc',str(started['supervisor_pid']),'cmdline').exists()
    for depth in (2,4,6):
        pool=[(precise,'L6_robust_precise')] if depth==6 else [(robust,f'L{depth}_robust'),(relation,f'L{depth}_relation')]
        candidates=[]
        for mod,key in pool:
            arm=mod.OUT/key;f=s.read(arm/'fit_result.json');h=s.read(arm/'history.json');ex=s.read(mod.OUT/(key+'.producer_exit.json'));rec=s.read(arm/'audit_completed.json')
            assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
            assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(arm/'science/audit.json') and s.read(arm/'science/audit.json')['passed']
            meta=s.read(arm/'metadata.json');assert meta['sources']==mod.sources();engine.verify_sources(meta['sources'],meta['git_commit'])
            assert f['last_sha256']==s.sha(arm/'last.pt') and len(h)==10 and h[-1]['tune_epoch']==10
            candidates.append(dict(module=mod.__name__,key=key,arm=str(arm),last_sha256=f['last_sha256'],ema_digest=h[-1]['ema_digest'],train=h[-1]['train'],val=h[-1]['val'],test=h[-1]['test'],audit_sha256=rec['audit_sha256']))
        choices[str(depth)]=max(candidates,key=lambda q:(q['val']['accuracy'],-q['val']['balanced_nll']))
    return choices

def get_parent(depth):
    choice=ready()[str(depth)];mod={m.__name__:m for m in (robust,precise,relation)}[choice['module']];arm=Path(choice['arm']);pt=torch.load(arm/'last.pt',map_location='cpu',weights_only=False);assert s.digest(pt['ema'])==choice['ema_digest']
    return choice,mod,pt

def theta(state):return np.ascontiguousarray(np.r_[state['linear.weight'].double().numpy().T,state['linear.bias'].double().numpy()[None]])

def probabilities(state,features):
    z=np.c_[features.astype(np.float64),np.ones(len(features))]@theta(state);p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)

def targets():return relation.kd.teacher_targets()

def objective(t,X,y,q,valid,anchor,lam):
    cfg=config();z=X@t;soft=F.one_hot(y,20).double()*(1-cfg['label_smoothing'])+cfg['label_smoothing']/20
    counts=torch.tensor(robust.base.manifest()['supports']['train'],dtype=torch.float64);weights=counts.sum()/(20*counts)
    ce=-(soft*weights[None]*torch.log_softmax(z,1)).sum(1).mean()
    kl=cfg['temperature']**2*(valid*(q*(q.clamp_min(1e-12).log()-torch.log_softmax(z/cfg['temperature'],1))).sum(1)).mean()
    penalty=lam*(t-anchor).square().sum()/(2*len(X))
    return .98*ce+.02*kl+penalty,(ce,kl,penalty)

def extract(depth,gpu):
    cfg=config();out=OUT/f'L{depth}_rho03';assert not out.exists() and os.environ['CUDA_VISIBLE_DEVICES']==gpu and gpu in robust.resources.GPU.values();robust.resources.slots();claim(out.with_name(out.name+'.extract_claim.json'),'frozen_postregularization_TRAIN_VAL_features');out.mkdir()
    s.gpu_budget.ALLOWED.update(robust.resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    choice,mod,pt=get_parent(depth);model=mod.RobustMoE(depth).cuda().eval();model.load_state_dict(pt['ema'],strict=True);before=s.digest(model.state_dict())
    train=robust.base.load('train');idx=torch.arange(4)
    with torch.no_grad():
        x=robust.base.encode(train[0][idx].cuda(),None,100);p=torch.cat([model(x[i:i+1])[0] for i in range(4)]);c=model(x)[0];err=float((p-c).abs().max());assert err<=3e-6
    metrics={}
    for role,data in [('train',train),('val',robust.base.load('val'))]:
        met,pred=robust.base.evaluate(model,data);robust.base.save_np(out/(role+'_features.npz'),pred);mathcheck.agree(met,choice[role]);metrics[role]=met
    test=load_np(Path(choice['arm'])/'test_epoch10.npz');assert np.max(np.abs(probabilities(pt['ema'],test['features'])-test['probabilities']))<=3e-6
    robust.base.save_np(out/'test_features_reused.npz',test);assert s.digest(model.state_dict())==before==choice['ema_digest']
    s.save(out/'metadata.json',dict(identity(),pid=os.getpid(),depth=depth,gpu_uuid=gpu,policy=policy,parent=choice,parent_core_digest=core_digest(pt['ema']),parent_model_digest=before,config=cfg,clean_micro1vs4_probability_maxabs=err,TEST_images_read=0,new_optical_TEST_forwards=0,optimizer_steps=0),exclusive=True)
    s.save(out/'extract_result.json',dict(passed=True,metrics=metrics,features_sha256={r:s.sha(out/(r+'_features.npz')) for r in ('train','val')},test_features_sha256=s.sha(out/'test_features_reused.npz'),model_unchanged=True),exclusive=True)

def fit(depth):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(4);out=OUT/f'L{depth}_rho03';assert not (out/'fit_result.json').exists();claim(out/'fit_claim.json','same5140_clean_feature_conditional_fit')
    ex=s.read(out/'extract_exit.json');assert ex['returncode']==0 and ex['actual_exit'];meta=s.read(out/'metadata.json');choice,mod,pt=get_parent(depth);assert meta['parent']==choice
    train,val=load_np(out/'train_features.npz'),load_np(out/'val_features.npz');q,valid=targets();assert train['ids'].tolist()==robust.base.manifest()['split_ids']['train']
    anchor=theta(pt['ema']);parameter=torch.nn.Parameter(torch.tensor(anchor).contiguous());X=torch.tensor(np.c_[train['features'].astype(np.float64),np.ones(len(train['labels']))]);y=torch.tensor(train['labels']).long();q=torch.tensor(q).double();valid=torch.tensor(valid).double();a=torch.tensor(anchor);lam=config()['anchor_by_depth'][str(depth)]
    initial,_=objective(parameter,X,y,q,valid,a,lam);gradient=torch.autograd.grad(initial,parameter)[0];initial_grad=float(gradient.abs().max());trace=[]
    optimizer=torch.optim.LBFGS([parameter],lr=1.,max_iter=200,max_eval=300,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
    def closure():
        optimizer.zero_grad();loss,parts=objective(parameter,X,y,q,valid,a,lam);loss.backward();assert torch.isfinite(loss) and torch.isfinite(parameter.grad).all();trace.append(dict(theta=parameter.detach().numpy().copy(),loss=float(loss.detach()),grad=float(parameter.grad.abs().max())));return loss
    if initial_grad>1e-5:optimizer.step(closure)
    final,_=objective(parameter,X,y,q,valid,a,lam);grad=torch.autograd.grad(final,parameter)[0];assert float(final)<=float(initial)+1e-9
    candidate={n:v.clone() for n,v in pt['ema'].items()};t=parameter.detach().numpy();candidate['linear.weight']=torch.tensor(t[:-1].T.astype(np.float32));candidate['linear.bias']=torch.tensor(t[-1].astype(np.float32));assert core_digest(candidate)==meta['parent_core_digest']
    metrics={}
    for role,z in [('train',train),('val',val)]:
        z['probabilities']=probabilities(candidate,z['features']);metrics[role]=mathcheck.metrics(z['labels'],z['probabilities']);robust.base.save_np(out/('candidate_'+role+'.npz'),z)
    model=mod.RobustMoE(depth);model.load_state_dict(candidate);engine.save_pt(out/'candidate.pt',dict(model=candidate,metrics=metrics,sources=sources(),config=config(),contract=model.contract(),epoch=80,parent=choice))
    engine.save_pt(out/'optimization.pt',dict(theta=parameter.detach(),anchor=a,final_gradient=grad.detach(),optimizer=optimizer.state_dict(),sources=sources()))
    robust.base.save_np(out/'closure_trace.npz',dict(theta=np.stack([t['theta'] for t in trace]) if trace else anchor[None],loss=np.array([t['loss'] for t in trace]) if trace else np.array([float(initial)])))
    state=optimizer.state.get(parameter,{});s.save(out/'fit_result.json',dict(passed=True,metrics=metrics,initial_gradient_inf=initial_grad,fit_needed=initial_grad>1e-5,initial_objective=float(initial),final_objective=float(final),final_gradient_inf=float(grad.abs().max()),actual_LBFGS_iterations=int(state.get('n_iter',0)),closure_evaluations=len(trace),candidate_sha256=s.sha(out/'candidate.pt'),parent_core_digest=meta['parent_core_digest'],candidate_core_digest=core_digest(candidate),new_optical_optimizer_steps=0,new_inference_parameters=0,TEST_images_read=0),exclusive=True)

def test(depth):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';out=OUT/f'L{depth}_rho03';rec=s.read(out/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(out/'science/audit.json');claim(out/'test_claim.json','once_samehead_cached_TEST_after_fit_science')
    candidate=torch.load(out/'candidate.pt',map_location='cpu',weights_only=False)['model'];choice,mod,parent=get_parent(depth);pred=load_np(out/'test_features_reused.npz');reused=s.digest(candidate)==s.digest(parent['ema'])
    if not reused:pred['probabilities']=probabilities(candidate,pred['features'])
    met=mathcheck.metrics(pred['labels'],pred['probabilities']);robust.base.save_np(out/'candidate_test.npz',pred)
    s.save(out/'test_result.json',dict(passed=True,metrics=met,model_digest=s.digest(candidate),test_reused=reused,new_cached_head_TEST_evaluations=0 if reused else 1,new_optical_TEST_forwards=0,TEST_images_read=0,test_used_for_selection=True,independent_generalization=False),exclusive=True)

def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists() and robust.resources.slots()['total']==0
    cpu=s.read(CPU/'checks.json');assert cpu['passed'] and cpu['sources']==sources();parents=ready();claim(OUT.with_name(OUT.name+'.claim.json'),'postrobustness_samehead_three_depths');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),parents=parents,config=config(),frozen_controls=robust.controls()),exclusive=True)
    jobs=[]
    for depth,index in [(6,0),(4,1),(2,2)]:
        gpu=robust.resources.GPU[index];log=(OUT/f'L{depth}.extract.log').open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','extract','--depth',str(depth),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append((depth,p,log))
    for depth,p,log in jobs:
        rc=p.wait();log.close();s.save(OUT/f'L{depth}_rho03/extract_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Preserve failure/no retry'
    jobs=[]
    for depth in (2,4,6):
        log=(OUT/f'L{depth}.fit.log').open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--depth',str(depth)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);jobs.append((depth,p,log))
    for depth,p,log in jobs:
        rc=p.wait();log.close();out=OUT/f'L{depth}_rho03';s.save(out/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0;claim(out/'audit_claim.json','independent_convex_objective_gradient_core_audit')
        with (out/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--depth',str(depth),'--phase','fit'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert arc==0 and s.read(out/'science/audit.json')['passed'];s.save(out/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(out/'science/audit.json')),exclusive=True)
        with (out/'test.log').open('x') as log:p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','test','--depth',str(depth)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
        s.save(out/'test_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0;claim(out/'test_audit_claim.json','independent_saved_TEST_linear_math')
        with (out/'test_audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--depth',str(depth),'--phase','test'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert arc==0 and s.read(out/'test_science/audit.json')['passed'];s.save(out/'test_audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(out/'test_science/audit.json')),exclusive=True)
    s.save(OUT/'status.json',dict(state='postrobustness_samehead_complete',active=[],pending=[],failures=[],frozen_controls=robust.controls()),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['extract','fit','test','campaign'],required=True);p.add_argument('--depth',type=int);p.add_argument('--gpu');a=p.parse_args()
    try:campaign() if a.phase=='campaign' else extract(a.depth,a.gpu) if a.phase=='extract' else globals()[a.phase](a.depth)
    except BaseException as e:
        out=OUT/f'L{a.depth}_rho03' if a.depth else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
