"""Conditional same5140-head fit on completed alpha40 encoder families."""
import argparse,ast,inspect,os,subprocess,time,types
from pathlib import Path
import torch
import caltech20_rich_robust_head as kernel
import caltech20_rich_projected_alpha_contained as projected
import caltech20_rich_alpha40_fresh246 as sigmoid
for n,v in vars(kernel).items():
    if not n.startswith('__'):globals()[n]=v
FAMILY='sigmoid';PHASES={'sigmoid':sigmoid,'projected':projected}
FILES=('caltech20_rich_alpha40_samehead.py','caltech20_rich_alpha40_samehead_profile.json','check_caltech20_rich_alpha40_samehead.py','audit_caltech20_rich_alpha40_samehead.py','CALTECH20_RICH_ALPHA40_SAMEHEAD_PROTOCOL_20261018.md')
PROFILE=HERE/FILES[1]
def config():
    c=s.read(PROFILE);c.update(family=FAMILY);return c
ROOT_OUT=HERE/'runs/simulation'/s.read(PROFILE)['run'];CPU=HERE/'runs/smoke/caltech20_rich_alpha40_samehead_cpu_20261018';OUT=ROOT_OUT/FAMILY
def setup(family):
    global FAMILY,OUT
    assert family in PHASES;FAMILY=family;OUT=ROOT_OUT/family
def sources():
    z=projected.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','load_np','core_digest','theta','probabilities','targets','objective','fit','test'):
    f=getattr(kernel,name)
    if name=='fit':
        tree=ast.parse(inspect.getsource(f));changed=0
        for node in ast.walk(tree):
            if isinstance(node,ast.keyword) and node.arg=='epoch' and isinstance(node.value,ast.Constant) and node.value.value==80:node.value=ast.Constant(50);changed+=1
        assert changed==1;exec(compile(ast.fix_missing_locations(tree),__file__+':same_clean_head_fitted_at_encoder50','exec'),globals())
    else:globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def get_parent(depth):
    mod=PHASES[FAMILY];key=f'L{depth}_fresh' if FAMILY=='sigmoid' else f'L{depth}_projected';folder=mod.OUT/key
    f=s.read(folder/'fit_result.json');meta=s.read(folder/'metadata.json');ex=s.read(mod.OUT/(key+'.producer_exit.json'));rec=s.read(folder/'audit_completed.json')
    assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed']
    if FAMILY=='projected':assert s.read(folder/'science/audit.json')['curriculum_checks_sha256']==s.sha(folder/'science/curriculum_checks.json') and s.read(folder/'science/curriculum_checks.json')['passed']
    assert f['actual_new_epochs']==50 and f['last_sha256']==s.sha(folder/'last.pt') and meta['sources']==mod.sources();engine.verify_sources(meta['sources'],meta['git_commit'])
    h=s.read(folder/'history.json');assert len(h)==50 and h[-1]['tune_epoch']==50 and 'test' in h[-1]
    pt=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);assert s.digest(pt['ema'])==h[-1]['ema_digest'] and pt['sources']==meta['sources']
    choice=dict(family=FAMILY,module=mod.__name__,key=key,arm=str(folder),last_sha256=f['last_sha256'],ema_digest=h[-1]['ema_digest'],train=h[-1]['train'],val=h[-1]['val'],test=h[-1]['test'],audit_sha256=rec['audit_sha256'],source_git=meta['git_commit'],gpu_uuid=meta['gpu_uuid'])
    return choice,types.SimpleNamespace(RobustMoE=mod.Alpha40MoE),pt
def extract(depth,gpu):
    choice,mod,pt=get_parent(depth);assert gpu==choice['gpu_uuid'],'Same physical GPU as encoder evaluation required';assert robust.resources.slots()['counts'][gpu]==1
    cfg=config();out=OUT/f'L{depth}_rho03';assert not out.exists() and os.environ['CUDA_VISIBLE_DEVICES']==gpu;claim(out.with_name(out.name+'.extract_claim.json'),'same_head_frozen_encoder50_TRAIN_VAL_only');out.mkdir()
    s.gpu_budget.ALLOWED.update(robust.resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    model=mod.RobustMoE(depth).cuda().eval().requires_grad_(False);model.load_state_dict(pt['ema'],strict=True);before=s.digest(model.state_dict());data=robust.base.load('train')
    with torch.no_grad():
        x=robust.base.encode(data[0][:4].cuda(),None,100);p=torch.cat([model(x[i:i+1])[0] for i in range(4)]);q=model(x)[0];err=float((p-q).abs().max());assert err<=3e-6
    metrics={}
    for role,d in [('train',data),('val',robust.base.load('val'))]:
        met,pred=robust.base.evaluate(model,d);mathcheck.agree(met,choice[role]);robust.base.save_np(out/(role+'_features.npz'),pred);metrics[role]=met
    test=load_np(Path(choice['arm'])/'test_epoch50.npz');assert np.max(np.abs(probabilities(pt['ema'],test['features'])-test['probabilities']))<=3e-6;robust.base.save_np(out/'test_features_reused.npz',test)
    assert s.digest(model.state_dict())==before==choice['ema_digest'] and all(p.grad is None and not p.requires_grad for p in model.parameters())
    s.save(out/'metadata.json',dict(identity(),pid=os.getpid(),family=FAMILY,depth=depth,gpu_uuid=gpu,policy=policy,parent=choice,parent_core_digest=core_digest(pt['ema']),parent_model_digest=before,config=cfg,clean_micro1vs4_probability_maxabs=err,TEST_images_read=0,new_optical_TEST_forwards=0,optimizer_steps=0),exclusive=True)
    s.save(out/'extract_result.json',dict(passed=True,metrics=metrics,features_sha256={r:s.sha(out/(r+'_features.npz')) for r in ('train','val')},test_features_sha256=s.sha(out/'test_features_reused.npz'),model_unchanged=True),exclusive=True)
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['full_actual_namespaces_resolved'];identity();robust.controls()
def one(family,depth,gpu):
    setup(family);ready();folder=OUT/f'L{depth}_rho03';OUT.mkdir(exist_ok=True)
    for phase,receipt,device in [('extract','extract_exit.json',gpu),('fit','producer_exit.json','')]:
        with (OUT/f'L{depth}.{phase}.log').open('x') as log:
            p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase',phase,'--family',family,'--depth',str(depth),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
        assert folder.exists();s.save(folder/receipt,dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Preserve failure/no retry'
    for phase,receipt,sub in [('fit','audit_completed.json','science'),('test','test_audit_completed.json','test_science')]:
        if phase=='test':
            with (folder/'test.log').open('x') as log:
                p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','test','--family',family,'--depth',str(depth)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
            s.save(folder/'test_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0
        claim(folder/('audit_claim.json' if phase=='fit' else 'test_audit_claim.json'),'same_head_independent_'+phase)
        with (folder/(phase+'.audit.log')).open('x') as log:rc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--family',family,'--depth',str(depth),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0 and s.read(folder/sub/'audit.json')['passed'];s.save(folder/receipt,dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(folder/sub/'audit.json')),exclusive=True)
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not ROOT_OUT.exists();ready();claim(ROOT_OUT.with_name(ROOT_OUT.name+'.claim.json'),'conditional_six_same_head_fits_encoder50');ROOT_OUT.mkdir();s.save(ROOT_OUT/'metadata.json',dict(identity(),pid=os.getpid(),profile=s.read(PROFILE),frozen_controls=robust.controls()),exclusive=True)
    pending=[(family,d) for family in ('projected','sigmoid') for d in (4,6,2)];complete=[];started=time.monotonic()
    while pending:
        assert time.monotonic()-started<18000,'Bounded night queue protection';chosen=None
        # Avoid racing with projected controller's remaining launch queue.
        state=s.read(projected.OUT/'status.json') if (projected.OUT/'status.json').exists() else None
        if state and not state['pending']:
            for family,depth in pending:
                setup(family);mod=PHASES[family];key=f'L{depth}_fresh' if family=='sigmoid' else f'L{depth}_projected';folder=mod.OUT/key
                if not (folder/'audit_completed.json').exists():continue
                choice,_,_=get_parent(depth);gpu=choice['gpu_uuid']
                if robust.resources.slots()['counts'][gpu]==0:chosen=(family,depth,gpu);break
        if chosen:
            family,depth,gpu=chosen;one(family,depth,gpu);pending.remove((family,depth));complete.append(dict(family=family,depth=depth));setup('sigmoid')
        s.save(ROOT_OUT/'status.json',dict(state='same_head_waiting_or_running' if pending else 'all_same_head_cases_complete',pending=pending,complete=complete,failures=[]))
        if pending:time.sleep(3)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['extract','fit','test','campaign'],required=True);p.add_argument('--family',choices=list(PHASES),default='sigmoid');p.add_argument('--depth',type=int);p.add_argument('--gpu');a=p.parse_args();setup(a.family)
    try:campaign() if a.phase=='campaign' else extract(a.depth,a.gpu) if a.phase=='extract' else globals()[a.phase](a.depth)
    except BaseException as e:
        root=OUT/f'L{a.depth}_rho03' if a.depth else ROOT_OUT;mark=root/'failure.json' if root.exists() else root.with_name(root.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
