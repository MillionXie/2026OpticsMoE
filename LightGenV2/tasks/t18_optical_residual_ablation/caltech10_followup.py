"""One bounded residual regularization, then matched depths3/2, human20261015."""
import argparse,copy,inspect,math,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import caltech10_gap as parent
import caltech10_paired_test as baseline
import caltech10_experiment_resources as resources
s=parent.s;HERE=parent.HERE;ROOT=parent.ROOT;PROFILE=HERE/'caltech10_followup_profile.json'
FILES=('caltech10_followup.py','caltech10_followup_profile.json','audit_caltech10_followup.py','CALTECH10_FOLLOWUP_PROTOCOL_20261015.md')
RUN=s.read(PROFILE)['run'];CPU=HERE/'runs/smoke/caltech10_followup_state_depth_20261015';NUMERIC=HERE/'runs/smoke/caltech10_followup_gpu0_numeric_20261015'
CASES={'regularize_L4_rho03':(4,.3), 'L3_rho03':(3,.3),'L3_rho0':(3,0.),'L2_rho03':(2,.3),'L2_rho0':(2,0.)}
def sources():
    v=baseline.sources()
    for n in FILES:v[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return v
def parameter_count(depth):return 224*224*(1+4*math.ceil(depth/2))+478*478*(depth//2)
def tensor_count(depth):return 1+4*math.ceil(depth/2)+depth//2
def original():return HERE/'runs/simulation'/parent.RUN
def config(case):
    c=parent.config();c.update(maximum_gpu_children=4,cpu_gate=CPU.relative_to(ROOT).as_posix()+'/checks.json',numeric_gate=NUMERIC.relative_to(ROOT).as_posix()+'/checks.json',scope='Human fixed10; one residual regularization then matched depth3/2; validation selection, no test tuning')
    if case=='regularize_L4_rho03':c.update(maximum_epochs=200,maximum_seconds=1800,mixup_alpha=.4)
    return c
def parent_state():
    baseline.barrier(original());p=s.read(PROFILE);arm=original()/'L4_gap10_rho03';assert s.sha(arm/'last_checkpoint.pt')==p['residual_parent_last_sha256'] and s.sha(arm/'history.json')==p['residual_parent_history_sha256']
    ck=torch.load(arm/'last_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==ck['scheduler']['last_epoch']==150 and all(int(v['step'])==7200 for v in ck['optimizer']['state'].values())
    return arm,ck
def gate(depth,rho,data,cfg,device,micro,state=None):
    assert depth in (2,3,4) and micro==8;s.engine.seed(cfg['seed']);model=parent.SubsetMoE(depth,rho).to(device)
    if state is not None:model.load_state_dict(state,strict=True)
    initial=s.digest(model.state_dict());weights=(len(data[1])/(10*torch.bincount(data[1],minlength=10).float())).to(device);order,theta=s.engine.epoch_plan(1,len(data[1]),cfg);tail=len(data[1])%16;checks=[]
    groups=[('all10',np.asarray([int((data[1]==k).nonzero()[0]) for k in range(10)]),0),('macro16',order[:16],0),('tail9',order[-tail:],math.ceil(len(order)/16)-1)]
    for label,idx,b in groups:
        x,y,_=s.engine.mixed_batch(data,idx,theta,1,b,cfg,device);v=parent.probe.compare(parent.probe.compute(model,x,y,weights,cfg,1),parent.probe.compute(model,x,y,weights,cfg,micro));assert v['passed'];checks.append(dict(v,samples=label))
    norms={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert len(norms)==tensor_count(depth) and sum(p.numel() for p in model.parameters())==parameter_count(depth) and all(np.isfinite(v) and v>0 for v in norms.values())
    del model;torch.cuda.empty_cache();return dict(passed=True,depth=depth,rho=rho,checks=checks,phase_gradients=norms,initial_parameter_sha256=initial,parameters=parameter_count(depth),all_phase_gradients_finite_nonzero=True,test_evaluations=0)
def bind(case):
    parent.bind();cfg=config(case);e=s.engine;e.PROFILE=PROFILE;e.read=lambda p:copy.deepcopy(cfg) if Path(p)==PROFILE else s.read(p);e.sources=sources;e.ALLOWED=resources.GPU;e.fresh_gate=gate
    src=textwrap.dedent(inspect.getsource(e.fit));changes={'20*torch.bincount(data[1],minlength=20).float()':'10*torch.bincount(data[1],minlength=10).float()'}
    if case=='regularize_L4_rho03':
        changes.update({"for v in (old_cfg,new_cfg):v.pop('shared_spare_memory_gib')":"for v in (old_cfg,new_cfg):\n            for k in ('shared_spare_memory_gib','maximum_epochs','maximum_seconds','minimum_epochs','mixup_alpha','maximum_gpu_children','cpu_gate','numeric_gate','scope'):v.pop(k)","if offset+time.monotonic()-started>=cfg['maximum_seconds']":"if time.monotonic()-started>=cfg['maximum_seconds']",'old_memory_gib=3,new_memory_gib=2':"old_memory_gib=resume['config']['shared_spare_memory_gib'],new_memory_gib=cfg['shared_spare_memory_gib'],stage_additional_seconds=cfg['maximum_seconds']"})
    for old,new in changes.items():assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    ns=dict(e.__dict__);exec(compile(src,__file__+':bounded_fit','exec'),ns);return cfg,ns['fit']
def cpu(a):
    assert a.out==CPU and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';resources.slots();arm,ck=parent_state();m=parent.manifest(Path(parent.config()['data']));assert sum(m['supports']['train'])==761 and sum(m['supports']['val'])==157
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=src,git_commit=commit,automatic_retry=False),exclusive=True);torch.set_num_threads(4);checks=[]
    for depth in (3,2):
        hashes=[]
        for rho in (.3,0.):
            s.engine.seed(17);model=parent.SubsetMoE(depth,rho);assert sum(p.numel() for p in model.parameters())==parameter_count(depth) and len(list(model.parameters()))==tensor_count(depth);hashes.append(s.digest(model.state_dict()))
            for black in (True,False):
                raw=torch.zeros(1,3,150,150,dtype=torch.uint8) if black else torch.randint(0,256,(1,3,150,150),dtype=torch.uint8)
                with torch.no_grad():p,c=model(parent.encode(raw,None,100))
                assert p.shape==(1,10) and torch.isfinite(p).all() and torch.allclose(p.sum(1),torch.ones(1),atol=1e-7)
                if black:assert torch.allclose(p,torch.full_like(p,.1),atol=1e-8)
            checks.append(dict(depth=depth,rho=rho,parameters=parameter_count(depth),phase_tensors=tensor_count(depth),stages=model.stages,captures=depth+1,black10_normalized=True))
        assert hashes[0]==hashes[1]
    for case in CASES:bind(case)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,parent_epoch=150,parent_last_sha256=s.sha(arm/'last_checkpoint.pt'),raw_sha256=s.digest(ck['model']),ema_sha256=s.digest(ck['ema']),optimizer_steps=7200,scheduler_epoch=150,restored_lr=ck['optimizer']['param_groups'][0]['lr'],checks=checks,models_constructed=4,synthetic_forward_calls=8,real_images_read=0,test_arrays_read=0),exclusive=True)
def numeric(a):
    assert a.out==NUMERIC and not a.out.exists() and a.gpu==resources.GPU[0] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();resources.slots();s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);cfg,_=bind('regularize_L4_rho03');m=parent.manifest(Path(cfg['data']));data=parent.load_fit(Path(cfg['data']),'train',m);arm,ck=parent_state()
    s.save(a.out.with_name(a.out.name+'.claim.json'),dict(sources=sources(),automatic_retry=False),exclusive=True);checks=[gate(4,.3,data,cfg,'cuda',8,ck['model'])]
    for depth in (3,2):
        pair=[gate(depth,rho,data,config('L'+str(depth)+'_rho03'),'cuda',8) for rho in (.3,0.)];assert pair[0]['initial_parameter_sha256']==pair[1]['initial_parameter_sha256'];checks+=pair
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'checks.json',dict(passed=True,sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checks=checks,optimizer_steps=0,test_arrays_read=0),exclusive=True)
def prerequisites():
    for out in (CPU,NUMERIC):
        p=s.read(out/'checks.json');assert p['passed'] and p['sources']==sources();s.engine.verify_sources(p['sources'],p['git_commit'])
def fit(a):
    assert a.case in CASES and (a.depth,a.rho)==CASES[a.case] and a.gpu in resources.GPU.values();prerequisites();resources.slots();cfg,fn=bind(a.case);a.data=Path(cfg['data']);a.resume_parent=parent_state()[0] if a.case=='regularize_L4_rho03' else None;fn(a)
    f=s.read(a.out/'fit_result.json');f['scope']=cfg['scope'];s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',parent.eligibility(f,s.read(a.out/'history.json')));assert f['epochs_completed']==cfg['maximum_epochs'] and f['stop_reason']=='maximum_epochs'
def wait_slot(index):
    while True:
        live=resources.slots();gpu=resources.GPU[index];s.gpu_budget.ALLOWED.update(resources.GPU.values())
        if live['total']<4 and live['counts'][gpu]==0:
            try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);return gpu
            except AssertionError:pass
        time.sleep(3)
def launch_fit(root,case,index):
    depth,rho=CASES[case];gpu=wait_slot(index);out=root/case;assert not out.exists();log=(root/(case+'.log')).open('x');cmd=[sys.executable,str(HERE/'caltech10_followup.py'),'--phase','fit','--case',case,'--depth',str(depth),'--rho',str(rho),'--gpu',gpu,'--out',str(out)];child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
    return dict(case=case,depth=depth,rho=rho,pid=child.pid,gpu_uuid=gpu,process=child,log=log)
def finish_job(root,job):
    rc=job['process'].wait();job['log'].close();rec={k:v for k,v in job.items() if k not in ('process','log')};rec.update(returncode=rc,actual_exit=parent.base.exited(job['pid']));s.save(root/job['case']/'producer_exit.json',rec,exclusive=True);assert rc==0 and rec['actual_exit'],'Fit failure preserved; no retry';return rec
def audit_stage(root,stage,cases):
    out=root/(stage+'_science');assert not out.exists();s.save(root/(stage+'_audit_claim.json'),dict(pid=os.getpid(),cases=cases,sources=sources(),automatic_retry=False),exclusive=True)
    with (root/(stage+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech10_followup.py'),'--root',str(root),'--stage',stage,'--out',str(out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(out/'audit.json')['passed'],'Independent science failed: no retry';s.save(root/(stage+'_audit_completed.json'),dict(passed=True,audit_sha256=s.sha(out/'audit.json'),actual_audit_returncode=rc),exclusive=True)
def test(a):
    root=a.out.parent.parent;stage='regularize' if a.case.startswith('regularize') else a.case[:2];done=s.read(root/(stage+'_audit_completed.json'));assert done['passed'] and done['audit_sha256']==s.sha(root/(stage+'_science')/'audit.json');claim=s.read(a.out/'claim.json');assert claim['sources']==sources();s.engine.verify_sources(claim['sources'],claim['git_commit']);f=s.read(root/a.case/'fit_result.json');assert f['checkpoint_sha256']==claim['checkpoint_sha256']==s.sha(root/a.case/'best_checkpoint.pt')
    assert not (a.out/'test_started.json').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;resources.slots();s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);s.save(a.out/'test_started.json',dict(pid=os.getpid(),checkpoint_sha256=f['checkpoint_sha256'],maximum_tests=1,automatic_retry=False),exclusive=True)
    m=parent.manifest(Path(f['config']['data']))
    with np.load(f['config']['data'],allow_pickle=False) as z:x,y,ids=[z['test_'+n].copy() for n in ('images','labels','ids')]
    keep=np.isin(y,parent.INDICES);x=x[keep];y=parent.subset_labels(y);ids=ids[keep];assert ids.tolist()==m['split_ids']['test'] and np.bincount(y,minlength=10).tolist()==m['supports']['test']
    parent.bind();s.engine.seed(17);model=parent.SubsetMoE(f['depth'],f['rho']).cuda();ck=torch.load(root/a.case/'best_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==f['selected_epoch'];model.load_state_dict(ck['model'],strict=True);metric,rows=parent.evaluate(model,(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids),8);s.engine.csvwrite(a.out/'test_predictions.csv',rows)
    s.save(a.out/'result.json',dict(test=metric,sources=claim['sources'],git_commit=claim['git_commit'],checkpoint_sha256=f['checkpoint_sha256'],selected_epoch=f['selected_epoch'],depth=f['depth'],rho=f['rho'],test_evaluations=1,test_used_for_selection=False,classes=m['classes'],scope='Validation-informed fixed10 exploratory simulation, prior test observed'),exclusive=True)
def test_once(root,case,index):
    out=root/'once_tests'/case;out.mkdir(parents=True,exist_ok=False);f=s.read(root/case/'fit_result.json');src=sources();commit=s.read(root/'metadata.json')['git_commit'];s.engine.verify_sources(src,commit);s.save(out/'claim.json',dict(sources=src,git_commit=commit,checkpoint_sha256=f['checkpoint_sha256'],selection='firstminimum validation EMA balancedNLL frozen',automatic_retry=False,maximum_tests=1),exclusive=True);gpu=wait_slot(index)
    with (out/'test.log').open('x') as log:child=subprocess.Popen([sys.executable,str(HERE/'caltech10_followup.py'),'--phase','test','--case',case,'--out',str(out),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);rc=child.wait()
    s.save(out/'producer_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=parent.base.exited(child.pid)),exclusive=True);assert rc==0 and parent.base.exited(child.pid),'Test failed: preserve no retry'
    v=s.read(out/'result.json');m=parent.manifest(Path(f['config']['data']))
    with np.load(f['config']['data'],allow_pickle=False) as z:y=parent.subset_labels(z['test_labels'].copy())
    baseline.predictions(out/'test_predictions.csv',v['test'],m['split_ids']['test'],y);assert v['checkpoint_sha256']==s.sha(root/case/'best_checkpoint.pt') and v['sources']==src
    s.save(out/'saved_test_audit.json',dict(passed=True,result_sha256=s.sha(out/'result.json'),independent_ID_label_argmax_CM_support_macro_NLL=True,model_forward_calls=0,test_image_arrays_read=0,test_label_arrays_read=1,test_evaluations_total=1),exclusive=True);s.save(out/'completed.json',dict(passed=True,result_sha256=s.sha(out/'result.json'),saved_audit_sha256=s.sha(out/'saved_test_audit.json')),exclusive=True);return v
def campaign(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==HERE/'runs/simulation'/RUN and not a.out.exists();prerequisites();baseline.barrier(original());base_test=original()/s.read(baseline.PROFILE)['output'];completed=s.read(base_test/'completed.json');assert completed['passed'] and completed['paired_result_sha256']==s.sha(base_test/'paired_result.json')
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'claim.json',dict(pid=os.getpid(),sources=src,git_commit=commit,profile=s.read(PROFILE),automatic_retry=False),exclusive=True);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,command=sys.argv,profile=s.read(PROFILE)),exclusive=True)
    s.save(a.out/'status.json',dict(state='regularize_running',test_used_for_selection=False));job=launch_fit(a.out,'regularize_L4_rho03',4);s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in job.items() if k not in ('process','log')}],pending=['L3_rho03','L3_rho0','L2_rho03','L2_rho0']));finish_job(a.out,job);audit_stage(a.out,'regularize',['regularize_L4_rho03'])
    oldck=torch.load(original()/'L4_gap10_rho03/best_checkpoint.pt',map_location='cpu',weights_only=False);newck=torch.load(a.out/'regularize_L4_rho03/best_checkpoint.pt',map_location='cpu',weights_only=False)
    if s.digest(oldck['model'])==s.digest(newck['model']):s.save(a.out/'regularize_test_reused.json',dict(reason='Selected weights unchanged; no second forward',old_result_sha256=s.sha(base_test/'L4_gap10_rho03/result.json'),model_state_sha256=s.digest(newck['model'])),exclusive=True)
    else:test_once(a.out,'regularize_L4_rho03',4)
    jobs=[launch_fit(a.out,case,index) for case,index in zip(('L3_rho03','L3_rho0','L2_rho03','L2_rho0'),(0,2,4,5))];s.save(a.out/'status.json',dict(state='depth_pairs_running',test_used_for_selection=False));s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in jobs],pending=[]))
    for j in jobs:finish_job(a.out,j)
    for stage in ('L3','L2'):
        audit_stage(a.out,stage,[stage+'_rho03',stage+'_rho0'])
        vals=[test_once(a.out,stage+suffix,index) for suffix,index in (('_rho03',4),('_rho0',2))];gap=100*(vals[0]['test']['accuracy']-vals[1]['test']['accuracy']);s.save(a.out/(stage+'_paired_test.json'),dict(residual_minus_control_pp=gap,macro_difference_pp=100*(vals[0]['test']['balanced_accuracy']-vals[1]['test']['balanced_accuracy']),target_gap10pp_achieved=gap>=10,test_used_for_selection=False,test_evaluations_per_model=1),exclusive=True)
    s.save(a.out/'current_processes.json',dict(active=[],pending=[]));s.save(a.out/'status.json',dict(state='complete_science_and_frozen_tests_passed',failures=[],test_used_for_selection=False,automatic_further_training=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','numeric','fit','test','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
