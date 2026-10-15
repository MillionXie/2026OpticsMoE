"""One fresh current-four-expert Caltech20 equal100 pair; old runs immutable."""
import argparse,copy,inspect,math,os,subprocess,sys,textwrap,time,types
from pathlib import Path
import numpy as np
import torch
import resisc20 as s
import resisc20_input_study as frozen
import resisc20_mask_batch as batch
import resisc20_batch_probe as probe
from resisc20_input_encoder import encode,contract as input_contract
from audit_caltech_repair234 import exited
HERE,ROOT=s.HERE,s.ROOT
PROFILE=HERE/'caltech20_current4_profile.json'
FILES=('caltech20_current4.py','caltech20_current4_profile.json','audit_caltech20_current4.py','CALTECH20_CURRENT4_PROTOCOL_20261015.md','caltech_selected_profile.json')
RUN='caltech20_screened_current4_s17_pair100_input100_mixup02_gpu25_20261015'
CASES={'L4_current_rho03':.3,'L4_current_rho0':0.}
GPU={k:s.ALLOWED[k] for k in (2,5)}
CPU=HERE/'runs/smoke/caltech20_current4_cpu_native_20261015'
NUMERIC=HERE/'runs/smoke/caltech20_current4_gpu2_numeric_20261015'

def config():
    p=s.read(PROFILE)
    assert p['maximum_epochs']==100 and p['minimum_epochs']>100 and p['mixup_alpha']==.2
    assert p['microbatch']==8 and p['batch_size']==16 and p['input_tile']==100
    assert p['cache_sha256']==s.read(HERE/'caltech_selected_profile.json')['data_contract']['cache_sha256']
    return p

def sources():
    src=frozen.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def manifest(data):
    cfg=config();assert data==Path(cfg['data'])
    m=s.read(data.parent/'data_manifest.json')
    assert s.sha(data)==cfg['cache_sha256']==m['cache_sha256']
    assert s.sha(data.parent/'data_manifest.json')==cfg['manifest_sha256']
    assert m['classes']==s.read(HERE/'caltech_selected_profile.json')['selected_classes']
    assert m['total_records']==3719 and len(m['classes'])==20
    assert [sum(m['supports'][role]) for role in ('train','val','test')]==[1372,283,2064]
    return m

def contract(depth,rho):
    return dict(input_contract(depth,rho,100),task='Frozen screened Caltech20; current four experts/four main stages; fixed20 windows; no electronic classifier')

def slots():
    v=s.slots();assert v['counts'][s.ALLOWED[6]]==0 and v['total']<=2 and all(v['counts'][u]<=1 for u in GPU.values()),v
    return dict(v,maximum=2,per_card_maximum=1,allowed_physical_cards=[2,5],a100_prohibited=True)

def claim(path,phase):
    assert not path.exists()
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    s.save(path,dict(pid=os.getpid(),phase=phase,sources=src,git_commit=commit,automatic_retry=False,test_evaluations=0),exclusive=True)

@torch.no_grad()
def evaluate(model,data,micro):
    model.eval();ps=[];cs=[];ws=[];device=model.raw_router.device
    for idx in torch.arange(len(data[1])).split(micro):
        p,c=model(encode(data[0][idx].to(device),None,100));ps.append(p.cpu().numpy());cs.append(c.cpu().numpy());ws.append(model.routing_weights.cpu().numpy())
    p=np.concatenate(ps).astype(np.float64);w=np.concatenate(ws).astype(np.float64);y=data[1].numpy();v=s.engine.metrics(y,p);v['detector_capture']=float(np.concatenate(cs).mean())
    v['routing']=dict(nonzero_fraction=(w>0).mean(0).tolist(),weight_mean=w.mean(0).tolist(),weight_std=w.std(0).tolist(),weight_min=w.min(0).tolist(),weight_max=w.max(0).tolist(),per_class_weight_mean=[w[y==k].mean(0).tolist() for k in range(20)],ranked_top2_diagnostic_not_hard_gate=True)
    rows=[dict(sample_id=str(sid),label_true=int(label),label_pred=int(row.argmax()),**{f'score{k}':float(row[k]) for k in range(20)}) for sid,label,row in zip(data[2],data[1],p)]
    return v,rows

def gate(depth,rho,data,cfg,device,micro,state=None):
    assert depth==4 and micro==8;s.engine.seed(cfg['seed']);model=s.r.CorrectedMoE(depth,rho).to(device)
    if state is not None:model.load_state_dict(state,strict=True)
    initial=s.digest(model.state_dict());weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
    order,theta=s.engine.epoch_plan(1,len(data[1]),cfg);tail=len(data[1])%16 or 16;checks=[]
    groups=[('all20',np.asarray([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('tail',order[-tail:],math.ceil(len(order)/16)-1)]
    for label,idx,b in groups:
        x,y,_=s.engine.mixed_batch(data,idx,theta,1,b,cfg,device)
        v=probe.compare(probe.compute(model,x,y,weights,cfg,1),probe.compute(model,x,y,weights,cfg,micro));assert v['passed'],v;checks.append(dict(v,samples=label))
    norms={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert len(norms)==11 and all(np.isfinite(v) and v>0 for v in norms.values())
    del model;torch.cuda.empty_cache()
    return dict(passed=True,depth=4,rho=rho,micro=8,classes_checked=20,macro_checked=16,tail_checked=tail,checks=checks,phase_gradients=norms,all_phase_gradients_finite_nonzero=True,initial_parameter_sha256=initial,parameters=908552,test_evaluations=0)

def bind():
    s.bind();batch.install();cfg=config();s.engine.PROFILE=PROFILE;s.engine.read=lambda path:copy.deepcopy(cfg) if Path(path)==PROFILE else s.read(path)
    s.engine.ALLOWED=GPU;s.engine.sources=sources;s.engine.manifest=manifest;s.engine.evaluate=evaluate;s.engine.fresh_gate=gate
    s.engine.input_amplitude=lambda images,theta=None:encode(images,theta,100);s.engine.deployment_contract=contract
    return cfg

def prerequisites():
    cfg=config();manifest(Path(cfg['data']))
    for out in (CPU,NUMERIC):
        v=s.read(out/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
    return cfg

def cpu(a):
    assert a.out==CPU and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    slots();claim(a.out.with_name(a.out.name+'.claim.json'),'new_caltech_current4_cpu');cfg=bind();m=manifest(Path(cfg['data']))
    data,val=[s.engine.load_fit(Path(cfg['data']),role,m) for role in ('train','val')]
    assert len(data[1])==1372 and len(val[1])==283 and len(set(data[1].tolist()))==20
    assert math.ceil(len(data[1])/16)==86 and len(data[1])%16==12
    q=s.engine.F_onehot(torch.arange(20),.08);assert q.shape==(20,20) and torch.allclose(q.sum(1),torch.ones(20))
    try:s.engine.load_fit(Path(cfg['data']),'test',m)
    except AssertionError:pass
    else:raise AssertionError('Fit test role must reject')
    del data,val
    import check_resisc20_input as sealed
    src=textwrap.dedent(inspect.getsource(sealed.check))
    for old,new in [('static_packages=3','static_packages=2'),('model_forward_calls=18','model_forward_calls=12')]:assert src.count(old)==1;src=src.replace(old,new)
    view=types.SimpleNamespace(CASES={k:(100,v) for k,v in CASES.items()},config=lambda case:config(),sources=sources,encode=encode)
    ns=dict(sealed.__dict__);ns['study']=view;exec(compile(src,__file__+':sealed20_native','exec'),ns);ns['check'](a.out)
    for rho in (.3,0.):
        torch.manual_seed(17);model=s.r.CorrectedMoE(4,rho)
        with torch.no_grad():p,_=model(encode(torch.zeros(1,3,150,150,dtype=torch.uint8),None,100))
        assert p.shape==(1,20) and torch.allclose(p,torch.full_like(p,.05),atol=1e-8) and torch.allclose(p.sum(1),torch.ones(1),atol=1e-7)
    v=s.read(a.out/'checks.json');v.update(train_samples=1372,val_samples=283,optimizer_steps_per_epoch=86,tail_batch=12,black20_probability_normalized=True,model_forward_calls=14,dataset_arrays_read=2,classes=m['classes'],test_evaluations=0)
    s.save(a.out/'checks.json',v)

def numeric(a):
    assert a.out==NUMERIC and not a.out.exists() and a.gpu==GPU[2] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    slots();v=s.read(CPU/'checks.json');assert v['passed'] and v['sources']==sources();claim(a.out.with_name(a.out.name+'.claim.json'),'new_caltech_current4_numeric')
    cfg=bind();s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    m=manifest(Path(cfg['data']));data=s.engine.load_fit(Path(cfg['data']),'train',m);a.out.mkdir(parents=True,exist_ok=False)
    checks=[gate(4,rho,data,cfg,'cuda',8) for rho in (.3,0.)];assert len({v['initial_parameter_sha256'] for v in checks})==1
    s.save(a.out/'checks.json',dict(passed=True,checks=checks,sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),policy=policy,optimizer_steps=0,test_arrays_read=0,test_evaluations=0),exclusive=True)

def fit(a):
    assert a.case in CASES and a.depth==4 and a.rho==CASES[a.case] and a.gpu in GPU.values()
    cfg=prerequisites();slots();bind();assert a.data==Path(cfg['data']);a.resume_parent=None;s.engine.fit(a)
    f=s.read(a.out/'fit_result.json');f['scope']='Frozen screened Caltech20, current four experts, fresh equal100 pair; validation only; no test'
    s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',s.eligibility(f,s.read(a.out/'history.json')))
    assert f['epochs_completed']==100 and f['stop_reason']=='maximum_epochs','Incomplete100: preserve and stop, no extension'

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();cfg=prerequisites();slots();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,config=cfg,command=sys.argv,test_evaluations=0))
    pending=list(CASES);active=[];completed=[];failures=[]
    while pending or active:
        for j in list(active):
            rc=j['process'].poll()
            if rc is not None:
                j['log'].close();rec={k:v for k,v in j.items() if k not in ('process','log')};rec['exit_code']=rc;(completed if rc==0 else failures).append(rec);active.remove(j)
        if failures:pending=[]
        for index in (2,5):
            if not pending or failures:break
            gpu=GPU[index];live=slots()
            if live['total']>=2 or live['counts'][gpu]>=1 or any(j['gpu_uuid']==gpu for j in active):continue
            s.gpu_budget.ALLOWED.update(GPU.values())
            try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            key=pending.pop(0);out=a.out/key;assert not out.exists();log=(a.out/(key+'.log')).open('x')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','fit','--case',key,'--depth','4','--rho',str(CASES[key]),'--gpu',gpu,'--data',cfg['data'],'--out',str(out)]
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(case=key,rho=CASES[key],pid=child.pid,gpu_uuid=gpu,process=child,log=log))
        s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        s.save(a.out/'status.json',dict(state='pair_failed' if failures else 'pair_running' if active or pending else 'pair_complete_awaiting_saved_science',completed=completed,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
        if active or pending:time.sleep(5)
    s.engine.verify_sources(src,commit)

def campaign(a):
    assert a.out==HERE/'runs/simulation'/RUN and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    prerequisites();slots();claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'new_caltech_current4_campaign')
    with a.out.with_name(a.out.name+'.queue.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','queue','--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=child.wait()
    s.save(a.out/'queue_exit.json',dict(pid=child.pid,returncode=rc,actual_exit=exited(child.pid)),exclusive=True);state=s.read(a.out/'status.json')
    assert rc==0 and exited(child.pid) and state['active']==state['pending']==0 and not state['failures'] and len(state['completed'])==2
    assert all(j['exit_code']==0 and exited(j['pid']) and s.read(a.out/j['case']/'fit_result.json')['epochs_completed']==100 for j in state['completed'])
    claim(a.out/'audit_claim.json','new_caltech_current4_saved_science')
    with (a.out/'audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_current4.py'),'--base',str(a.out),'--out',str(a.out/'cpu_report'),'--data',config()['data']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(a.out/'cpu_report/audit.json')['passed'],'Saved science failed: no retry'
    s.save(a.out/'audit_completed.json',dict(passed=True,audit_sha256=s.sha(a.out/'cpu_report/audit.json'),actual_audit_returncode=rc),exclusive=True)
    s.save(a.out/'status.json',dict(state,state='caltech_current4_complete_science_passed'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','numeric','fit','queue','campaign'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');p.add_argument('--data',type=Path);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        dest=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not dest.exists():s.save(dest,dict(error=repr(e),pid=os.getpid(),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
