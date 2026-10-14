"""New human-authorized10-class task; original dataset/source/runs preserved."""
import argparse,copy,hashlib,inspect,math,os,subprocess,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import resisc20 as s
import resisc20_input_study as parent
import resisc20_mask_batch as batch
import resisc20_batch_probe as probe
from resisc20_input_encoder import encode
from resisc10_model import SubsetMoE,contract,INDICES
HERE,ROOT=s.HERE,s.ROOT
PLAN=HERE/'resisc10_plan.json'
FILES=('resisc10_plan.json','resisc10.py','resisc10_model.py','check_resisc10.py','audit_resisc10.py','export_resisc10.py','RESISC10_PROTOCOL_20261015.md')
CASES={'L4_10_rho03':.3,'L4_10_rho0':0.}
CPU=HERE/'runs/smoke/resisc10_cpu_native_normalization_checked_20261015'
NUMERIC=HERE/'runs/smoke/resisc10_gpu2_numeric_normalization_checked_20261015'
RUN='resisc10_s17_mainL4_input100_rho_pair120_micro8_gpu25_20261015'
GPU={k:s.ALLOWED[k] for k in (2,5)}
def plan():
    import csv
    v=s.read(PLAN);path=ROOT/v['selection_source'];assert s.sha(path)==v['selection_source_sha256']
    rows=[x for x in csv.DictReader(path.open()) if x['case']==v['selection_case']]
    assert len(rows)==20 and len({x['class_name'] for x in rows})==20
    ranked=sorted(rows,key=lambda x:(-float(x['val_recall']),x['class_name']))
    assert v['classes']==sorted(x['class_name'] for x in ranked[:10]) and tuple(v['original_indices'])==INDICES
    assert v['validation_informed_class_selection'] and v['selection_epoch']==111
    return v

def sources():
    v=parent.sources()
    for n in FILES+('audit_resisc20_input_pair.py',):v[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    v[plan()['selection_source']]=s.sha(ROOT/plan()['selection_source'])
    return v
def private_manifest():return Path(plan()['out'])/'data_manifest.json'
def slots():
    v=s.slots();assert v['counts'][s.ALLOWED[6]]==0 and v['total']<=2 and all(v['counts'][u]<=1 for u in GPU.values()),v
    return dict(v,maximum=2,per_card_maximum=1,allowed_physical_cards=[2,5],a100_prohibited=True)
def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    v=plan();base=Path(v['out']);assert not base.exists();old=s.manifest(Path(v['parent_data']))
    assert old['classes']==v['parent_classes'] and old['cache_sha256']==v['parent_cache_sha256']
    base.mkdir(parents=True,exist_ok=False);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    s.save(base/'preparation_claim.json',dict(pid=os.getpid(),sources=src,git_commit=commit,plan_sha256=s.sha(PLAN),automatic_retry=False),exclusive=True)
    selected=set(v['classes']);ids={role:[sid for sid in old['split_ids'][role] if Path(sid).parent.name in selected] for role in ('train','val','test')}
    supports={role:[sum(Path(sid).parent.name==c for sid in ids[role]) for c in v['classes']] for role in ids}
    assert supports=={role:[old['supports'][role][k] for k in INDICES] for role in ids}
    assert tuple(sum(supports[k]) for k in ('train','val','test'))==(5600,700,700)
    m=dict(dataset=v['dataset'],classes=v['classes'],original_indices=list(INDICES),total_records=7000,supports=supports,split_ids=ids,cache_sha256=v['parent_cache_sha256'],parent_manifest_sha256=v['parent_manifest_sha256'],plan_sha256=s.sha(PLAN),selection_seed=v['selection_seed'],selection_rule=v['selection_rule'],selection_source_sha256=v['selection_source_sha256'],validation_informed_class_selection=True,original_roles_unchanged=True,data_storage='Referenced immutable original20 cache; train/val filtered only at load, test images never materialized',test_arrays_read=0,test_evaluations=0,scope='Human requested10-class development task, not original20 target acceptance')
    s.save(base/'data_manifest.json',m,exclusive=True)
    return dict(passed=True,classes=v['classes'],supports=supports,manifest_sha256=s.sha(base/'data_manifest.json'),cache_sha256=m['cache_sha256'],test_arrays_read=0,test_evaluations=0)
def manifest(data):
    v=plan();assert data==Path(v['parent_data']);old=s.manifest(data);m=s.read(private_manifest())
    assert m['plan_sha256']==s.sha(PLAN) and m['classes']==v['classes'] and m['original_indices']==list(INDICES)
    assert m['cache_sha256']==old['cache_sha256']==v['parent_cache_sha256'] and m['parent_manifest_sha256']==v['parent_manifest_sha256']
    for role in ('train','val','test'):
        assert m['split_ids'][role]==[sid for sid in old['split_ids'][role] if Path(sid).parent.name in set(v['classes'])]
        assert m['supports'][role]==[old['supports'][role][k] for k in INDICES]
    return m
def config():
    v=plan();c=s.read(parent.PROFILE);c.pop('cases')
    c.update(dataset=v['dataset'],input_tile=100,subset_plan_sha256=s.sha(PLAN),subset_manifest_sha256=s.sha(private_manifest()),class_count=10,original_indices=list(INDICES),maximum_gpu_children=2,maximum_gpu_children_per_card=1,numeric_gate=NUMERIC.relative_to(ROOT).as_posix()+'/checks.json',cpu_gate=CPU.relative_to(ROOT).as_posix()+'/checks.json',scope='Explicit human requested10-class development task, original20 acceptance not claimed')
    assert c['maximum_epochs']==120 and c['maximum_seconds']==7200
    return c
def subset_labels(labels):
    lut=np.full(20,-1,np.int64);lut[list(INDICES)]=np.arange(10);values=lut[labels];return values[values>=0]
def load_fit(data,role,m):
    assert role in ('train','val'),'No test arrays during preparation/gates/fit/audit'
    with np.load(data,allow_pickle=False) as z:x,y,ids=[z[role+'_'+k].copy() for k in ('images','labels','ids')]
    keep=np.isin(y,INDICES);x=x[keep];y=subset_labels(y);ids=ids[keep]
    assert ids.tolist()==m['split_ids'][role] and np.bincount(y,minlength=10).tolist()==m['supports'][role]
    assert x.dtype==np.uint8 and x.shape==(len(y),150,150,3)
    return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids
def metrics(y,p):
    assert p.shape==(len(y),10) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    pred=p.argmax(1);support=np.bincount(y,minlength=10);assert (support>0).all()
    cm=np.zeros((10,10),np.int64);np.add.at(cm,(y,pred),1);recall=cm.diagonal()/support;nll=-np.log(p[np.arange(len(y)),y].clip(1e-12))
    return dict(n=len(y),accuracy=float((pred==y).mean()),balanced_accuracy=float(recall.mean()),recall=recall.tolist(),support=support.tolist(),confusion_matrix=cm.tolist(),nll=float(nll.mean()),balanced_nll=float(np.mean([nll[y==k].mean() for k in range(10)])))
@torch.no_grad()
def evaluate(model,data,micro):
    model.eval();ps=[];cs=[];ws=[];device=model.raw_router.device
    for idx in torch.arange(len(data[1])).split(micro):
        p,c=model(encode(data[0][idx].to(device),None,100));ps.append(p.cpu().numpy());cs.append(c.cpu().numpy());ws.append(model.routing_weights.cpu().numpy())
    p=np.concatenate(ps).astype(np.float64);w=np.concatenate(ws);y=data[1].numpy();v=metrics(y,p);v['detector_capture']=float(np.concatenate(cs).mean())
    v['routing']=dict(nonzero_fraction=(w>0).mean(0).tolist(),weight_mean=w.mean(0).tolist(),weight_std=w.std(0).tolist(),weight_min=w.min(0).tolist(),weight_max=w.max(0).tolist(),per_class_weight_mean=[w[y==k].mean(0).tolist() for k in range(10)],ranked_top2_diagnostic_not_hard_gate=True)
    rows=[dict(sample_id=str(sid),label_true=int(label),label_pred=int(row.argmax()),**{f'score{k}':float(row[k]) for k in range(10)}) for sid,label,row in zip(data[2],data[1],p)]
    return v,rows
def eligibility(fit,hist):
    tm,vm=fit['metrics']['train'],fit['metrics']['val'];tail=hist[-min(20,len(hist)):]
    checks=dict(val_accuracy=vm['accuracy']>=.65,val_macro=vm['balanced_accuracy']>=.60,no_zero_train_val=min(tm['recall'])>0 and min(vm['recall'])>0,val_each_recall=min(vm['recall'])>=.2,no_persistent_zero=all(sum(h['val']['recall'][k]==0 for h in tail)<=len(tail)/2 for k in range(10)),gap=100*(tm['accuracy']-vm['accuracy'])<=15,routing_all_slots=all(v==1 for split in (tm,vm) for v in split['routing']['nonzero_fraction']),routing_input_dependent=all(max(v['routing']['weight_std'])>1e-4 for v in (tm,vm)))
    return dict(passed=all(checks.values()),checks=checks,selected_epoch=fit['selected_epoch'],gap_pp=100*(tm['accuracy']-vm['accuracy']),class_count=10,original20_target_achieved=False,test_evaluations=0)
def gate(depth,rho,data,cfg,device,micro,state=None):
    assert depth==4 and micro==8;s.engine.seed(cfg['seed']);model=SubsetMoE(depth,rho).to(device)
    if state is not None:model.load_state_dict(state,strict=True)
    initial=s.digest(model.state_dict());weights=(len(data[1])/(10*torch.bincount(data[1],minlength=10).float())).to(device)
    order,theta=s.engine.epoch_plan(1,len(data[1]),cfg);tail=len(data[1])%16 or 16
    groups=[np.asarray([int((data[1]==k).nonzero()[0]) for k in range(10)]),order[:16],order[-tail:]];checks=[]
    for idx in groups:
        x,y,_=s.engine.mixed_batch(data,idx,theta,1,0,cfg,device)
        v=probe.compare(probe.compute(model,x,y,weights,cfg,1),probe.compute(model,x,y,weights,cfg,micro));assert v['passed'];checks.append(v)
    norms={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert len(norms)==11 and all(np.isfinite(v) and v>0 for v in norms.values())
    del model;torch.cuda.empty_cache()
    return dict(passed=True,depth=depth,rho=rho,micro=8,classes_checked=10,macro_checked=16,tail_checked=tail,checks=checks,phase_gradients=norms,all_phase_gradients_finite_nonzero=True,initial_parameter_sha256=initial,parameters=908552,test_evaluations=0)
def bind():
    s.bind();batch.install();cfg=config();s.engine.PROFILE=PLAN;s.engine.read=lambda path:copy.deepcopy(cfg) if Path(path)==PLAN else s.read(path)
    s.engine.DeployMoE=SubsetMoE;s.engine.ALLOWED=GPU;s.engine.sources=sources;s.engine.manifest=manifest;s.engine.load_fit=load_fit;s.engine.metrics=metrics;s.engine.evaluate=evaluate;s.engine.fresh_gate=gate;s.engine.input_amplitude=lambda images,theta=None:encode(images,theta,100)
    s.engine.deployment_contract=contract;s.engine.F_onehot=lambda y,smoothing:torch.nn.functional.one_hot(y,10).float()*(1-smoothing)+smoothing/10
    # Only the class-balancing divisor changes; sealed fit updates and budgets remain.
    source=textwrap.dedent(inspect.getsource(s.engine.fit));old='20*torch.bincount(data[1],minlength=20).float()';assert source.count(old)==1
    namespace=dict(s.engine.__dict__);exec(compile(source.replace(old,'10*torch.bincount(data[1],minlength=10).float()'),__file__+':sealed10fit','exec'),namespace)
    return cfg,namespace['fit']
def prerequisites():
    cfg=config();manifest(Path(cfg['data']));src=sources()
    for out in (CPU,NUMERIC):
        v=s.read(out/'checks.json');assert v['passed'] and v['sources']==src;s.engine.verify_sources(src,v['git_commit'])
    return cfg
def numeric(a):
    assert a.gpu==GPU[2] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and not a.out.exists();slots()
    cpu=s.read(CPU/'checks.json');assert cpu['passed'] and cpu['sources']==sources();s.engine.verify_sources(cpu['sources'],cpu['git_commit'])
    cfg,_=bind();s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    m=manifest(Path(cfg['data']));data=load_fit(Path(cfg['data']),'train',m);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);a.out.mkdir(parents=True,exist_ok=False)
    checks=[gate(4,rho,data,cfg,'cuda',8) for rho in (.3,0.)];assert len({v['initial_parameter_sha256'] for v in checks})==1
    s.save(a.out/'checks.json',dict(passed=True,checks=checks,sources=src,git_commit=commit,policy=policy,train_samples=len(data[1]),optimizer_steps_per_epoch=math.ceil(len(data[1])/16),tail_batch=len(data[1])%16 or 16,same_phase_initialization=True,optimizer_steps=0,test_arrays_read=0,test_evaluations=0),exclusive=True)
def fit(a):
    assert a.case in CASES and a.depth==4 and a.rho==CASES[a.case] and a.gpu in GPU.values();prerequisites();slots();cfg,function=bind();assert a.data==Path(cfg['data']);a.resume_parent=None;function(a)
    f=s.read(a.out/'fit_result.json');f['scope']='Human requested10-class development task; does not establish original20 target success; no test';s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',eligibility(f,s.read(a.out/'history.json')))
def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();cfg=prerequisites();slots();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,config=cfg,command=sys.argv,phase='human_requested10_two_fresh_full_size_L4_models',test_evaluations=0))
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
            case=pending.pop(0);rho=CASES[case];out=a.out/case;assert not out.exists();log=(a.out/(case+'.log')).open('x')
            cmd=[sys.executable,str(HERE/'resisc10.py'),'--phase','fit','--case',case,'--depth','4','--rho',str(rho),'--gpu',gpu,'--data',cfg['data'],'--out',str(out)]
            process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(case=case,rho=rho,pid=process.pid,gpu_uuid=gpu,process=process,log=log))
        s.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        s.save(a.out/'status.json',dict(state='resisc10_failed' if failures else 'resisc10_running' if active or pending else 'resisc10_complete_awaiting_CPU_audit',completed=completed,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
        if active or pending:time.sleep(5)
    s.engine.verify_sources(src,commit)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','numeric','fit','queue'],required=True);p.add_argument('--out',type=Path);p.add_argument('--case');p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--gpu');p.add_argument('--data',type=Path);a=p.parse_args()
    try:
        if a.phase=='prepare':print(__import__('json').dumps(prepare()))
        else:globals()[a.phase](a)
    except BaseException as e:
        if a.out is not None and a.out.exists() and not (a.out/'failure.json').exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
