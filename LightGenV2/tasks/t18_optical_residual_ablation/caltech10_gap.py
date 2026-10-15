"""One frozen validation-informed Caltech10 fresh paired development attempt."""
import argparse,copy,csv,inspect,math,os,subprocess,sys,textwrap,types
from pathlib import Path
import numpy as np
import torch
import caltech20_current4 as base
import resisc10 as ten
import resisc20_batch_probe as probe
from resisc20_input_encoder import encode
s=base.s;HERE=base.HERE;ROOT=base.ROOT
PROFILE=HERE/'caltech10_gap_profile.json'
FILES=('caltech10_gap.py','caltech10_gap_profile.json','audit_caltech10_gap.py','CALTECH10_GAP_PROTOCOL_20261015.md')
RUN='caltech10_gap_s17_current4_pair150_gpu425_20261015'
CASES={'L4_gap10_rho03':.3,'L4_gap10_rho0':0.}
GPU={4:'GPU-1b963983-7909-af6e-0528-f0f0661ab549',**base.GPU}
CPU=HERE/'runs/smoke/caltech10_gap_cpu_native_20261015';NUMERIC=HERE/'runs/smoke/caltech10_gap_gpu4_numeric_20261015'
def plan():
    p=s.read(PROFILE);assert s.sha(ROOT/p['selection_source'])==p['selection_source_sha256']
    rows=list(csv.DictReader((ROOT/p['selection_source']).open()));eligible=[r for r in rows if float(r['residual_val_recall_pct'])>=60 and float(r['val_recall_difference_pp'])>=0]
    chosen=sorted(eligible,key=lambda r:(-(float(r['residual_val_recall_pct'])+.5*float(r['val_recall_difference_pp'])),r['class']))[:10]
    assert sorted(r['class'] for r in chosen)==p['classes'] and len(chosen)==10
    assert p['original_indices']==[s.read(HERE/'caltech_selected_profile.json')['selected_classes'].index(c) for c in p['classes']]
    return p
INDICES=tuple(s.read(PROFILE)['original_indices'])
def private_manifest():return Path(plan()['out'])/'data_manifest.json'
def sources():
    src={**ten.sources(),**base.sources()}
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    p=plan();src[p['selection_source']]=s.sha(ROOT/p['selection_source']);return src
def contract(depth,rho):return dict(base.contract(depth,rho),class_count=10,original_detector_indices=list(INDICES),readout='Ten frozen retained original20 windows, normalized ten probabilities; no learned electronic head')
class SubsetMoE(s.r.CorrectedMoE):
    def __init__(self,depth,rho):
        super().__init__(depth,rho);self.detectors=self.detectors[list(INDICES)].clone()
SubsetMoE.forward=ten.SubsetMoE.forward
# Reuse frozen ten-output calculations with this module's explicit identities.
for _name in ('subset_labels','load_fit','metrics','evaluate','eligibility','gate'):
    exec(compile(textwrap.dedent(inspect.getsource(getattr(ten,_name))),__file__+':sealed_'+_name,'exec'),globals())
def manifest(data):
    p=plan();assert data==Path(p['parent_data']);old=base.manifest(data);m=s.read(private_manifest());assert m['plan_sha256']==s.sha(PROFILE) and m['classes']==p['classes'] and m['original_indices']==list(INDICES)
    assert m['cache_sha256']==old['cache_sha256']==p['parent_cache_sha256'] and m['parent_manifest_sha256']==p['parent_manifest_sha256']
    for role in ('train','val','test'):
        assert m['split_ids'][role]==[sid for sid,label in zip(old['split_ids'][role],np.repeat(np.arange(20),old['supports'][role])) if label in INDICES]
        assert m['supports'][role]==[old['supports'][role][i] for i in INDICES]
    return m
def config():
    p=plan();c=base.config();c.update(dataset=p['dataset'],class_count=10,original_indices=list(INDICES),subset_plan_sha256=s.sha(PROFILE),subset_manifest_sha256=s.sha(private_manifest()),maximum_epochs=150,maximum_seconds=3600,minimum_epochs=1000,maximum_gpu_children=3,cpu_gate=CPU.relative_to(ROOT).as_posix()+'/checks.json',numeric_gate=NUMERIC.relative_to(ROOT).as_posix()+'/checks.json',scope=p['scope']);return c
def slots():
    jobs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(HERE) not in cmd:continue
            gpu=next((v.decode().split('=',1)[1] for v in (p/'environ').read_bytes().split(b'\0') if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if gpu in set(GPU.values())|{s.ALLOWED[6]}:jobs.append(dict(pid=int(p.name),gpu_uuid=gpu))
        except (OSError,UnicodeError):continue
    counts={gpu:sum(j['gpu_uuid']==gpu for j in jobs) for gpu in set(GPU.values())|{s.ALLOWED[6]}}
    assert counts[s.ALLOWED[6]]==0 and len(jobs)<=3 and max(counts.values(),default=0)<=1,jobs
    return dict(jobs=jobs,counts=counts,total=len(jobs),maximum=3,per_card_maximum=1,allowed_physical_cards=[4,2,5],a100_prohibited=True)
def claim(path,phase):
    assert not path.exists();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(path,dict(pid=os.getpid(),phase=phase,sources=src,git_commit=commit,automatic_retry=False,test_evaluations=0),exclusive=True)
def prepare(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';p=plan();out=Path(p['out']);assert not out.exists();old=base.manifest(Path(p['parent_data']));out.mkdir(exist_ok=False);claim(out/'preparation_claim.json','new_gap10_metadata_only')
    supports={role:[old['supports'][role][i] for i in INDICES] for role in ('train','val','test')};ids={role:[sid for sid,label in zip(old['split_ids'][role],np.repeat(np.arange(20),old['supports'][role])) if label in INDICES] for role in supports}
    s.save(out/'data_manifest.json',dict(dataset=p['dataset'],classes=p['classes'],original_indices=list(INDICES),supports=supports,split_ids=ids,total_records=sum(map(sum,supports.values())),cache_sha256=p['parent_cache_sha256'],parent_manifest_sha256=p['parent_manifest_sha256'],plan_sha256=s.sha(PROFILE),original_roles_unchanged=True,test_arrays_read=0,test_evaluations=0,scope=p['scope']),exclusive=True)
def bind():
    base.bind();cfg=config();s.engine.PROFILE=PROFILE;s.engine.read=lambda path:copy.deepcopy(cfg) if Path(path)==PROFILE else s.read(path)
    s.engine.DeployMoE=SubsetMoE;s.engine.ALLOWED=GPU;s.engine.sources=sources;s.engine.manifest=manifest;s.engine.load_fit=load_fit;s.engine.metrics=metrics;s.engine.evaluate=evaluate;s.engine.fresh_gate=gate;s.engine.deployment_contract=contract;s.engine.F_onehot=lambda y,smoothing:torch.nn.functional.one_hot(y,10).float()*(1-smoothing)+smoothing/10
    src=textwrap.dedent(inspect.getsource(s.engine.fit));old='20*torch.bincount(data[1],minlength=20).float()';assert src.count(old)==1;ns=dict(s.engine.__dict__);exec(compile(src.replace(old,'10*torch.bincount(data[1],minlength=10).float()'),__file__+':sealed10fit','exec'),ns);return cfg,ns['fit']
def cpu(a):
    assert a.out==CPU and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';slots();claim(a.out.with_name(a.out.name+'.claim.json'),'gap10_cpu');cfg,_=bind();m=manifest(Path(cfg['data']));data,val=[load_fit(Path(cfg['data']),role,m) for role in ('train','val')];n=len(data[1]);nv=len(val[1]);assert n==761 and nv==157 and len(set(data[1].tolist()))==10
    assert torch.allclose(s.engine.F_onehot(torch.arange(10),.08).sum(1),torch.ones(10));del data,val
    try:load_fit(Path(cfg['data']),'test',m)
    except AssertionError:pass
    else:raise AssertionError('Test role must reject')
    import check_resisc20_input as sealed
    import export_resisc10 as export
    export.INDICES=INDICES;export.contract=contract;export._ns['deployment_contract']=contract
    src=textwrap.dedent(inspect.getsource(sealed.check))
    changes={'s.r.CorrectedMoE(4,rho)':'study.SubsetMoE(4,rho)','range(20)':'range(10)','static_packages=3':'static_packages=2','model_forward_calls=18':'model_forward_calls=12','dataset_arrays_read=0':'dataset_arrays_read=2'}
    for old,new in changes.items():assert src.count(old)==1;src=src.replace(old,new)
    view=types.SimpleNamespace(CASES={k:(100,v) for k,v in CASES.items()},config=lambda case:config(),sources=sources,encode=encode,SubsetMoE=SubsetMoE)
    ns=dict(sealed.__dict__,study=view,export=export);exec(compile(src,__file__+':sealed10native','exec'),ns);ns['check'](a.out)
    for rho in (.3,0.):
        torch.manual_seed(17);model=SubsetMoE(4,rho)
        with torch.no_grad():q,_=model(encode(torch.zeros(1,3,150,150,dtype=torch.uint8),None,100))
        assert q.shape==(1,10) and torch.allclose(q,torch.full_like(q,.1),atol=1e-8) and torch.allclose(q.sum(1),torch.ones(1),atol=1e-7)
    v=s.read(a.out/'checks.json');v.update(train_samples=n,val_samples=nv,optimizer_steps_per_epoch=math.ceil(n/16),tail_batch=n%16,black10_normalized=True,model_forward_calls=14,test_arrays_read=0,test_evaluations=0);s.save(a.out/'checks.json',v)
def numeric(a):
    assert a.out==NUMERIC and not a.out.exists() and a.gpu==GPU[4] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;slots();claim(a.out.with_name(a.out.name+'.claim.json'),'gap10_numeric');v=s.read(CPU/'checks.json');assert v['passed'] and v['sources']==sources();cfg,_=bind();s.gpu_budget.ALLOWED.update(GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);m=manifest(Path(cfg['data']));data=load_fit(Path(cfg['data']),'train',m);a.out.mkdir(parents=True,exist_ok=False)
    checks=[gate(4,rho,data,cfg,'cuda',8) for rho in (.3,0.)];assert len({x['initial_parameter_sha256'] for x in checks})==1
    s.save(a.out/'checks.json',dict(passed=True,checks=checks,sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),policy=policy,optimizer_steps=0,test_arrays_read=0,test_evaluations=0),exclusive=True)
def prerequisites():
    cfg=config();manifest(Path(cfg['data']))
    for out in (CPU,NUMERIC):
        v=s.read(out/'checks.json');assert v['passed'] and v['sources']==sources();s.engine.verify_sources(v['sources'],v['git_commit'])
    return cfg
def fit(a):
    assert a.case in CASES and a.rho==CASES[a.case] and a.depth==4 and a.gpu in GPU.values();prerequisites();slots();cfg,fn=bind();assert a.data==Path(cfg['data']);a.resume_parent=None;fn(a)
    f=s.read(a.out/'fit_result.json');f['scope']=plan()['scope'];s.save(a.out/'fit_result.json',f);s.save(a.out/'validation_gate.json',eligibility(f,s.read(a.out/'history.json')));assert f['epochs_completed']==150 and f['stop_reason']=='maximum_epochs'
_q=textwrap.dedent(inspect.getsource(base.queue));assert _q.count('for index in (2,5):')==1;_q=_q.replace('for index in (2,5):','for index in (4,2,5):').replace("live['total']>=2","live['total']>=3")
_ns=dict(base.__dict__,__file__=__file__,prerequisites=prerequisites,sources=sources,config=config,RUN=RUN,CASES=CASES,GPU=GPU,slots=slots);exec(compile(_q,__file__+':gap10queue','exec'),_ns);queue=_ns['queue']
_c=textwrap.dedent(inspect.getsource(base.campaign));_c=_c.replace("'audit_caltech20_current4.py'","'audit_caltech10_gap.py'").replace("['epochs_completed']==100","['epochs_completed']==150").replace("'caltech_current4_complete_science_passed'","'caltech10_gap_complete_science_passed'")
_ns=dict(base.__dict__,__file__=__file__,prerequisites=prerequisites,sources=sources,config=config,RUN=RUN,CASES=CASES,GPU=GPU,slots=slots,claim=claim);exec(compile(_c,__file__+':gap10campaign','exec'),_ns);campaign=_ns['campaign']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','cpu','numeric','fit','queue','campaign'],required=True);p.add_argument('--out',type=Path,required=True)
    for k in ('case','gpu'):p.add_argument('--'+k)
    p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--data',type=Path);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
