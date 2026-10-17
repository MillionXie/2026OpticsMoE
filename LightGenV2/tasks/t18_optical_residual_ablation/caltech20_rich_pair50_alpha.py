"""Fresh richest20 paired50; no old run/model/state is overwritten or resumed."""
import argparse,ast,copy,hashlib,inspect,json,math,os,subprocess,sys,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import caltech20_equal60_logrepair as frozen
from caltech20_rich_pair50_alpha_model import PairModel,BOUNDS,INITIAL
from resisc20_input_encoder import encode

HERE,ROOT,s,engine=frozen.HERE,frozen.ROOT,frozen.s,frozen.engine
FILES=('caltech20_rich_pair50_alpha.py','caltech20_rich_pair50_alpha_model.py','caltech20_rich_pair50_alpha_profile.json','check_caltech20_rich_pair50_alpha.py','audit_caltech20_rich_pair50_alpha.py','CALTECH20_RICH_PAIR50_ALPHA_PROTOCOL_20261017.md')
RUN='caltech20_rich20_current246_s17_pair50_asymmetricAlpha_20261017'
OUT=HERE/'runs/simulation'/RUN
CPU=HERE/'runs/smoke/caltech20_rich_pair50_asymmetricAlpha_cpu_20261017'
CASES={f'L{d}_rho{tag}':(d,rho) for d in (6,4,2) for tag,rho in [('03',.3),('0',0.)]}
GPU=dict(frozen.GPU)
ALL_GPU={**GPU,2:'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',5:'GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb',6:'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'}
dataset=frozen.dataset
RegularizedMoE=PairModel

@lru_cache(None)
def manifest():
    c=s.read(HERE/FILES[2]);m=dataset.manifest();p=Path(c['data_directory'])
    assert s.sha(p/'data_manifest.json')==c['manifest_sha256'] and m['cache_sha256']==c['cache_sha256'] and m['classes']==c['classes']
    assert [sum(m['supports'][r]) for r in ('train','val','test')]==[3141,383,411]
    return m

def config():
    c=s.read(HERE/FILES[2]);m=manifest();c.update(data=str(Path(c['data_directory'])/'caltech20_rich_split.npz'),training_samples=3141,steps_per_epoch=197,tail_samples=5,class_count=20)
    assert c['maximum_epochs']==50 and c['alpha_bounds_by_arm']=={'rho0':[.8501,.95],'rho03':[.7001,.95]} and c['alpha_initial']==INITIAL and m['classes']==c['classes'];return c

def sources():
    z=frozen.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def identity():
    z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def cpu_ready():
    c=s.read(CPU/'checks.json');assert c['passed'] and c['sources']==sources();engine.verify_sources(c['sources'],c['git_commit'])

def slots():
    jobs=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            cmd=path.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(HERE) not in cmd:continue
            gpu=next((x.decode().split('=',1)[1] for x in path.joinpath('environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if gpu in ALL_GPU.values():jobs.append(dict(pid=int(path.name),gpu_uuid=gpu))
        except (OSError,UnicodeError):continue
    counts={g:sum(j['gpu_uuid']==g for j in jobs) for g in ALL_GPU.values()}
    assert counts[ALL_GPU[6]]==counts[ALL_GPU[2]]==counts[ALL_GPU[5]]==0 and len(jobs)<=4 and max(counts.values(),default=0)<=1,jobs
    return dict(total=len(jobs),counts=counts,jobs=jobs)

@lru_cache(None)
def cache_identity():assert s.sha(config()['data'])==config()['cache_sha256'];return True

def load(role):
    assert role in ('train','val','test');c=config();m=manifest();cache_identity()
    with np.load(c['data'],allow_pickle=False) as z:x,y,ids=[z[role+'_'+k].copy() for k in ('images','labels','ids')]
    assert ids.tolist()==m['split_ids'][role] and np.bincount(y,minlength=20).tolist()==m['supports'][role]
    assert x.dtype==np.uint8 and x.shape==(len(y),150,150,3)
    return torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids

mixed,loss,save_np,evaluate=frozen.mixed,frozen.loss,frozen.save_np,frozen.evaluate

def gradients(model,x,y,mask,weights,cfg,micro):
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[]
    for off in range(0,len(x),micro):
        model.linear.mask=None if mask is None else mask[off:off+micro];p,c=model(x[off:off+micro]);(loss(model,p,c,y[off:off+micro],weights,cfg)*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach())
    model.linear.mask=None
    return torch.cat(ps),torch.cat(cs),{n:p.grad.detach().cpu().numpy().astype(np.float64) for n,p in model.named_parameters()}

def gate(depth,rho,data,device):
    cfg=config();engine.seed(cfg['seed']);model=PairModel(depth,rho).to(device);initial=s.digest(model.state_dict());weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
    order,theta=engine.epoch_plan(1,len(data[1]),cfg);checks=[]
    for name,idx,number in [('all20',np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]),0),('macro16',order[:16],0),('actual_tail5',order[-5:],196)]:
        x,y,mask,_=mixed(data,idx,theta,1,number,cfg,device)
        for method in ('ordinary','dropout'):
            used=None if method=='ordinary' else mask;p,c,g=gradients(model,x,y,used,weights,cfg,1);q,d,h=gradients(model,x,y,used,weights,cfg,4)
            norms={n:float(np.linalg.norm(v.ravel())) for n,v in h.items()};assert len(norms)=={2:12,4:21,6:30}[depth] and all(np.isfinite(v).all() and norms[n]>0 for n,v in h.items())
            rel=max(float(np.linalg.norm((g[n]-v).ravel())/np.linalg.norm(v.ravel())) for n,v in h.items());cosine=min(float(np.dot(g[n].ravel(),v.ravel())/(np.linalg.norm(g[n].ravel())*np.linalg.norm(v.ravel()))) for n,v in h.items());err=float((p-q).abs().max());cerr=float((c-d).abs().max())
            assert err<=2e-6 and cerr<=2e-6 and rel<=.02 and cosine>=.999,(name,method,err,cerr,rel,cosine)
            checks.append(dict(samples=name,n=len(idx),method=method,probability_max_error=err,capture_max_error=cerr,gradient_max_relative_l2=rel,gradient_min_cosine=cosine,gradient_norms=norms))
    assert s.digest(model.state_dict())==initial and all(BOUNDS[rho][0]<v<=BOUNDS[rho][1] for v in model.alphas())
    result=dict(passed=True,depth=depth,rho=rho,initial_parameter_sha256=initial,initial_nonalpha_sha256=s.digest({n:v for n,v in model.state_dict().items() if not n.startswith('fusion_logits.')}),initial_effective_alpha=model.alphas(),contract=model.contract(),checks=checks,optimizer_steps=0,test_arrays_read=0);del model;torch.cuda.empty_cache();return result

# Reuse the frozen complete macro/Adam/EMA queue kernel in an explicit namespace.
# Epoch count comes from the new profile, not a source literal. Only status and
# the independently owned auditor target are renamed; no frozen file is edited.
for routine in ('fit','campaign','supervise'):
    tree=ast.parse(inspect.getsource(getattr(frozen,routine)))
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            node.value=node.value.replace('audit_caltech20_equal60_logrepair.py','audit_caltech20_rich_pair50_alpha.py').replace('full60','full50').replace('equal60','equal50').replace('audit60','audit50').replace('Full60','Full50')
        if isinstance(node,ast.keyword) and node.arg=='controls_frozen_at60':node.arg='controls_frozen_at50'
    exec(compile(ast.fix_missing_locations(tree),str(HERE/FILES[0])+':'+routine,'exec'),globals())
exec(compile(ast.parse(inspect.getsource(frozen.b.test)),str(HERE/FILES[0])+':test','exec'),globals())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','supervise','test'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
