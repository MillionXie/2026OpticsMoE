"""One fixed TRAIN/VAL alpha sensitivity and gradient diagnosis, TEST0."""
import argparse,importlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_robust_head_identity_repair as previous
from caltech20_rich_alpha40_model import Alpha40MoE,remap,LOW,HIGH
s,base,engine,resources=previous.s,previous.base,previous.engine,previous.resources
HERE,ROOT=base.HERE,base.ROOT
FILES=('caltech20_rich_alpha40_model.py','caltech20_rich_alpha40_diagnosis.py','caltech20_rich_alpha40_diagnosis_profile.json','CALTECH20_RICH_ALPHA40_DIAGNOSIS_PROTOCOL_20261018.md')
CPU=HERE/'runs/smoke/caltech20_rich_alpha40_diagnosis_cpu_20261018'
OUT=HERE/'runs/smoke/caltech20_rich_alpha40_trainval_diagnosis_20261018'
def config():return s.read(HERE/FILES[2])
def sources():
    z=previous.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();z=sources();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def parent(d):
    q=config()['parents'][str(d)];path=Path(q['path']);assert s.sha(path)==q['sha256'];pt=torch.load(path,map_location='cpu',weights_only=False)
    assert isinstance(pt['sources'],dict);engine.verify_sources(pt['sources'],q['source_git'])
    return q,pt
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not CPU.exists();claim(CPU.with_name(CPU.name+'.claim.json'),'alpha40_diagnostic_CPU')
    checks={};torch.set_num_threads(4)
    for d in (2,4,6):
        q,pt=parent(d);z=remap(pt['model'],q['alpha_low']);model=Alpha40MoE(d);model.load_state_dict(z)
        old=[float((q['alpha_low']+(.95-q['alpha_low'])*v.sigmoid()).float()) for n,v in pt['model'].items() if n.startswith('fusion_logits.')]
        assert old==model.alphas() and len(list(model.parameters()))=={2:12,4:21,6:30}[d]
        assert sum(p.numel() for p in model.linear.parameters())==5140
        model.set_alpha(.45);assert max(abs(a-.45) for a in model.alphas())<1e-7
        checks[str(d)]=dict(effective_parent_alpha_exactly_preserved=old,parameter_tensors=len(list(model.parameters())))
    previous.controls();base.cache_identity();CPU.mkdir();s.save(CPU/'checks.json',dict(identity(),passed=True,checks=checks,resources=resources.proof(),real_forward_calls=0,optimizer_steps=0,TEST_images_read=0),exclusive=True)
def run():
    assert not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==resources.GPU[2]
    slots=resources.slots();assert slots['total']==1 and slots['jobs'][0]['pid']==os.getpid() and slots['jobs'][0]['gpu_uuid']==resources.GPU[2]
    proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();claim(OUT.with_name(OUT.name+'.claim.json'),'fixed_TRAIN_VAL_alpha_sensitivity');OUT.mkdir()
    s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,resources.GPU[2],shared=True,memory_gib=2);torch.set_num_threads(4)
    train,val=base.load('train'),base.load('val');records={}
    for d in (2,4,6):
        q,pt=parent(d);model=Alpha40MoE(d).cuda();model.load_state_dict(remap(pt['model'],q['alpha_low']));model.eval();initial=s.digest(model.state_dict());old=importlib.import_module(q['module']);reference=getattr(old,q['class_name'])(d,.3).cuda();reference.load_state_dict(pt['model']);reference.eval()
        ti=np.array([int((train[1]==k).nonzero()[0]) for k in range(20)]);vi=np.array([int((val[1]==k).nonzero()[0]) for k in range(20)])
        x=base.encode(train[0][ti[:4]].cuda(),None,100)
        with torch.no_grad():
            p,c=reference(x);a,b=model(x);pe=float((p-a).abs().max());ce=float((c-b).abs().max());assert pe<=3e-6 and ce<=3e-6,(d,pe,ce)
        del reference;torch.cuda.empty_cache();points=[]
        for value in [None,*config()['alpha_probe_values']]:
            model.load_state_dict(remap(pt['model'],q['alpha_low']))
            if value is not None:model.set_alpha(value)
            row=dict(forced_alpha=value,effective_alphas=model.alphas(),sigmoid_derivative=[float((HIGH-LOW)*p.sigmoid()*(1-p.sigmoid())) for p in model.fusion_logits])
            for role,data,idx in [('train',train,ti),('val',val,vi)]:
                ps=[];stats=[]
                with torch.no_grad():
                    for off in range(0,20,4):
                        p,_=model(base.encode(data[0][idx[off:off+4]].cuda(),None,100));ps.append(p.cpu());stats.append(model.branch_statistics)
                row[role]=engine.metrics(data[1][idx].numpy(),torch.cat(ps).numpy());row[role+'_branch_statistics']=stats
            if value in (None,.45):
                model.eval();model.zero_grad(set_to_none=True)
                for off in range(0,20,4):
                    idx=ti[off:off+4];p,_=model(base.encode(train[0][idx].cuda(),None,100));loss=-p.clamp_min(1e-12).log().gather(1,train[1][idx].cuda()[:,None]).sum()/20;loss.backward()
                gradients={n:float(p.grad.double().norm()) for n,p in model.named_parameters()};assert all(np.isfinite(a) and a>0 for a in gradients.values());row['CE_gradient_norms']=gradients
            points.append(row)
        model.load_state_dict(remap(pt['model'],q['alpha_low']));assert s.digest(model.state_dict())==initial
        records[str(d)]=dict(parent=q,clean_coordinate_parity_probability=pe,clean_coordinate_parity_capture=ce,points=points,TRAIN_images=20,VAL_images=20,optimizer_steps=0,state_restored=True)
        del model;torch.cuda.empty_cache()
    previous.controls();s.save(OUT/'diagnosis.json',dict(identity(),passed=True,records=records,policy=policy,TEST_images_read=0,TEST_forward_calls=0,optimizer_steps=0,diagnostic_only_not_accuracy_estimate=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','run'],required=True);a=p.parse_args()
    try:globals()[a.phase]()
    except BaseException as e:
        root=CPU if a.phase=='cpu' else OUT;mark=root/'failure.json' if root.exists() else root.with_name(root.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
