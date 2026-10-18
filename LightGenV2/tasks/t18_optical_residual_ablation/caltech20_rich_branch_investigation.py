"""TRAIN/VAL-only branch usefulness, normalization and gradient investigation."""
import argparse,importlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_rich_alpha40_samehead as prior
import caltech20_rich_alpha40_diagnosis as parents
import caltech20_rich_fivegpu_resources as resources
from caltech20_rich_alpha40_namespace_repair import Alpha40MoE
from caltech20_rich_alpha40_model import remap,rms,features,energies
HERE,ROOT,s,engine,base=prior.HERE,prior.ROOT,prior.s,prior.engine,prior.base
FILES=('caltech20_rich_branch_investigation.py','caltech20_rich_fivegpu_resources.py','check_caltech20_rich_branch_investigation.py','CALTECH20_RICH_BRANCH_INVESTIGATION_PROTOCOL_20261018.md')
OUT=HERE/'runs/smoke/caltech20_rich_branch_gradient_investigation_20261018'
CPU=HERE/'runs/smoke/caltech20_rich_branch_gradient_cpu_20261018'
def sources():
    z=prior.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();z=sources();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
class ProbeMoE(Alpha40MoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho);self.force=None;self.full_rms_gradient=False
    def forward(self,amplitude,callback=None,ids=None):
        assert callback is None and ids is None,'Simulation diagnosis only';self.last={};self.trace=[];self.branch_statistics=[]
        intensity=self.capture(amplitude,'router');self.trace.append('router');energy=energies(intensity,self.router_masks);centered=energy-energy.mean(1,keepdim=True)
        dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1);weights=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8);self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights);self.trace.append(self.stages[k]);target=self.stages[k+1] if k<self.depth-1 else 'global';size=478 if target=='global' else 224
            optical=self.nonlinear(intensity,target);electronic=self.electronic[k](amplitude,size);re,ro=rms(electronic),rms(optical)
            if not self.full_rms_gradient:re,ro=re.detach(),ro.detach()
            a=(.4001+(.95-.4001)*self.fusion_logits[k].sigmoid()).float()
            if self.force is not None and self.force[k] is not None:a=torch.tensor(self.force[k],device=amplitude.device,dtype=torch.float32)
            en,on=electronic/re,optical/ro;cos=(en.detach()*on.detach()).mean((-2,-1));mixed=(1-a)*en+a*on;rm=rms(mixed)
            self.branch_statistics.append(dict(alpha=float(a.detach()),normalized_cosine=float(cos.mean()),electronic_rms=float(re.detach().mean()),optical_rms=float(ro.detach().mean()),nominal_electronic_to_optical=float((1-a.detach())/a.detach()) if float(a.detach()) else None))
            if not self.full_rms_gradient:rm=rm.detach()
            amplitude=re*mixed/rm
        f=features(amplitude);p=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors);self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return p,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)
def group(n):return 'optical' if n=='raw_router' or n.startswith('phase_sets.') else 'alpha' if n.startswith('fusion_logits.') else 'readout' if n.startswith('linear.') else 'stage_electronic'
def summarize(a,b,named):
    z={}
    for g in ('optical','stage_electronic','alpha','readout'):
        aa=torch.cat([x.detach().double().flatten() for x,(n,p) in zip(a,named) if group(n)==g]);bb=torch.cat([x.detach().double().flatten() for x,(n,p) in zip(b,named) if group(n)==g]);an=float(aa.norm());bn=float(bb.norm())
        z[g]=dict(first_norm=an,second_norm=bn,cosine=float((aa*bb).sum()/(aa.norm()*bb.norm()).clamp_min(1e-30)),second_to_first=bn/max(an,1e-30))
    return z
def gradients(model,data,idx,weights,q,mass,full):
    model.force=None;model.full_rms_gradient=full;model.eval();named=list(model.named_parameters());parameters=[p for n,p in named];cg=[torch.zeros_like(p) for p in parameters];kg=[torch.zeros_like(p) for p in parameters];cv=kv=0.
    for off in range(0,len(idx),4):
        ix=idx[off:off+4];p,_=model(base.encode(data[0][ix].cuda(),None,100));y=data[1][ix].cuda();factor=len(ix)/len(idx)
        ce=-(F.one_hot(y,20)*weights[None]*p.clamp_min(1e-12).log()).sum(1).mean()*factor;kd=prior.kd.KD_loss(p,torch.tensor(q[ix],device='cuda'),torch.tensor(mass[ix],device='cuda'),dict(temperature=4.))*factor
        a=torch.autograd.grad(ce,parameters,retain_graph=True);b=torch.autograd.grad(kd,parameters)
        for x,y,u,w in zip(cg,kg,a,b):x.add_(u);y.add_(w)
        cv+=float(ce.detach());kv+=float(kd.detach())
    assert all(torch.isfinite(g).all() for g in cg+kg)
    return cg,kg,dict(CE=cv,KD=kv,CE_vs_KD=summarize(cg,kg,named))
def probe(model,data,idx,force):
    model.force=force;model.full_rms_gradient=False;model.eval();ps=[];statistics=[]
    with torch.no_grad():
        for off in range(0,len(idx),4):
            ix=idx[off:off+4];p,_=model(base.encode(data[0][ix].cuda(),None,100));ps.append(p.cpu());statistics.append(model.branch_statistics)
    return dict(metrics=engine.metrics(data[1][idx].numpy(),torch.cat(ps).numpy()),branch_statistics=statistics)
def run(depth,gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu and gpu in resources.GPU.values();slots=resources.slots();assert slots['counts'][gpu]==1 and any(j['pid']==os.getpid() and j['gpu_uuid']==gpu for j in slots['jobs'])
    proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();identity();prior.controls();out=OUT/f'L{depth}';assert not out.exists();OUT.mkdir(exist_ok=True);claim(out.with_name(out.name+'.claim.json'),'TRAIN_VAL_branch_diagnosis');out.mkdir()
    s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    q,pt=parents.parent(depth);model=ProbeMoE(depth).cuda().eval();model.load_state_dict(remap(pt['model'],q['alpha_low']));before=s.digest(model.state_dict());data={r:base.load(r) for r in ('train','val')}
    ix={role:np.array([int(k) for c in range(20) for k in (d[1]==c).nonzero()[:5,0]]) for role,d in data.items()};assert all(len(x)==100 for x in ix.values())
    original=importlib.import_module(q['module']);reference=getattr(original,q['class_name'])(depth,.3).cuda().eval();reference.load_state_dict(pt['model'])
    with torch.no_grad():
        x=base.encode(data['train'][0][ix['train'][:4]].cuda(),None,100);a,b=model(x);c,d=reference(x);pe=float((a-c).abs().max());ce=float((b-d).abs().max());assert pe<=3e-6 and ce<=3e-6
        model.full_rms_gradient=True;e,f=model(x);assert torch.equal(a,e) and torch.equal(b,f),'Gradient policy must not alter clean forward'
    del reference;torch.cuda.empty_cache();records={}
    for label,forced in [('retained',None),('alpha85',.85),('alpha75',.75),('alpha60',.6),('alpha45',.45),('optical_only',1.),('electronic_only',0.)]:
        records[label]={role:probe(model,d,ix[role],None if forced is None else [forced]*depth) for role,d in data.items()}
    small=np.array([int((data['train'][1]==c).nonzero()[0]) for c in range(20)]);ablation={}
    for stage in range(depth):
        force=[None]*depth;force[stage]=1.;ablation[str(stage)]=probe(model,data['train'],small,force)
    qt,valid=prior.targets();counts=torch.tensor(base.manifest()['supports']['train'],device='cuda',dtype=torch.float32);weights=counts.sum()/(20*counts)
    g0,k0,r0=gradients(model,data['train'],small,weights,qt,valid,False);g1,k1,r1=gradients(model,data['train'],small,weights,qt,valid,True)
    assert s.digest(model.state_dict())==before and all(p.grad is None for p in model.parameters());prior.controls();identity()
    s.save(out/'diagnosis.json',dict(identity(),passed=True,depth=depth,parent=q,policy=policy,gpu_uuid=gpu,original_probability_maxabs=pe,original_capture_maxabs=ce,fullRMS_forward_exact=True,TRAIN_samples=100,VAL_samples=100,TRAIN_sample_ids=[base.manifest()['split_ids']['train'][i] for i in ix['train']],VAL_sample_ids=[base.manifest()['split_ids']['val'][i] for i in ix['val']],balanced_probe_not_full_accuracy=True,records=records,TRAIN_single_stage_electronic_ablation=ablation,gradient_detached_RMS=r0,gradient_full_RMS=r1,classification_gradient_direction_change=summarize(g0,g1,list(model.named_parameters())),teacher_valid_fraction=float(valid[small].mean()),optimizer_steps=0,TEST_images_read=0,TEST_forward_calls=0,state_unchanged=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth',type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:run(a.depth,a.gpu)
    except BaseException as e:
        out=OUT/f'L{a.depth}';mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
