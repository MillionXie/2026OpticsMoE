"""One TRAIN-only gradient diagnosis; active fresh240 producers remain frozen."""
import argparse,hashlib,io,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_freshreg as fresh

HERE,ROOT,s,engine=fresh.HERE,fresh.ROOT,fresh.s,fresh.engine
FILES=('caltech20_richsplit_objective_diagnosis.py','check_caltech20_richsplit_objective_diagnosis.py','CALTECH20_RICHSPLIT_OBJECTIVE_DIAGNOSIS_PROTOCOL_20261017.md')
CPU=HERE/'runs/smoke/caltech20_richsplit_objective_diagnosis_cpu_20261017'
OUT=HERE/'runs/smoke/caltech20_richsplit_objective_diagnosis_train_20261017'
GPU=fresh.b.GPU[2]

def sources():
    q=fresh.sources()
    for n in FILES:q[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return q

def identity():
    q=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(q,commit);return dict(sources=q,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def guard(actual,own_pid,parents):
    expected={int(own_pid):GPU,**{int(pid):gpu for pid,gpu in parents.items()}}
    assert len(expected)==3 and set(parents.values())=={fresh.b.GPU[4],fresh.b.GPU[5]} and {q['pid']:q['gpu_uuid'] for q in actual}==expected and len(actual)==3
    assert all(q['gpu_uuid']!=fresh.b.FORBIDDEN for q in actual);return True

def parents():
    q={}
    for case,card in [('regularized',5),('regularized_kd',4)]:
        root=fresh.expected_root(case);meta=s.read(root/'fit/metadata.json');assert meta['sources']==fresh.sources() and meta['config']==fresh.config(case);engine.verify_sources(meta['sources'],meta['git_commit']);p=Path('/proc',str(meta['pid']),'cmdline');assert p.exists() and str(root/'fit').encode() in p.read_bytes() and meta['gpu_uuid']==fresh.b.GPU[card];q[meta['pid']]=meta['gpu_uuid']
    return q

def group(name):
    if name=='raw_router' or name.startswith('phase_sets.'):return 'optical_phase'
    if name=='linear.weight' or name=='linear.bias':return 'terminal_linear'
    if 'alpha' in name:return 'learned_alpha'
    return 'middle_electronic'

def diagnose(out):
    assert out==OUT and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU;proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();guard(fresh.b.slots()['jobs'],os.getpid(),parents());claim(out.with_name(out.name+'.claim.json'),'TRAIN_only_objective_gradient_diagnosis');out.mkdir();s.gpu_budget.ALLOWED.update(fresh.b.GPU.values());policy=s.gpu_budget.configure(torch,GPU,shared=True,memory_gib=2);torch.set_num_threads(4);cfg=fresh.config('regularized_kd');data=fresh.b.load('train');teacher=fresh.k.teacher_data();weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda();records=[]
    targets=[('original_selected86',fresh.BASE/'L6_rho03/best.pt','model'),('fresh_KD_current_lastEMA',fresh.expected_root('regularized_kd')/'fit/last.pt','ema')]
    for label,path,state_key in targets:
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);assert checkpoint['sources']==(fresh.b.sources() if label.startswith('original') else fresh.sources());epoch=int(checkpoint['epoch'])
        if label.startswith('original'):assert sha==s.read(fresh.BASE/'L6_rho03/fit_result.json')['best_sha256'] and epoch==86
        else:assert checkpoint['config']==cfg and epoch>=30 and checkpoint['steps']==197*epoch
        state=checkpoint[state_key];model=fresh.RegularizedMoE(6,.3).cuda();model.load_state_dict(state,strict=True);before=s.digest(model.state_dict());named=list(model.named_parameters());groups={n:group(n) for n,_ in named};assert len(named)==30;macros=[]
        for number,classes in enumerate([range(16),range(4,20)]):
            idx=np.asarray([int((data[1]==c).nonzero()[number]) for c in classes]);plan_epoch=30;_,theta=engine.epoch_plan(plan_epoch,len(data[1]),cfg);x,y,mask,hashes=fresh.mixed(data,idx,theta,plan_epoch,number,cfg,'cuda');target,eligible=fresh.k.kd_targets(teacher,idx,plan_epoch,number,cfg);target=torch.as_tensor(target,device='cuda');eligible=torch.as_tensor(eligible,device='cuda');assert eligible.any();grads={kind:{n:np.zeros(p.shape,np.float64) for n,p in named} for kind in ('CE','KD','capture','TV')};values={kind:0. for kind in grads};model.train()
            for off in range(0,16,4):
                model.linear.mask=mask[off:off+4];p,c=model(x[off:off+4]);T=cfg['kd_temperature'];logq=torch.nn.functional.log_softmax(p.clamp_min(1e-12).log()/T,1);q=target[off:off+4];objectives=dict(CE=-(y[off:off+4]*weights[None]*p.clamp_min(1e-12).log()).sum(1).mean(),KD=cfg['kd_alpha']*T*T*((q*(q.clamp_min(1e-12).log()-logq)).sum(1)*eligible[off:off+4]).mean(),capture=-cfg['capture_weight']*c.clamp_min(1e-12).log().mean())
                for j,(kind,value) in enumerate(objectives.items()):
                    gradient=torch.autograd.grad(value,[p for _,p in named],allow_unused=True,retain_graph=j<2);values[kind]+=float(value.detach())/4
                    for (name,_),a in zip(named,gradient):
                        if a is not None:grads[kind][name]+=a.detach().cpu().numpy().astype(np.float64)/4
            model.linear.mask=None;tv=cfg['phase_smooth_weight']*model.phase_smoothness();gradient=torch.autograd.grad(tv,[p for _,p in named],allow_unused=True);values['TV']=float(tv.detach())
            for (name,_),a in zip(named,gradient):
                if a is not None:grads['TV'][name]=a.detach().cpu().numpy().astype(np.float64)
            result={}
            for group_name in sorted(set(groups.values())):
                vectors={kind:np.concatenate([g[n].ravel() for n,_ in named if groups[n]==group_name]) for kind,g in grads.items()};norms={kind:float(np.linalg.norm(v)) for kind,v in vectors.items()};ce=vectors['CE'];den=norms['CE'];assert den>0 and all(np.isfinite(v).all() for v in vectors.values());result[group_name]=dict(norms=norms,ratio_to_CE={kind:norms[kind]/den for kind in ('KD','capture','TV')},cosine_with_CE={kind:float(ce@vectors[kind]/(den*norms[kind])) if norms[kind]>0 else None for kind in ('KD','capture','TV')})
            macros.append(dict(macro=number,samples=16,classes=list(classes),teacher_eligible=int(eligible.sum()),training_plan_epoch=plan_epoch,hashes=hashes,weighted_objective_values=values,groups=result))
        assert s.digest(model.state_dict())==before==s.digest(state) and all(p.grad is None for _,p in named);records.append(dict(parent=label,path=str(path),loaded_checkpoint_sha256=sha,actual_checkpoint_epoch=epoch,state_key=state_key,model_digest_before=before,model_digest_after=s.digest(model.state_dict()),contract=model.contract(),macros=macros));del model;torch.cuda.empty_cache()
    guard(fresh.b.slots()['jobs'],os.getpid(),parents());s.save(out/'diagnosis.json',dict(identity(),passed=True,policy=policy,records=records,optimizer_updates=0,teacher_forwards=0,student_forward_batches=16,real_TRAIN_inputs=64,VAL_images_read=0,TEST_images_read=0,interpretation_limit='two fixed TRAIN macros, no full-dataset causality or heldout improvement proof'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:diagnose(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
