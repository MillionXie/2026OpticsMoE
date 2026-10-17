"""One TRAIN-only scalar phase-coordinate derivative/probe; no optimizer."""
import argparse,json,math,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_equal60_cutmix as current
import caltech20_equal60_phase_origin_diagnosis as previous
m=current.prior;s=m.s;HERE,ROOT=m.HERE,m.ROOT
FILES=('caltech20_equal60_collective_phase_diagnosis.py','CALTECH20_EQUAL60_COLLECTIVE_PHASE_DIAGNOSIS_PROTOCOL_20261017.md')
OUT=HERE/'runs/smoke/caltech20_equal60_collective_phase_TRAIN_cpu_20261017'

def sources():
    src=current.sources()
    for name in FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();m.engine.verify_sources(src,commit)
    return dict(sources=src,git_commit=commit)

def toy():
    raw=torch.linspace(-.3,.5,15,dtype=torch.float64).requires_grad_();bias=torch.tensor(.17,dtype=torch.float64,requires_grad=True);phase=2*math.pi*raw.sigmoid();z=.7*torch.exp(1j*(phase+bias))+.3;loss=(z.abs().square()*torch.linspace(.5,1.5,15,dtype=torch.float64)).mean();gr,gb=torch.autograd.grad(loss,(raw,bias));jac=2*math.pi*raw.detach().sigmoid()*(1-raw.detach().sigmoid());aggregate=(gr/jac).sum();assert abs(float(gb-aggregate))<1e-14
    f=phase.detach().numpy();w=np.linspace(.5,1.5,15);a=float(bias.detach());eps=1e-6
    def value(t):return np.mean(np.abs(.7*np.exp(1j*(f+t))+.3)**2*w)
    fd=(value(a+eps)-value(a-eps))/(2*eps);assert abs(fd-float(gb))<1e-9
    return dict(passed=True,chainrule_maxabs=abs(float(gb-aggregate)),finite_difference_maxabs=abs(fd-float(gb)))

def resources():
    slots=m.base.slots();state=s.read(current.OUT/'status.json');allowed={q['pid'] for q in state.get('active',[])};assert {q['pid'] for q in slots['jobs']}<=allowed
    assert s.read(current.OUT/'metadata.json')['sources']==current.sources();return slots

def transport(state,slices,delta):
    result={k:v.clone() for k,v in state.items()}
    for stage,(lo,hi) in enumerate(slices):
        for i in range(lo,hi):
            name='phase_sets.'+str(i);v=state[name];q=torch.remainder(torch.sigmoid(v.double())+float(delta[stage])/(2*math.pi),1);assert q.min()>1e-8 and q.max()<1-1e-8;result[name]=torch.logit(q).to(v.dtype)
    return result

def score(model,state,data,indices):
    from resisc20_input_encoder import encode
    model.load_state_dict(state,strict=True);model.eval();probs=[]
    with torch.no_grad():
        for i in indices:probs.append(model(encode(data[0][i:i+1],None,100))[0].numpy())
    probs=np.concatenate(probs);y=data[1][indices].numpy();return dict(NLL=float(-np.log(np.maximum(probs[np.arange(len(y)),y],1e-12)).mean()),accuracy=float((probs.argmax(1)==y).mean()))

def diagnose(out):
    assert out==OUT and not out.exists() and not out.with_name(out.name+'.claim.json').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized();resources();proof=toy();source=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();m.engine.verify_sources(source,commit);meta=dict(sources=source,git_commit=commit,pid=os.getpid());s.save(out.with_name(out.name+'.claim.json'),meta,exclusive=True)
    torch.set_num_threads(4);data=m.base.load('train');indices=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);folder=m.OUT/'L6_rho03';fit=s.read(folder/'fit_result.json');assert fit['best_sha256']==s.sha(folder/'best.pt');pt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);state=pt['model'];model=m.base.RegularizedMoE(6,.3);model.load_state_dict(state,strict=True);original=s.digest(model.state_dict());retained=previous.evaluate(model,state,data,indices);retained.pop('probabilities');derivatives=[]
    for stage,(lo,hi) in enumerate(model.stage_slices):
        summation=0.;pixel=[]
        for i in range(lo,hi):
            raw=model.phase_sets[i];u=raw.detach().double().sigmoid();g=raw.grad.detach().double()/(2*math.pi*u*(1-u));summation+=float(g.sum());pixel.append(g.numpy().ravel())
        pixel=np.concatenate(pixel);norm=float(np.linalg.norm(pixel));derivatives.append(dict(stage=stage,kind=model.stages[stage],global_CE_derivative=summation,physical_pixel_gradient_norm=norm,pixels=len(pixel),uniform_direction_cosine=summation/(math.sqrt(len(pixel))*norm)))
    g=np.array([q['global_CE_derivative'] for q in derivatives]);assert np.isfinite(g).all() and np.max(np.abs(g))>0;descent=-g/np.max(np.abs(g))*(2*math.pi/256);probes={}
    for name,direction in [('descent_one_code',descent),('opposite_one_code',-descent)]:probes[name]=score(model,transport(state,model.stage_slices,direction),data,indices)
    model.load_state_dict(state,strict=True);assert s.digest(model.state_dict())==original and not torch.cuda.is_initialized() and sources()==source;resources();out.mkdir();result=dict(meta,passed=True,parent_PT_sha256=fit['best_sha256'],selected_epoch=pt['epoch'],independent_toy=proof,retained=retained,stage_derivatives=derivatives,descent_vector_radians=descent.tolist(),directional_probes=probes,CPU_models=1,TRAIN_forwards=60,TRAIN_backwards=20,optimizer_steps=0,VAL_images_read=0,TEST_images_read=0,saved_weights_unchanged=True,scope='TRAIN-only local collective phase-coordinate derivative, one fixed +/- one-code directional probe; no performance or generalization claim');s.save(out/'diagnosis.json',result,exclusive=True);print(json.dumps(dict(passed=True,sha256=s.sha(out/'diagnosis.json'),retained_NLL=retained['NLL'],derivatives=derivatives,probes=probes,TRAIN_forwards=60,TRAIN_backwards=20,optimizer_steps=0,TEST_images_read=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    try:diagnose(a.out)
    except BaseException as e:
        f=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not f.exists():s.save(f,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
