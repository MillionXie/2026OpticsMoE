"""Fixed TRAIN-only optical phase-origin diagnostic; no optimizer or TEST."""
import argparse,json,math,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_equal60_objective as m

HERE,ROOT,s=m.HERE,m.ROOT,m.s
FILES=('caltech20_equal60_phase_origin_diagnosis.py','CALTECH20_EQUAL60_PHASE_ORIGIN_DIAGNOSIS_PROTOCOL_20261017.md')
OUT=HERE/'runs/smoke/caltech20_equal60_phase_origin_TRAIN_cpu_20261017'

def sources():
    src=m.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();m.engine.verify_sources(src,commit)
    return dict(sources=src,git_commit=commit)

def shift(state):
    changed={k:v.clone() for k,v in state.items()}
    for name,value in state.items():
        if not name.startswith('phase_sets.'):continue
        phase=torch.sigmoid(value.double());fraction=torch.remainder(phase+.25,1)
        assert fraction.min()>1e-8 and fraction.max()<1-1e-8
        changed[name]=torch.logit(fraction).to(value.dtype)
    assert torch.equal(changed['raw_router'],state['raw_router'])
    assert all(torch.equal(changed[k],v) for k,v in state.items() if not k.startswith('phase_sets.'))
    return changed

def toy():
    raw=np.array([-.71,-.18,.09,.51],np.float64)
    f=1/(1+np.exp(-raw));moved=shift({'phase_sets.0':torch.from_numpy(raw),'raw_router':torch.tensor([.1])})['phase_sets.0'].numpy()
    expected=np.remainder(f+.25,1)
    assert np.max(np.abs(1/(1+np.exp(-moved))-expected))<1e-14
    original=np.floor(f*256).astype(np.int64)%256;new=np.floor(expected*256).astype(np.int64)%256
    assert np.array_equal(new,(original+64)%256)
    rho=.3;before=(1-rho)*np.exp(2j*np.pi*f)+rho;after=(1-rho)*np.exp(2j*np.pi*expected)+rho
    assert np.max(np.abs(after-((1-rho)*np.exp(2j*np.pi*f)*1j+rho)))<1e-14
    return dict(passed=True,phase_shift_radians=math.pi/2,phase_code_shift=64,analytic_maxabs=float(np.max(np.abs(1/(1+np.exp(-moved))-expected))),no_claim_of_residual_function_equivalence=True)

def modulation(state,rho):
    values=[]
    for name,raw in state.items():
        if name.startswith('phase_sets.'):
            phase=2*math.pi*torch.sigmoid(raw.double());values.append(((1-rho)*torch.exp(1j*phase)+rho).abs().numpy().ravel())
    amp=np.concatenate(values)
    return dict(mean_amplitude=float(amp.mean()),amplitude_CV=float(amp.std()/amp.mean()),minimum_amplitude=float(amp.min()))

def evaluate(model,state,data,indices):
    model.load_state_dict(state,strict=True);model.eval();model.zero_grad(set_to_none=True);probs=[];loss=0.;captures=[]
    for index in indices:
        # Use the exact protected TRAIN encoder, clean input, no new augmentation.
        from resisc20_input_encoder import encode
        x=encode(data[0][index:index+1],None,100);p,c=model(x);value=-p[0,int(data[1][index])].clamp_min(1e-12).log()/len(indices);value.backward();loss+=float(value.detach());probs.append(p.detach().numpy());captures.append(float(c.detach()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norms={name:float(torch.linalg.vector_norm(p.grad.double())) for name,p in model.named_parameters()}
    assert all(q>0 for q in norms.values())
    p=np.concatenate(probs);labels=data[1][indices].numpy();phase=sum(q*q for name,q in norms.items() if name.startswith('phase_sets.'))**.5
    return dict(n=len(indices),NLL=loss,accuracy=float((p.argmax(1)==labels).mean()),capture_mean=float(np.mean(captures)),phase_CE_gradient_norm=phase,gradient_norms=norms,probabilities=p)

def diagnose(out):
    assert out==OUT and not out.exists() and not out.with_name(out.name+'.claim.json').exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and m.base.slots()['total']==0
    assert s.read(m.OUT/'status.json')['state']=='objective10_science_and_reviewed_tests_complete'
    for root in (m.BASE,m.OUT):
        proof=s.read(root/'audit_completed.json');assert proof['actual_audit_returncode']==0 and proof['audit_sha256']==s.sha(root/'science/audit.json') and s.read(root/'science/audit.json')['passed']
    controls=m.admit();metadata=dict(identity(),pid=os.getpid(),purpose='one fixed quarter-wave TRAIN diagnostic, not a new model/training run');s.save(out.with_name(out.name+'.claim.json'),metadata,exclusive=True)
    torch.set_num_threads(4);proof=toy();data=m.base.load('train');indices=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);results={};forwards=0
    for key,root,rho in [('L6_rho03',m.OUT,.3),('L6_rho0',m.BASE,0.)]:
        folder=root/key;fit=s.read(folder/'fit_result.json');assert fit['best_sha256']==s.sha(folder/'best.pt');pt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);state=pt['model'];model=m.base.RegularizedMoE(6,rho);model.load_state_dict(state,strict=True);original=s.digest(model.state_dict());altered=shift(state);conditions={};saved=[]
        for label,weights in [('retained',state),('quarter_wave',altered)]:
            q=evaluate(model,weights,data,indices);forwards+=len(indices);saved.append(q.pop('probabilities'));q['modulation']=modulation(weights,rho);conditions[label]=q
        model.load_state_dict(state,strict=True);assert s.digest(model.state_dict())==original;delta=float(np.max(np.abs(saved[1]-saved[0])));results[key]=dict(selected_epoch=pt['epoch'],checkpoint_sha256=fit['best_sha256'],unchanged_saved_weights=True,restored_model_digest=original,conditions=conditions,probability_change_maxabs=delta,quantized_control_gauge_actual_error=delta if rho==0 else None)
        del model,pt,state,altered
    assert forwards==80 and not torch.cuda.is_initialized() and m.admit()==controls
    assert sources()==metadata['sources'];out.mkdir();result=dict(metadata,passed=True,toy=proof,results=results,scope='TRAIN first sample per20labels, fixed positive quarter-wave shift only; clean CE gradient diagnostic, not augmented optimizer/full scientific acceptance or a performance improvement',real_CPU_models=2,real_TRAIN_forwards=80,real_TRAIN_backwards=80,optimizer_steps=0,VAL_images_read=0,TEST_images_read=0,saved_weights_changed=False)
    s.save(out/'diagnosis.json',result,exclusive=True);print(json.dumps(dict(passed=True,sha256=s.sha(out/'diagnosis.json'),results=results,TRAIN_forwards=80,optimizer_steps=0,TEST_images_read=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    try:diagnose(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
