"""Actual phase TV NumPy derivative and full-state EMA admission; zero forwards."""
import argparse,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_emastable as m

def independent_TV(state,coefficient):
    arrays={n:q.numpy().astype(np.float64) for n,q in state.items() if n=='raw_router' or n.startswith('phase_sets.')};count=2*len(arrays);value=0.;grads={}
    for name,raw in arrays.items():
        sig=1/(1+np.exp(-raw));phase=2*math.pi*sig;vertical=phase[1:]-phase[:-1];horizontal=phase[:,1:]-phase[:,:-1];value+=coefficient*(np.mean(vertical**2)+np.mean(horizontal**2))/count;gp=np.zeros_like(raw);gp[1:]+=2*vertical/vertical.size;gp[:-1]-=2*vertical/vertical.size;gp[:,1:]+=2*horizontal/horizontal.size;gp[:,:-1]-=2*horizontal/horizontal.size;grads[name]=coefficient*gp*(2*math.pi*sig*(1-sig))/count
    return value,grads

def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';p=m.parent();m.claim(out.with_name(out.name+'.claim.json'),'EMA999_restore_and_independent_phase_TV_CPU');torch.set_num_threads(4);old=torch.load(p['last'],map_location='cpu',weights_only=False);assert old['epoch']==100 and old['steps']==19700 and old['scheduler']['last_epoch']==100 and m.s.sha(p['last'])==p['last_sha256'];model=m.RegularizedMoE(6,.3);model.load_state_dict(old['model'],strict=True);contract=model.contract();assert contract['terminal_linear_parameters']==5140 and len(list(model.parameters()))==30;model.double();checks=[]
    for case in m.CASES:
        cfg=m.config(case);model.zero_grad(set_to_none=True);value=cfg['phase_smooth_weight']*model.phase_smoothness();value.backward();ref,grad=independent_TV(model.state_dict(),cfg['phase_smooth_weight']);assert abs(float(value.detach())-ref)<1e-10;err=max(float(np.max(np.abs(dict(model.named_parameters())[n].grad.numpy()-g))) for n,g in grad.items());assert err<1e-10 and all(np.isfinite(g).all() and np.linalg.norm(g)>0 for g in grad.values());checks.append(dict(case=case,phase_coefficient=cfg['phase_smooth_weight'],phase_TV_value=float(value.detach()),NumPy_gradient_maxabs=err,phase_tensors=len(grad)))
    before=np.linspace(-1,1,31);raw=np.linspace(.3,-.5,31);ema=torch.tensor(before,dtype=torch.float64);ema.mul_(.999).add_(torch.tensor(raw,dtype=torch.float64),alpha=.001);assert np.max(np.abs(ema.numpy()-(before*.999+raw*.001)))<1e-15;assert old['initial_digest']==m.s.read(m.BASE/'L6_rho03/fit_result.json')['initial_digest'];assert all(float(q['step'])==19700 for q in old['optimizer']['state'].values());out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,versions=m.versions(),parent=p,contract=contract,restored_raw_digest=m.s.digest(old['model']),restored_EMA_digest=m.s.digest(old['ema']),restored_optimizer_moment_digest=m.g.optimizer_moment_digest(old['optimizer']),analytical_phase_checks=checks,EMA_half_life_epochs=math.log(.5)/math.log(.999)/197,optical_model_forwards=0,optimizer_updates=0,TEST_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
