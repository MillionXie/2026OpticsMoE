"""One-code bounded rotation, deterministic macros and exact parent load."""
import numpy as np,torch
import caltech20_rich_phasecode_consistency as m
import audit_caltech20_rich_phasecode_consistency as audit
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();torch.set_num_threads(4);m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'phasecode_consistency_CPU');cfg=m.config();assert cfg['phase_jitter_sigma_codes']==1 and cfg['phase_jitter_augmented_view_only'] and cfg['alpha_restart'] is None and not cfg['reset_trained_weights'];checks={}
    for name in ('fit','gradients','gradient_kernel','mixed','gate','numeric_gate_kernel','initialize','ready','prototype_data','target_data','campaign','incumbent'):
        f=getattr(m,name);f=getattr(f,'__wrapped__',f);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
    for name in ('run','prototype_run','base_run','inherited','kernel'):
        f=getattr(audit,name);assert not unresolved(f.__code__,f.__globals__)
    rng=np.random.default_rng(17);phase=rng.uniform(-np.pi,np.pi,100);delta=rng.uniform(-2,2,100)/256*2*np.pi;z=torch.tensor(np.exp(1j*phase),dtype=torch.complex128)*torch.exp(1j*torch.tensor(delta));assert np.max(np.abs(z.numpy()-np.exp(1j*(phase+delta))))<1e-12 and np.max(np.abs(z.abs().numpy()-1))<1e-12;expected=np.floor(np.mod(phase+delta,2*np.pi)/(2*np.pi)*256);actual=torch.floor(torch.remainder(torch.angle(z),2*np.pi)/(2*np.pi)*256).numpy();assert np.array_equal(expected,actual)
    for d in (4,6):
        model=m.initialize(d,False);q,pt,_=m.incumbent(d);assert m.s.digest(model.state_dict())==m.s.digest(m.remap(pt['model'],q['alpha_low']));assert len(list(model.parameters()))=={4:21,6:30}[d] and model.full_rms_gradient;before=m.s.digest(model.state_dict());model.context(17,5,cfg);noise=np.clip(np.random.default_rng(np.random.SeedSequence([17,20261018,421])).standard_normal((d+1,1,16,16)).astype(np.float32),-2,2);assert model.phase_noise_SHA256==m.s.digest(noise) and np.array_equal(noise,m.phase_noise(d,17,cfg)) and float(model.phase_shift.abs().sub(1).abs().max())<2e-6 and m.s.digest(model.state_dict())==before;assert sum(p.numel() for p in model.linear.parameters())==5140;m.prototype_data(d);checks[str(d)]=dict(parent_SHA256=q['sha256'],initial_digest=before,phase_noise_SHA256=model.phase_noise_SHA256,initial_alpha=model.alphas())
    m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,phase_rotation_and_RNG_proof=True,no_inference_parameters_added=True,five_gpu_guard=m.resources.proof(),real_model_constructions=2,real_model_forward_calls=0,optimizer_steps=0,TRAIN_images_read=0,VAL_images_read=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
