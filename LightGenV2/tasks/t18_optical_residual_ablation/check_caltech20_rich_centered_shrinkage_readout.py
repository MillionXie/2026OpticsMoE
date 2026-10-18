"""Real TRAIN/VAL float32 gauge proof, unchanged probability threshold."""
import types
import numpy as np
import torch
import caltech20_rich_centered_shrinkage_readout as m
import check_caltech20_rich_shrinkage_readout as previous
def run():
    before=m.s.sha(m.original.OUT/'fit_failure.json');ex=m.s.read(m.original.OUT/'fit_exit.json');assert ex['returncode']==1 and ex['actual_exit'] and not m.Path('/proc',str(ex['pid']),'cmdline').exists()
    assert not (m.original.OUT/'test_exit.json').exists() and not any((m.original.OUT/f'L{d}/candidate.pt').exists() for d in (2,4,6))
    ns=dict(vars(previous));ns['m']=m;f=types.FunctionType(previous.run.__code__,ns,'same_Gaussian_CPU_math',previous.run.__defaults__,previous.run.__closure__);f();torch.set_num_threads(4)
    parent=m.parents()['2'];pt=torch.load(parent['path'],map_location='cpu',weights_only=False);arrays={r:m.load(m.Path(parent['features'][r])) for r in ('train','val')};x=arrays['train']['features'].astype(float);mean,cov,count=m.train_statistics(x,arrays['train']['labels']);old=x@pt['model']['linear.weight'].double().numpy().T+pt['model']['linear.bias'].double().numpy();sd=float((old-old.mean(1,keepdims=True)).std());records=[]
    def soft(a):p=np.exp(a-a.max(1,keepdims=True));return p/p.sum(1,keepdims=True)
    for beta in m.config()['shrinkage_grid']:
        w,b=m.discriminant(mean,cov,count,beta);z=x@w.T+b;scale=sd/float((z-z.mean(1,keepdims=True)).std());w*=scale;b*=scale;wc=w-w.mean(0,keepdims=True);bc=b-b.mean()
        for role,a in arrays.items():
            xx=a['features'];raw=torch.softmax(torch.nn.functional.linear(torch.tensor(xx,dtype=torch.float32),torch.tensor(w,dtype=torch.float32),torch.tensor(b,dtype=torch.float32)),1).numpy();raw64=soft(xx.astype(float)@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float));new,error=m.probabilities(xx,wc,bc);invariance=float(np.max(np.abs(soft(xx.astype(float)@w.T+b)-soft(xx.astype(float)@wc.T+bc))));assert invariance<1e-10
            records.append(dict(beta=beta,role=role,original_float32_error=float(np.max(np.abs(raw-raw64))),centered_float32_error=error,real_arithmetic_gauge_invariance_maxabs=invariance))
    assert m.s.sha(m.original.OUT/'fit_failure.json')==before;q=m.s.read(m.CPU/'checks.json');q.update(real_role_centering_proof_passed=True,real_model_forwards=24,real_optical_forwards=0,cached_TRAIN_VAL_head_forward_count=24,TEST_images_read=0,original_failure_sha256=before,original_TEST_calls=0,records=records);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
