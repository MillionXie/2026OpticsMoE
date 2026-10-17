"""Physical float32 bound containment plus complete actual CPU contract."""
import types
import numpy as np
import torch
import caltech20_rich_projected_alpha_contained as m
import check_caltech20_rich_alpha40_rebalance as original
from check_caltech20_rich_alpha40_fresh246 import unresolved
import audit_caltech20_rich_projected_alpha_contained as auditor
namespace=dict(vars(original));namespace['m']=m;run=types.FunctionType(original.run.__code__,namespace,'run')
if __name__=='__main__':
    try:
        assert not m.prior.OUT.exists() and not m.prior.OUT.with_name(m.prior.OUT.name+'.dispatch_claim.json').exists()
        for name in ('initialize','gate','fit','campaign','ready','identity','gradients','incumbent'):
            f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
        assert not unresolved(auditor.run.__code__,auditor.run.__globals__)
        model=m.initialize(2,True)
        with torch.no_grad():model.fusion_logits[0].fill_(.1);model.fusion_logits[1].fill_(1.1)
        model.project();assert all(.4001<a<.95 for a in model.alphas()) and float(model.fusion_logits[0])==m.ALPHA_MIN
        # Match actual projected Adam including inward recovery from both bounds.
        p=torch.nn.Parameter(torch.tensor([m.ALPHA_MIN,.95],dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.1,weight_decay=0);initial=p.detach().clone();p.grad=torch.tensor([1.,-1.],dtype=torch.float64);opt.step();p.data.clamp_(m.ALPHA_MIN,.95);assert torch.equal(p,(initial-.1*p.grad/(p.grad.abs()+1e-8)).clamp(m.ALPHA_MIN,.95))
        p.grad=torch.tensor([-4.,4.],dtype=torch.float64);before=p.detach().clone();st=opt.state[p];e=.9*st['exp_avg']+.1*p.grad;v=.999*st['exp_avg_sq']+.001*p.grad.square();expected=before-.1*(e/(1-.9**2))/(v.sqrt()/np.sqrt(1-.999**2)+1e-8);opt.step();p.data.clamp_(m.ALPHA_MIN,.95);assert torch.allclose(p,expected.clamp(m.ALPHA_MIN,.95),atol=1e-14,rtol=0) and p[0]>m.ALPHA_MIN and p[1]<.95
        for depth in (2,4,6):
            q,pt=m.diagnosis.parent(depth);model=m.Alpha40MoE(depth);model.load_state_dict(m.remap(pt['model'],q['alpha_low']));old=[float((q['alpha_low']+(.95-q['alpha_low'])*v.sigmoid()).float()) for n,v in pt['model'].items() if n.startswith('fusion_logits.')];assert old==model.alphas()
        assert abs(m.alpha_schedule(1)-.45)<1e-14 and abs(m.alpha_schedule(10)-.65)<1e-14
        run();proof=m.s.read(m.CPU/'checks.json');proof.update(all_kernel_and_audit_global_symbols_resolved=True,projection_analytic_proof=dict(passed=True,analytic_projected_AdamW_steps=2,recovers_from_both_bounds=True,actual_model_float32_strict_range_checked=True,parameter_min=m.ALPHA_MIN,all_parent_effective_alphas_preserved=True),original328_GPU_unlaunched=True,new_inference_parameters=0);m.s.save(m.CPU/'checks.json',proof)
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
