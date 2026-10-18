"""Preserved actual numerical policy plus independent raw TRAIN teacher scale."""
import types,numpy as np,torch
import caltech20_rich_currentKD46 as m
import check_caltech20_rich_phasecode_consistency as previous
import audit_caltech20_rich_currentKD46 as audit
ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);original_run=types.FunctionType(previous.run.__code__,ns,'same_independent_phase_and_retained_load_proof',previous.run.__defaults__,previous.run.__closure__)
def independent_teacher(depth):
    t=m.teacher_source;pt=torch.load(t.OUT/'selected_head.pt',map_location='cpu',weights_only=False)
    with np.load(t.OUT/'train_features.npz',allow_pickle=False) as a:f,y,ids=[a[k].copy() for k in ('features','labels','ids')]
    x=((f-pt['mean'])/pt['std']).astype(np.float32).astype(float);logits=x@pt['model']['weight'].double().numpy().T+pt['model']['bias'].double().numpy();parent,state,pred=m.incumbent(depth);sl=pred['train']['features'].astype(float)@state['model']['linear.weight'].double().numpy().T+state['model']['linear.bias'].double().numpy();teacher_std=(logits-logits.mean(1,keepdims=True)).std();student_std=(sl-sl.mean(1,keepdims=True)).std();tau=4*teacher_std/student_std;assert abs(tau-m.config()['teacher_temperature_by_depth'][str(depth)])<1e-10;z=logits/tau;p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);raw=np.exp(logits-logits.max(1,keepdims=True));raw/=raw.sum(1,keepdims=True);valid=((logits.argmax(1)==y)&(raw.max(1)>=.5)).astype(np.float32);q,v,info=m.teacher_data(depth);assert np.array_equal(q,p.astype(np.float32)) and np.array_equal(v,valid) and ids.tolist()==m.base.manifest()['split_ids']['train'];return dict(temperature=tau,teacher_centered_std=float(teacher_std),student_centered_std=float(student_std),target_SHA256=m.s.digest(q),gate_coverage=float(valid.mean()))
def run():
    original_run();proof=m.s.read(m.CPU/'checks.json');checks={str(d):independent_teacher(d) for d in (4,6)};rng=np.random.default_rng(17);z=torch.tensor(rng.normal(size=(5,20)),requires_grad=True);q=torch.tensor(rng.uniform(.1,1,(5,20)));q=q/q.sum(1,keepdim=True);p=torch.softmax(z,1);mass=torch.ones(5,dtype=torch.float64);loss=m.old.kd.KD_loss(p,q,mass,dict(temperature=4));g=torch.autograd.grad(loss,z)[0];expected=4*(torch.softmax(z.detach()/4,1)-q)/5;assert float((g-expected).abs().max())<1e-12;example=m.complete_gate_contract(dict(parent_capture_maxabs=1e-7,parent_clean_probability_maxabs=2e-7));assert example['parent_clean_capture_maxabs']==example['parent_capture_maxabs'];proof.update(independent_current_teacher_calibration=True,current_teacher_calibration=checks,analytic_KD_gradient=True,gate_producer_auditor_capture_key_contract_proved=True,numerical_thresholds_unchanged=True,L2_model_constructions=0);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
