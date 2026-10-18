"""Frozen optical state, exact four trainable tensors and analytic loss proofs."""
import numpy as np,torch
import caltech20_rich_postCCD_electronic_adapt as m
import audit_caltech20_rich_postCCD_electronic_adapt as audit
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'post_last_CCD_electronic_CPU');cfg=m.config();assert cfg['electronic_epochs']==cfg['maximum_new_TEST_models_per_depth']==5 and cfg['microbatch_size']==8 and cfg['alpha_restart'] is None and cfg['CCD_gain_noise']==.1 and cfg['CCD_read_noise']==.01
    rec=m.s.read(m.closure.OUT/'test_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==m.s.sha(m.closure.OUT/'test_science/audit.json') and m.s.read(m.closure.OUT/'test_science/audit.json')['passed']
    for d in (4,6):assert m.s.read(m.closure.OUT/f'L{d}/test_result.json')['metrics']['accuracy']<m.metrics(m.parent(d)[2]['test']['labels'],m.parent(d)[2]['test']['probabilities'])['accuracy']
    for n in ('fit','test','gate','gradients','mixed','CE','KL','JS','initialize','ready','rates','configure','identity'):f=getattr(m,n);assert not unresolved(f.__code__,f.__globals__)
    for n in ('run','verify_fit','verify_test','loss_parts'):f=getattr(audit,n);assert not unresolved(f.__code__,f.__globals__)
    rg=np.random.default_rng(17);logits=torch.tensor(rg.normal(size=(5,20)),requires_grad=True);teacher=rg.uniform(.1,1,(5,20));teacher/=teacher.sum(1,keepdims=True);q=torch.tensor(teacher);temperature=2.;loss=m.KL(torch.softmax(logits,1),q,temperature);g=torch.autograd.grad(loss,logits)[0].numpy();z=logits.detach().numpy()/temperature;p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);assert np.max(np.abs(g-temperature*(p-teacher)/5))<1e-12
    a=rg.uniform(.1,1,(5,20));a/=a.sum(1,keepdims=True);b=rg.uniform(.1,1,(5,20));b/=b.sum(1,keepdims=True);aa=torch.tensor(a,requires_grad=True);bb=torch.tensor(b,requires_grad=True);v=m.JS(aa,bb);ga,gb=torch.autograd.grad(v,(aa,bb));mid=(a+b)/2;assert np.max(np.abs(ga.numpy()-.5*np.log(a/mid)/5))<1e-12 and np.max(np.abs(gb.numpy()-.5*np.log(b/mid)/5))<1e-12
    w=torch.tensor(rg.uniform(.5,2,20));y=torch.tensor(np.eye(20)[[0,2,7,13,19]]*.95+.05/20);z=torch.tensor(rg.normal(size=(5,20)),requires_grad=True);p=torch.softmax(z,1);loss=m.CE(p,y,w);g=torch.autograd.grad(loss,z)[0].numpy();target=y.numpy()*w.numpy();expected=(p.detach().numpy()*target.sum(1,keepdims=True)-target)/5;assert np.max(np.abs(g-expected))<1e-12
    checks={}
    for d in (2,4,6):
        model=m.initialize(d);train=[n for n,p in model.named_parameters() if p.requires_grad];assert set(train)==set(m.names(d)) and sum(p.numel() for p in model.parameters() if p.requires_grad)==5150 and sum(p.numel() for p in model.linear.parameters())==5140 and len(train)==4;assert m.s.digest(model.state_dict())==m.s.digest(m.initial_state(d));cfg=m.config();model.context(17,5,cfg);expected=np.random.default_rng(17).standard_normal((5,2,32,32)).astype(np.float32);assert model.noise_sha==m.s.digest(expected) and torch.equal(model.CCD_noise,torch.tensor(expected)) and not model.noisy;checks[str(d)]=dict(parent_SHA256=m.parent(d)[0]['sha256'],initial_physical_alpha=model.alphas(),trainable_names=train,trainable_parameters=5150,all_optical_and_upstream_frozen=True)
    m.closure.diagnosis.closure.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,actual_four_tensor_trainability_proved=True,independent_CE_JS_KL_derivatives=True,deterministic_last_CCD_noise_array=True,five_gpu_guard=m.resources.proof(),real_CPU_model_constructions=3,real_model_forward_calls=0,optimizer_steps=0,TRAIN_images_read=0,VAL_images_read=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
