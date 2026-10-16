"""Independent analytic KD proof and exact restored inference/optimizer state."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_deepdistill as study

def check(out):
    assert out==study.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';p=study.admit();assert study.b.slots()['total']==0;study.claim(out.with_name(out.name+'.claim.json'),'KD_CPU_admission');torch.set_num_threads(4)
    cfg=study.config();last=torch.load(p['parent_last']['path'],map_location='cpu',weights_only=False);assert last['epoch']==120 and last['steps']==10320
    model=study.RegularizedMoE(6);model.load_state_dict(last['model'],strict=True);teacher=study.teacher_model(p,'cpu');td=study.s.digest(teacher.state_dict());assert teacher.contract()['main_depth']==2 and model.contract()['main_depth']==6 and model.contract()['terminal_linear_parameters']==5140
    phase=[];electronic=[]
    for n,v in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(v)
    optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=.01)]);optimizer.load_state_dict(last['optimizer']);assert study.g.optimizer_moment_digest(optimizer.state_dict())==study.g.optimizer_moment_digest(last['optimizer']) and all(float(v['step'])==10320 for v in optimizer.state.values())
    torch.manual_seed(17);logits=torch.randn(20,20,dtype=torch.float64,requires_grad=True);q=torch.softmax(torch.randn(20,20,dtype=torch.float64),1);y=torch.eye(20,dtype=torch.float64);q[:5]=.01;q[torch.arange(5),torch.arange(5)]=.81;prob=logits.softmax(1);loss,gate=study.kd(prob,q,y,cfg);loss.backward();T=cfg['distillation_temperature'];qt=q.numpy()**(1/T);qt/=qt.sum(1,keepdims=True);st=np.exp(logits.detach().numpy()/T);st/=st.sum(1,keepdims=True);G=(q.argmax(1)==y.argmax(1)).numpy();expected=(st-qt)*G[:,None]*T/20
    assert np.allclose(expected,logits.grad.numpy(),atol=1e-10,rtol=0) and G.any() and (~G).any() and all(v.grad is None for v in teacher.parameters()) and study.s.digest(teacher.state_dict())==td
    assert model.contract()==study.b.HybridMoE(6,.3).contract();out.mkdir();study.s.save(out/'checks.json',dict(study.identity(),passed=True,KD_logit_gradient_maxabs=float(np.abs(expected-logits.grad.numpy()).max()),teacher_frozen=True,teacher_digest=td,optimizer_restored_steps=10320,optimizer_updates=0,real_model_forwards=0,test_arrays_read=0,inference_parameters_added=0,student_contract=model.contract()),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        path=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not path.exists():study.s.save(path,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
