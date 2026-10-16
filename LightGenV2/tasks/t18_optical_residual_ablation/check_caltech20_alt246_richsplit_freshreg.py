"""Independent fresh initialization, TRAIN augmentation and KD math admission."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_freshreg as study
from audit_caltech20_alt246_richsplit_cachedkd import reference_kd
from audit_caltech20_alt246_richsplit_viewreg import reference_parameters,reference_pixels

def check(out):
    assert out==study.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';study.prerequisites();study.claim(out.with_name(out.name+'.claim.json'),'freshreg_CPU_independent_admission');torch.set_num_threads(4);teacher=study.k.teacher_data();model=study.initial_state();initial=study.s.digest(model.state_dict());contract=model.contract();assert len(list(model.named_parameters()))==30 and contract['terminal_linear_parameters']==5140 and all(.7<a<=.95 for a in model.alphas());checks=[]
    images=np.random.default_rng(17).integers(0,256,(20,150,150,3),dtype=np.uint8)
    for case in study.CASES:
        cfg=study.config(case)
        for epoch,batch,idx in ((1,0,np.array([np.where(teacher[1]==c)[0][0] for c in range(20)])),(10,0,np.arange(16)),(100,196,np.arange(len(teacher[1])-5,len(teacher[1])))):
            target,gate=study.k.kd_targets(teacher,idx,epoch,batch,cfg);q,h=reference_kd(teacher[0],teacher[1],idx,epoch,batch,cfg);assert np.array_equal(target,q) and np.array_equal(gate,h)
            params=study.v.view_parameters(len(idx),epoch,batch,cfg);reference=reference_parameters(len(idx),epoch,batch,cfg);assert np.array_equal(params,reference);pixels=study.v.transform(images[:len(idx)],params);independent=reference_pixels(images[:len(idx)],reference);assert np.array_equal(pixels,independent);checks.append(dict(case=case,epoch=epoch,n=len(idx),target_sha256=study.s.digest(target),gate_sha256=study.s.digest(gate),view_sha256=study.s.digest(params),pixel_sha256=study.s.digest(pixels)))
        class Dummy:
            def phase_smoothness(self):return torch.tensor(0.,dtype=torch.float64)
        logits=torch.linspace(-1,1,400,dtype=torch.float64).reshape(20,20).requires_grad_();y=torch.eye(20,dtype=torch.float64)*.95+.05/20;weights=torch.linspace(.5,2,20,dtype=torch.float64);target=torch.flip(logits.detach(),[1]).softmax(1);eligible=torch.arange(20)%2==0;p=logits.softmax(1);proof=dict(study.effective_config(cfg,1),capture_weight=0.,phase_smooth_weight=0.);loss=study.k.loss_core(Dummy(),p,torch.ones(20,dtype=torch.float64),y,weights,proof,target,eligible);loss.backward();T=cfg['kd_temperature'];expected=(p.detach()*(y*weights).sum(1,keepdim=True)-y*weights)/20+proof['kd_alpha']*T*((logits.detach()/T).softmax(1)-target)*eligible[:,None]/20;assert float((logits.grad-expected).abs().max())<1e-10
    toy=[dict(epoch=1,val=dict(accuracy=.8,nll=.9)),dict(epoch=2,val=dict(accuracy=.8,nll=.8)),dict(epoch=3,val=dict(accuracy=.79,nll=.1)),dict(epoch=4,val=dict(accuracy=.8,nll=.8))];assert study.selected_row(toy)['epoch']==2;assert study.s.digest(model.state_dict())==initial;out.mkdir();study.s.save(out/'checks.json',dict(study.identity(),passed=True,versions=study.versions(),initial_digest=initial,contract=contract,checks=checks,real_model_forwards=0,real_optimizer_updates=0,teacher_forwards=0,test_images_read=0,selection_tie_proof=True),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
