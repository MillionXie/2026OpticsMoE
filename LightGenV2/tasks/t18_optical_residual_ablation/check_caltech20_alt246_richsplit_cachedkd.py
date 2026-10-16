"""CPU restore and independent cached-teacher/KL admission; no real forwards."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_cachedkd as study
from audit_caltech20_alt246_richsplit_cachedkd import reference_kd

def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and out==study.CPU and not out.exists();study.base_ready();study.claim(out.with_name(out.name+'.claim.json'),'cached_teacher_CPU_admission');torch.set_num_threads(4);cfg=study.config();parent=study.parent_info(study.BASE,6);info=study.teacher_info();data=study.teacher_data();last=torch.load(parent['last'],map_location='cpu',weights_only=False);model=study.RegularizedMoE(6);model.load_state_dict(last['model'],strict=True);assert last['epoch']==100 and last['steps']==19700 and model.contract()['terminal_linear_parameters']==5140 and all(.7<v<=.95 for v in model.alphas());checks=[]
    for epoch,batch,indices in ((101,0,np.array([np.where(data[1]==k)[0][0] for k in range(20)])),(101,0,np.arange(16)),(120,196,np.arange(len(data[1])-5,len(data[1])))):
        a,g=study.kd_targets(data,indices,epoch,batch,cfg);b,h=reference_kd(data[0],data[1],indices,epoch,batch,cfg);assert np.array_equal(a,b) and np.array_equal(g,h);checks.append(dict(epoch=epoch,batch=batch,count=len(indices),eligible=int(g.sum()),target_sha256=study.s.digest(a),gate_sha256=study.s.digest(g)))
    class Dummy:
        def phase_smoothness(self):return torch.tensor(0.,dtype=torch.float64)
    logits=torch.linspace(-1,1,400,dtype=torch.float64).reshape(20,20).requires_grad_();y=torch.eye(20,dtype=torch.float64)*.95+.05/20;weights=torch.linspace(.5,2,20,dtype=torch.float64);target=torch.flip(logits.detach(),[1]).softmax(1);eligible=(torch.arange(20)%2==0);p=logits.softmax(1);proof_cfg=dict(cfg,capture_weight=0.,phase_smooth_weight=0.);loss=study.loss_core(Dummy(),p,torch.ones(20,dtype=torch.float64),y,weights,proof_cfg,target,eligible);loss.backward();T=cfg['kd_temperature'];expected=(p.detach()*(y*weights).sum(1,keepdim=True)-y*weights)/20+cfg['kd_alpha']*T*((logits.detach()/T).softmax(1)-target)*eligible[:,None]/20;err=float((logits.grad-expected).abs().max());assert err<1e-10 and torch.isfinite(logits.grad).all() and logits.grad.norm()>0
    out.mkdir();study.s.save(out/'checks.json',dict(study.identity(),passed=True,versions=study.versions(),teacher=info,teacher_TRAIN_correct_confident_fraction=float(((data[0].argmax(1)==data[1])&(data[0].max(1)>=.5)).mean()),cached_target_math=checks,KL_and_balanced_CE_gradient_maxabs=err,parent_last_sha256=study.s.sha(parent['last']),parent_raw_digest=study.s.digest(last['model']),parent_EMA_digest=study.s.digest(last['ema']),parent_optimizer_moment_digest=study.g.optimizer_moment_digest(last['optimizer']),optimizer_updates=0,real_model_forwards=0,teacher_optical_forwards=0,test_images_read=0,inference_parameters_added=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
