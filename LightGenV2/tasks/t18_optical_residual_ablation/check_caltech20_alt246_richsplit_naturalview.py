"""Restore-state and independent TRAIN view math admission; no model forwards."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_naturalview as study
from audit_caltech20_alt246_richsplit_naturalview import reference_parameters,reference_pixels

def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and out==study.CPU and not out.exists();study.base_ready();study.b.slots();study.claim(out.with_name(out.name+'.claim.json'),'viewreg_CPU_admission');torch.set_num_threads(4);cfg=study.config();assert study.s.sha(cfg['data'])==cfg['cache_sha256'];proofs=[]
    images=np.stack([np.zeros((150,150,3),np.uint8),np.full((150,150,3),255,np.uint8),np.arange(150*150*3,dtype=np.int64).reshape(150,150,3).astype(np.uint8)])
    for epoch,batch in ((101,0),(101,196),(120,55)):
        a=study.view_parameters(3,epoch,batch,cfg);b=reference_parameters(3,epoch,batch,cfg);assert np.array_equal(a,b);actual=study.transform(images,a);expected=reference_pixels(images,b);assert actual.dtype==np.uint8 and actual.shape==images.shape and np.array_equal(actual,expected) and not actual[0].any();proofs.append(dict(epoch=epoch,batch=batch,parameters_maxabs=float(np.abs(a-b).max()),pixel_maxabs=int(np.abs(actual.astype(int)-expected.astype(int)).max())))
    states=[]
    for depth in (6,):
        parent=study.parent_info(study.BASE,depth);last=torch.load(parent['last'],map_location='cpu',weights_only=False);model=study.RegularizedMoE(depth);model.load_state_dict(last['model'],strict=True);assert last['epoch']==100 and last['steps']==19700 and model.contract()['terminal_linear_parameters']==5140 and model.contract()['main_depth']==depth and all(.7<v<=.95 for v in model.alphas())
        phase=[];electronic=[]
        for name,value in model.named_parameters():(phase if name=='raw_router' or name.startswith('phase_sets.') else electronic).append(value)
        opt=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);opt.load_state_dict(last['optimizer']);assert study.g.optimizer_moment_digest(opt.state_dict())==study.g.optimizer_moment_digest(last['optimizer']) and all(float(v['step'])==19700 for v in opt.state.values());states.append(dict(depth=depth,parent_last_sha256=study.s.sha(parent['last']),raw_digest=study.s.digest(last['model']),ema_digest=study.s.digest(last['ema']),optimizer_steps=19700,contract=model.contract()))
    class Dummy:
        def phase_smoothness(self):return torch.tensor(0.,dtype=torch.float64)
    logits=torch.linspace(-1,1,400,dtype=torch.float64).reshape(20,20).requires_grad_();targets=torch.eye(20,dtype=torch.float64)*.95+.05/20;prob=logits.softmax(1);capture=torch.ones(20,dtype=torch.float64);value=study.b.loss(Dummy(),prob,capture,targets,torch.ones(20,dtype=torch.float64),dict(capture_weight=0.,phase_smooth_weight=0.));value.backward();expected=(prob.detach().numpy()-targets.numpy())/20;assert np.max(np.abs(logits.grad.numpy()-expected))<1e-12
    out.mkdir();study.s.save(out/'checks.json',dict(study.identity(),passed=True,versions=study.versions(),view_math=proofs,natural_CE_gradient_maxabs=float(np.max(np.abs(logits.grad.numpy()-expected))),restored_states=states,optimizer_updates=0,real_model_forwards=0,test_images_read=0,inference_parameters_added=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
