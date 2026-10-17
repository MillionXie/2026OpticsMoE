"""Real-parent shape and independent pixel/KD math, zero optical forwards."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import caltech20_alt246_richsplit_selfview2 as m
from audit_caltech20_alt246_richsplit import verify_predictions
from audit_caltech20_alt246_richsplit_viewreg import reference_parameters,reference_pixels
def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';p=m.parent();cfg=m.config();m.claim(out.with_name(out.name+'.claim.json'),'selfview2_real_parent_CPU_and_analytic_math');torch.set_num_threads(4);state=torch.load(p['best'],map_location='cpu',weights_only=False)['model'];model=m.RegularizedMoE(2,.3);model.load_state_dict(state,strict=True);contract=model.contract();assert contract['optical_phase_parameters']==479364 and contract['electronic_parameters']==5160 and len(dict(model.named_parameters()))==12;digest=m.s.digest(model.state_dict());assert digest==m.s.digest(state);_,saved=verify_predictions(Path(p['train']),state,'train',m.b.dataset.manifest());data=m.b.load('train');idx=np.arange(3);images=data[0][idx].permute(0,2,3,1).numpy();params=m.v.view_parameters(3,101,0,cfg);expected=reference_parameters(3,101,0,cfg);assert np.array_equal(params,expected);actual=m.v.transform(images,params);independent=reference_pixels(images,expected);assert np.array_equal(actual,independent)
    torch.manual_seed(17);logits=torch.randn(20,20,dtype=torch.float64,requires_grad=True);target=torch.softmax(torch.randn(20,20,dtype=torch.float64),1);labels=torch.eye(20,dtype=torch.float64)*.95+.05/20;gate=(torch.arange(20)%3!=0).double();prob=torch.softmax(logits,1);logT=F.log_softmax(logits/2,1);loss=-(labels*F.log_softmax(logits,1)).sum(1).mean()+.5*4*(target*(target.log()-logT)).sum(1).mul(gate).mean();loss.backward();z=logits.detach().numpy();p0=np.exp(z-z.max(1,keepdims=True));p0/=p0.sum(1,keepdims=True);zt=z/2;pT=np.exp(zt-zt.max(1,keepdims=True));pT/=pT.sum(1,keepdims=True);grad=(p0-labels.numpy())/20+.5*2*(pT-target.numpy())*gate.numpy()[:,None]/20;err=float(np.abs(grad-logits.grad.numpy()).max());assert err<=1e-10
    out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,parent=p,initial_digest=digest,parent_prediction_check=saved,contract=contract,independent_view_pixels_equal=True,CE_KD_analytic_gradient_maxabs=err,real_model_objects=1,optical_forwards=0,optimizer_updates=0,teacher_forwards=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        path=a.out.with_name(a.out.name+'.failure.json')
        if not path.exists():m.s.save(path,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
