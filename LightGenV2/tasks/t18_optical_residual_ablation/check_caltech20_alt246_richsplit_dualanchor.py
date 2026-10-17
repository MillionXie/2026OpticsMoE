"""Two real selected parent shapes, independent dual weightedCE/KD gradient math."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import caltech20_alt246_richsplit_dualanchor as m
from audit_caltech20_alt246_richsplit import verify_predictions
from audit_caltech20_alt246_richsplit_viewreg import reference_parameters,reference_pixels
def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';m.claim(out.with_name(out.name+'.claim.json'),'dual_clean075_mild025_two_parent_CPU_math');torch.set_num_threads(4);parents={};checks={};manifest=m.b.dataset.manifest()
    for depth in (2,4):
        m.configure(depth);p=m.parent();cfg=m.config();parents[str(depth)]=p;state=torch.load(p['best'],map_location='cpu',weights_only=False)['model'];model=m.RegularizedMoE(depth,.3);model.load_state_dict(state,strict=True);assert model.contract()['terminal_linear_parameters']==5140 and len(dict(model.named_parameters()))==(12 if depth==2 else 21) and m.s.digest(model.state_dict())==m.s.digest(state);_,saved=verify_predictions(Path(p['train']),state,'train',manifest);data=m.b.load('train');images=data[0][:3].permute(0,2,3,1).numpy();params=m.v.view_parameters(3,201,0,cfg);ref=reference_parameters(3,201,0,cfg);assert np.array_equal(params,ref) and np.array_equal(m.v.transform(images,params),reference_pixels(images,ref));torch.manual_seed(17);z0=torch.randn(20,20,dtype=torch.float64,requires_grad=True);z1=torch.randn(20,20,dtype=torch.float64,requires_grad=True);y=torch.eye(20,dtype=torch.float64)*.95+.05/20;weights=torch.ones(20,dtype=torch.float64) if depth==2 else torch.tensor(np.sqrt(3141/(20*np.array(manifest['supports']['train']))));Y=y*weights[None];target=torch.softmax(torch.randn(20,20,dtype=torch.float64),1);gate=(torch.arange(20)%3!=0).double();value=.75*-(Y*F.log_softmax(z0,1)).sum(1).mean()+.25*(-(Y*F.log_softmax(z1,1)).sum(1).mean()+.5*4*(target*(target.log()-F.log_softmax(z1/2,1))).sum(1).mul(gate).mean());value.backward();a=torch.softmax(z0.detach(),1).numpy();b=torch.softmax(z1.detach(),1).numpy();T=torch.softmax(z1.detach()/2,1).numpy();expected0=.75*(a*Y.numpy().sum(1,keepdims=True)-Y.numpy())/20;expected1=.25*((b*Y.numpy().sum(1,keepdims=True)-Y.numpy())/20+.5*2*(T-target.numpy())*gate.numpy()[:,None]/20);err=max(float(np.abs(expected0-z0.grad.numpy()).max()),float(np.abs(expected1-z1.grad.numpy()).max()));assert err<=1e-10;checks[str(depth)]=dict(contract=model.contract(),parent_prediction_check=saved,dual_CE_KD_analytic_gradient_maxabs=err,independent_mild_view_pixels_equal=True);del model
    out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,parents=parents,checks=checks,real_model_objects=2,optical_forwards=0,optimizer_updates=0,teacher_forwards=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        path=a.out.with_name(a.out.name+'.failure.json')
        if not path.exists():m.s.save(path,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
