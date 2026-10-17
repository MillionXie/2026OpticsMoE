"""Actual loader/augmentation/alpha paths plus independent math admission."""
import os
import numpy as np
import torch
import caltech20_rich_robust as m

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.CPU.exists()
    m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'new_robustness_CPU_actual_paths')
    torch.set_num_threads(4);cfg=m.config();controls=m.controls();data=m.base.load('train');m.base.load('val');m.kd.teacher_ready()
    order,theta=m.engine.epoch_plan(71,3141,cfg);idx=order[:5]
    x,y,mask=m.mixed(data,idx,theta,71,0,cfg,'cpu');a,b,c=m.mixed(data,idx,theta,71,0,cfg,'cpu')
    assert torch.equal(x,a) and torch.equal(y,b) and torch.equal(mask,c) and x.shape==(5,224,224) and y.shape==(5,20) and torch.allclose(y.sum(1),torch.ones(5))
    q,mass=m.targets(idx,71,0,cfg,'cpu');assert q.shape==(5,20) and torch.allclose(q.sum(1),torch.ones(5),atol=1e-6) and (mass>0).all()
    checks={}
    for depth in (2,4,6):
        path,pt=m.parent(depth);state=m.remap(pt['model']);model=m.RobustMoE(depth);model.load_state_dict(state,strict=True);model.anchor()
        assert sum(p.numel() for n,p in model.named_parameters() if n=='raw_router' or n.startswith('phase_sets.'))=={2:479364,4:908552,6:1337740}[depth]
        assert len(list(model.parameters()))=={2:12,4:21,6:30}[depth] and model.linear.weight.numel()+model.linear.bias.numel()==5140
        old_alpha=[.7001+(.95-.7001)*v.sigmoid().item() for n,v in pt['model'].items() if n.startswith('fusion_logits.')]
        assert np.allclose(old_alpha,model.alphas(),atol=8e-8,rtol=0)
        model.context(m.seed_for(71,0),5,cfg);first=m.s.digest(model.robust['masks']);noise=m.s.digest(model.robust['noise']);model.context(m.seed_for(71,0),5,cfg)
        assert first==m.s.digest(model.robust['masks']) and noise==m.s.digest(model.robust['noise']) and all(.6001<v<=.95 for v in model.alphas())
        folder,result,pred=m.incumbent_test(depth)
        checks[str(depth)]=dict(parent_sha256=m.s.sha(path),effective_alpha_preserved=True,mask_reproducible=True,noise_reproducible=True,parent_test_reused_no_forward=True,parent_test_result_sha256=m.s.sha(folder/'result.json'),parameters=len(list(model.parameters())))
        del model
    # Independent float64 SAM/Adam algebra: one step, exact restore, no real model forward.
    p=torch.nn.Parameter(torch.tensor([.2,-.4],dtype=torch.float64));opt=torch.optim.AdamW([p],lr=.003,weight_decay=.02)
    initial=p.detach().clone();first=2*p.detach();perturbed=initial+.5*first/first.norm();p.data.copy_(perturbed);p.grad=2*p.detach().clone();g=p.grad.clone();p.data.copy_(initial)
    assert torch.equal(p,initial);opt.step();expected=initial*(1-.003*.02)-.003*g/(g.abs()+1e-8)
    assert torch.allclose(p,expected,atol=1e-14,rtol=0) and float(opt.state[p]['step'])==1
    m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,resources=m.resources.proof(),frozen_controls=controls,real_model_forward_calls=0,TEST_images_read=0,optimizer_steps_real_models=0,synthetic_Adam_steps=1,production_augmentation_path_checked=True),exclusive=True)

if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
