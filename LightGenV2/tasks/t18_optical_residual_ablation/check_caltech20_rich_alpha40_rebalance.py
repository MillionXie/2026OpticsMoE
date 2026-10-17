"""Independent alpha/freeze/augmentation/optimizer contract admission."""
import os
import numpy as np
import torch
import caltech20_rich_alpha40_rebalance as m
def run():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'alpha40_rebalance_CPU')
    torch.set_num_threads(4);d=m.s.read(m.diagnosis.OUT/'diagnosis.json');assert d['passed'] and d['sources']==m.diagnosis.sources() and d['TEST_images_read']==0
    cfg=m.config();data=m.base.load('train');m.base.load('val');m.old.kd.teacher_ready();order,theta=m.engine.epoch_plan(1,3141,cfg);idx=order[:5]
    x,y,mask,crop=m.mixed(data,idx,theta,1,0,cfg,'cpu');xx,yy,mm,cc=m.mixed(data,idx,theta,1,0,cfg,'cpu');assert torch.equal(x,xx) and torch.equal(y,yy) and torch.equal(mask,mm) and crop==cc
    assert x.shape==(5,224,224) and y.shape==(5,20) and torch.allclose(y.sum(1),torch.ones(5),atol=1e-6) and torch.isfinite(x).all() and 0<=x.min()<=x.max()<=1
    checks={}
    for key,case in cfg['cases'].items():
        model=m.initialize(case['depth'],case['fresh']);g=m.groups(model);assert len(list(model.parameters()))=={2:12,4:21,6:30}[case['depth']]
        assert sum(p.numel() for n,p in g[0])=={2:479364,4:908552,6:1337740}[case['depth']] and sum(p.numel() for p in model.linear.parameters())==5140
        assert all(abs(a-.45)<1e-7 for a in model.alphas())
        if case['fresh']:
            m.engine.seed(17);reference=m.old.PairModel(case['depth'],.3)
            assert all(torch.equal(p,reference.state_dict()[n]) for n,p in model.state_dict().items() if not n.startswith('fusion_logits.'))
        for epoch in (1,case['epochs']):assert all(np.isfinite(a) and a>0 for a in m.rates(case,epoch))
        checks[key]=dict(parameter_tensors=len(list(model.parameters())),phase_parameters=sum(p.numel() for n,p in g[0]),alpha_restart=model.alphas(),fresh=case['fresh'],new_epochs=case['epochs'],electronic_only_epochs=case['electronic_warm_epochs'])
    # Synthetic three-group AdamW: frozen phase/alpha do not get Adam state.
    p=[torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64)) for _ in range(3)];opt=torch.optim.AdamW([dict(params=[a],lr=lr,weight_decay=wd) for a,lr,wd in zip(p,[.002,.001,.003],[0,.05,0])])
    old=[a.detach().clone() for a in p];p[1].grad=2*p[1].detach();opt.step();assert torch.equal(p[0],old[0]) and torch.equal(p[2],old[2]) and not opt.state[p[0]] and not opt.state[p[2]]
    expected=old[1]*(1-.001*.05)-.001*(2*old[1])/((2*old[1]).abs()+1e-8);assert torch.allclose(p[1],expected,atol=1e-14,rtol=0)
    m.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,resources=m.resources.proof(),actual_augmentation_path_deterministic=True,real_model_forward_calls=0,real_optimizer_steps=0,TEST_images_read=0,synthetic_Adam_steps=1),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
