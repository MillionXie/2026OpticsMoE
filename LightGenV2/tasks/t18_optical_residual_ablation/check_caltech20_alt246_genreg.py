"""New continuation admission: same inference graph, masks and restored state."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_genreg as study
from caltech20_alt246_genreg_model import RegularizedMoE,MaskedLinear

def check(out):
    s=study.s;assert out==study.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    original=study.baseline();assert original['sources']==study.b.sources() and study.b.slots()['total']==0
    study.claim(out.with_name(out.name+'.claim.json'),'new_genreg_CPU');torch.set_num_threads(4);data=study.b.load('train');x=study.b.encode(data[0][:1],None,100);checks=[]
    for depth in (2,4,6):
        key=f'L{depth}_rho03';cfg=study.config(depth);parent=torch.load(study.BASE/key/'last.pt',map_location='cpu',weights_only=False);assert parent['epoch']==100 and parent['steps']==8600 and parent['scheduler']['last_epoch']==100
        old=study.b.HybridMoE(depth,.3);new=RegularizedMoE(depth);old.load_state_dict(parent['ema'],strict=True);new.load_state_dict(parent['ema'],strict=True)
        assert list(dict(old.named_parameters()))==list(dict(new.named_parameters())) and old.contract()==new.contract() and s.digest(old.state_dict())==s.digest(new.state_dict())
        with torch.no_grad():
            old.eval();new.eval();p,c=old(x);q,d=new(x);assert torch.equal(p,q) and torch.equal(c,d) and new.trace==['router']+['expert','global']*(depth//2)
        phase=[];electronic=[]
        for n,v in new.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(v)
        optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=.0001)],lr=.002);optimizer.load_state_dict(parent['optimizer'])
        assert study.optimizer_moment_digest(optimizer.state_dict())==study.optimizer_moment_digest(parent['optimizer']) and all(float(v['step'])==8600 for v in optimizer.state.values())
        before=study.optimizer_moment_digest(optimizer.state_dict())
        for group in optimizer.param_groups:group['lr']=cfg['lr']
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,threshold=.002,threshold_mode='abs',min_lr=2e-5);scheduler.best=1.
        for _ in range(4):scheduler.step(1.1)
        assert scheduler.last_epoch==4 and scheduler.num_bad_epochs==0 and all(abs(group['lr']-cfg['lr']*.5)<1e-12 for group in optimizer.param_groups)
        state=scheduler.state_dict();restored=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer);restored.load_state_dict(state);restored.step(.99)
        assert restored.best==.99 and restored.num_bad_epochs==0 and study.optimizer_moment_digest(optimizer.state_dict())==before
        flip,mask=study.augmentation_masks(16,101,0,cfg);f2,m2=study.augmentation_masks(16,101,0,cfg);assert np.array_equal(flip,f2) and np.array_equal(mask,m2) and mask.shape==(16,256)
        assert set(np.unique(mask))==set(np.asarray([0,1/(1-cfg['feature_dropout'])],np.float32))
        # Analytic feature/weight/bias gradients for masked linear softmax CE.
        torch.manual_seed(17);linear=MaskedLinear().double().train();f=torch.randn(16,256,dtype=torch.float64,requires_grad=True);linear.mask=torch.from_numpy(mask.astype(np.float64));y=torch.arange(16)%20
        probability=torch.softmax(linear(f),1);loss=-probability[torch.arange(16),y].log().mean();loss.backward()
        F=f.detach().numpy();M=mask.astype(np.float64);W=linear.weight.detach().numpy();B=linear.bias.detach().numpy();logits=(F*M)@W.T+B;p0=np.exp(logits-logits.max(1,keepdims=True));p0/=p0.sum(1,keepdims=True);delta=p0.copy();delta[np.arange(16),y.numpy()]-=1;delta/=16
        assert np.allclose(delta.T@(F*M),linear.weight.grad.numpy(),atol=1e-10) and np.allclose(delta.sum(0),linear.bias.grad.numpy(),atol=1e-10) and np.allclose((delta@W)*M,f.grad.numpy(),atol=1e-10)
        linear.eval();assert torch.equal(linear(f.detach()),torch.nn.functional.linear(f.detach(),linear.weight,linear.bias))
        checks.append(dict(depth=depth,inference_parameter_count=sum(v.numel() for v in new.parameters()),inference_probability_maxabs=float((p-q).abs().max()),contract=new.contract(),parent_raw_digest=s.digest(parent['model']),parent_ema_digest=s.digest(parent['ema']),optimizer_restored_steps=8600,scheduler_replay_restore_without_optimizer_step=True,flip_count=int(flip.sum()),mask_nonzero_fraction=float((mask>0).mean()),masked_linear_analytic_gradients=True))
        del old,new,optimizer,parent
    # Pixel reference: horizontal flip does not swap RGB channels.
    images=torch.arange(2*3*4*5,dtype=torch.int32).reshape(2,3,4,5);flags=torch.tensor([True,False]);flipped=torch.where(flags[:,None,None,None],images.flip(-1),images)
    reference=images.numpy().copy();reference[0]=reference[0,:,:,::-1];assert np.array_equal(flipped.numpy(),reference)
    out.mkdir();s.save(out/'checks.json',dict(study.identity(),passed=True,checks=checks,real_model_forwards=6,optimizer_steps=0,test_arrays_read=0,new_graph_parameters=0,RGB_flip_reference=True,method='TRAIN-only horizontal flip and feature dropout; evaluation exactly unchanged'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        path=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not path.exists():study.s.save(path,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
