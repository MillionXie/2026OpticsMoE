"""TRAIN-only CPU branch-gradient accounting; never changes any model weight."""
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_equal60_logrepair as m
FILES=('caltech20_equal60_skip_diagnosis.py','CALTECH20_EQUAL60_SKIP_DIAGNOSIS_PROTOCOL_20261017.md')
OUT=m.HERE/'runs/smoke/caltech20_equal60_shared_skip_TRAIN_cpu_20261017'
def sources():
    src=m.sources()
    for name in FILES:src[(m.HERE/name).relative_to(m.ROOT).as_posix()]=m.s.sha(m.HERE/name)
    return src
def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.ROOT,text=True).strip();m.engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)
def hook_proof():
    class Toy(torch.nn.Module):
        def forward(self,x,size):return x@torch.tensor([[1.,3.],[2.,4.]],dtype=torch.float64)
    x=torch.tensor([[.2,.8]],dtype=torch.float64,requires_grad=True);block=Toy();saved={}
    def pre(module,args):saved['input']=args[0];args[0].retain_grad()
    def back(module,grad_input,grad_output):saved['branch']=grad_input[0].detach().clone();assert grad_input[1] is None
    a=block.register_forward_pre_hook(pre);b=block.register_full_backward_hook(back);(2*x+.3*block(x,2)).sum().backward();a.remove();b.remove()
    assert torch.allclose(saved['input'].grad,torch.tensor([[3.2,3.8]],dtype=torch.float64),atol=1e-12,rtol=0) and torch.allclose(saved['branch'],torch.tensor([[1.2,1.8]],dtype=torch.float64),atol=1e-12,rtol=0) and torch.allclose(saved['input'].grad-saved['branch'],torch.full_like(x,2),atol=1e-12,rtol=0)
    return dict(passed=True,synthetic_forward_backward_calls=1,analytic_total_and_branch_gradient_exact=True)
def diagnose(out):
    assert out==OUT and not out.exists() and not out.with_name(out.name+'.claim.json').exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=m.config();meta=m.s.read(m.OUT/'metadata.json');assert meta['sources']==m.sources();m.engine.verify_sources(meta['sources'],meta['git_commit']);parents={}
    for key in ('L6_rho03','L6_rho0'):
        folder=m.OUT/key;f=m.s.read(folder/'fit_result.json');ex=m.s.read(folder.with_name(key+'.fit_exit.json'));assert f['epochs_completed']==60 and f['steps']==11820 and ex['returncode']==0 and ex['actual_exit'];p=Path('/proc',str(ex['pid']),'cmdline');assert not p.exists() or str(folder).encode() not in p.read_bytes();assert f['best_sha256']==m.s.sha(folder/'best.pt') and f['history_sha256']==m.s.sha(folder/'history.json');parents[key]=f
    proof=hook_proof();m.s.save(out.with_name(out.name+'.claim.json'),dict(identity(),pid=os.getpid(),phase='TRAIN20_CPU_shared_branch_gradient_accounting',model_weight_updates=0,TEST_images_read=0),exclusive=True);torch.set_num_threads(4);data=m.load('train');indices=[int((data[1]==label).nonzero()[0]) for label in range(20)];weights=torch.tensor(len(data[1])/(20*np.bincount(data[1].numpy(),minlength=20)),dtype=torch.float32);rows=[];slots_before=m.slots()
    for key,rho in [('L6_rho03',.3),('L6_rho0',0.)]:
        checkpoint=torch.load(m.OUT/key/'best.pt',map_location='cpu',weights_only=False);m.engine.seed(17);model=m.RegularizedMoE(6,rho);model.load_state_dict(checkpoint['model'],strict=True);model.eval();before=m.s.digest(model.state_dict());captured={};handles=[]
        def pre(index):
            def hook(module,args):
                value=args[0];value.retain_grad();captured[index]=dict(input=value)
            return hook
        def backward(index):
            def hook(module,grad_input,grad_output):captured[index]['electronic_input_gradient']=grad_input[0].detach().clone()
            return hook
        for index,block in enumerate(model.electronic):handles.extend([block.register_forward_pre_hook(pre(index)),block.register_full_backward_hook(backward(index))])
        statistics={index:[] for index in range(6)}
        for idx in indices:
            model.zero_grad(set_to_none=True);captured.clear();x=m.b.encode(data[0][idx:idx+1],None,100).requires_grad_();p,c=model(x);target=m.engine.F_onehot(data[1][idx:idx+1],cfg['label_smoothing']);loss=m.loss(model,p,c,target,weights,cfg);loss.backward()
            for index,values in captured.items():
                total=values['input'].grad.detach();electronic=values['electronic_input_gradient'];other=total-electronic;en=float(electronic.double().norm());on=float(other.double().norm());tn=float(total.double().norm());den=en*on;cos=float((electronic.double()*other.double()).sum()/den) if den else None
                assert np.isfinite([en,on,tn]).all();statistics[index].append(dict(electronic_gradient_norm=en,other_gradient_norm=on,total_gradient_norm=tn,electronic_norm_share=en/max(en+on,1e-30),electronic_other_cosine=cos))
            captured.clear()
        for handle in handles:handle.remove()
        assert before==m.s.digest(model.state_dict()) and not torch.cuda.is_initialized()
        aggregate=[]
        for index,records in statistics.items():
            shares=[q['electronic_norm_share'] for q in records];aggregate.append(dict(stage=index,kind=model.stages[index],electronic_norm_share_mean=float(np.mean(shares)),electronic_norm_share_min=min(shares),electronic_norm_share_max=max(shares),mean_electronic_gradient_norm=float(np.mean([q['electronic_gradient_norm'] for q in records])),mean_other_gradient_norm=float(np.mean([q['other_gradient_norm'] for q in records])),first_stage_other_includes_router=index==0))
        rows.append(dict(case=key,depth=6,rho=rho,selected_epoch=checkpoint['epoch'],parent_PT_sha256=parents[key]['best_sha256'],model_digest_before_after=before,alphas=model.alphas(),statistics=aggregate,TRAIN_samples=20));del model,checkpoint
    out.mkdir();m.s.save(out/'diagnosis.json',dict(identity(),passed=True,scope='branch input-gradient accounting on20 fixed TRAIN examples; norms are not physical power fractions or causal ablation proof',hook_cpu_proof=proof,rows=rows,TRAIN_ids=[str(data[2][i]) for i in indices],forward_calls=40,backward_calls=40,model_objects=2,optimizer_steps=0,weights_unchanged=True,VAL_images_read=0,TEST_images_read=0,CUDA_initialized=False,slots_before=slots_before,slots_after=m.slots()),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:diagnose(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
