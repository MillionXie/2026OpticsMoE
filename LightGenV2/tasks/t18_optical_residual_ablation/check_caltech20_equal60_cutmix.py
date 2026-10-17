"""Own weighted-objective, restoration and orchestration admission; no images."""
import argparse,ast,builtins,contextlib,io,json,os
from pathlib import Path
import numpy as np
import torch
import caltech20_equal60_cutmix as m
def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and m.base.slots()['total']==0;controls=m.admit();cfg=m.config();torch.set_num_threads(4)
    counts=np.asarray(m.base.dataset.manifest()['supports']['train'],np.float64);weights=np.sqrt(counts.sum()/(20*counts));weights*=counts.sum()/np.dot(counts,weights);assert abs(np.dot(counts,weights)/counts.sum()-1)<1e-12 and np.allclose(weights,m.class_weights(cfg,'cpu').numpy(),atol=1e-7,rtol=1e-7)
    logits=torch.linspace(-2,2,400,dtype=torch.float64).reshape(20,20).requires_grad_();target=torch.eye(20,dtype=torch.float64)*.95+.05/20;weighted=target*torch.tensor(weights)[None];value=-(weighted*torch.log_softmax(logits,1)).sum(1).mean();grad=torch.autograd.grad(value,logits)[0].numpy();z=logits.detach().numpy();p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);expected=(p*weighted.numpy().sum(1,keepdims=True)-weighted.numpy())/20;error=float(np.abs(expected-grad).max());assert error<1e-12

    plans=[]
    for n in (5,16,20):
        for mode in ('random','forced','clean'):
            force=mode=='forced';plan=m.cutmix_plan(n,71,0,cfg,force);assert np.array_equal(plan['perm'],m.cutmix_plan(n,71,0,cfg,force)['perm'])
            if mode=='clean':plan.update(apply=False,box=(0,0,0,0),lam=1.)
            x=torch.linspace(.01,.7,n*224*224,dtype=torch.float64).reshape(n,224,224);y=torch.eye(20,dtype=torch.float64)[np.arange(n)%20];actual,target=m.cutmix_apply(x,y,plan);a=x.numpy().copy();xx=x.numpy();x1,y1,x2,y2=plan['box']
            for top in (0,112):
                for left in (0,112):a[:,top+y1:top+y2,left+x1:left+x2]=xx[plan['perm'],top+y1:top+y2,left+x1:left+x2]
            a*=.25/np.sqrt(np.mean(a*a,axis=(-2,-1),keepdims=True));lam=1-(x2-x1)*(y2-y1)/(112**2);expected=lam*y.numpy()+(1-lam)*y.numpy()[plan['perm']];assert plan['lam']==lam and np.max(np.abs(a-actual.numpy()))<1e-12 and np.max(np.abs(expected-target.numpy()))<1e-12 and np.allclose(target.sum(1).numpy(),1)
            plans.append(dict(n=n,mode=mode,forced=force,apply=plan['apply'],box=list(plan['box']),lam=lam,pixel_and_label_numpy_maxabs=float(np.max(np.abs(a-actual.numpy())))))
    restored={}
    for key in m.CASES:
        parent=m.BASE/key;last=torch.load(parent/'last.pt',map_location='cpu',weights_only=False);assert last['epoch']==70 and last['scheduler']['last_epoch']==10 and last['steps']==13790 and all(float(q['step'])==13790 for q in last['optimizer']['state'].values());restored[key]=dict(raw_digest=m.s.digest(last['model']),ema_digest=m.s.digest(last['ema']),optimizer_moment_digest=m.moments(last['optimizer']),last_sha256=m.s.sha(parent/'last.pt'))
    source=Path(m.__file__).read_text();tree=ast.parse(source)
    for fn in (n for n in tree.body if isinstance(n,ast.FunctionDef)):
        reads={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)};local={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Param))}|{a.arg for a in fn.args.args};missing=reads-local-set(m.__dict__)-set(dir(builtins));assert not missing,(fn.name,missing)
    fake=dict(case='proof_only',epoch=61,train=.5,val=.4);buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):print(m.json.dumps(fake),flush=True)
    assert json.loads(buffer.getvalue())==fake
    m.claim(out.with_name(out.name+'.claim.json'),'own_sqrt_objective_CPU');out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,independent_numpy_CutMix_checks=plans,analytic_weighted_softmax_gradient_maxabs=error,restored=restored,frozen_controls=controls,all_entry_global_names_resolved=True,actual_log_serialization_passed=True,model_objects=0,model_forward_calls=0,optimizer_steps=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
