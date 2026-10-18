"""Two fixed TRAIN macros: actual CE/KD/prototype gradients, no optimizer."""
import argparse,types,os,sys,json,subprocess,hashlib,math
from pathlib import Path
import numpy as np,torch
import caltech20_rich_currentKD46 as method
import caltech20_rich_overallCE46 as closure
HERE,ROOT,s,base,engine,resources=method.HERE,method.ROOT,method.s,method.base,method.engine,method.resources
FILES=('caltech20_rich_loss_gradient_diagnosis46.py','CALTECH20_RICH_LOSS_GRADIENT_DIAGNOSIS46_PROTOCOL_20261018.md')
CPU=HERE/'runs/smoke/caltech20_rich_loss_gradient46_CPU_20261018';OUT=HERE/'runs/smoke/caltech20_rich_loss_gradient46_TRAIN_20261018'
def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def summarize(a,b):
    a,b=a.astype(float),b.astype(float);aa=float(np.linalg.norm(a));bb=float(np.linalg.norm(b));return dict(norm=aa,relative_to_CE=aa/max(bb,1e-30),cosine_to_CE=float(np.dot(a,b)/max(aa*bb,1e-30)))
def cpu():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not CPU.exists();claim(CPU.with_name(CPU.name+'.claim.json'),'actual_loss_gradient_diagnosis_analytic_CPU');rng=np.random.default_rng(17);a=torch.tensor(rng.normal(size=(7,5)));x=torch.tensor(rng.normal(size=5),requires_grad=True);y=torch.tensor(rng.normal(size=7));pred=a@x;l1=.5*(pred-y).square().sum();l2=.5*pred.square().sum();g1=torch.autograd.grad(l1,x,retain_graph=True)[0];g2=torch.autograd.grad(l2,x,retain_graph=True)[0];g=torch.autograd.grad(l1+.3*l2,x)[0];expected=a.numpy().T@(a.numpy()@x.detach().numpy()-y.numpy())+.3*a.numpy().T@(a.numpy()@x.detach().numpy());assert np.max(abs(g.numpy()-expected))<1e-12 and float((g-g1-.3*g2).abs().max())<1e-12;CPU.mkdir();s.save(CPU/'checks.json',dict(identity(),passed=True,independent_linear_gradient_decomposition=True,real_model_constructions=0,real_model_forward_calls=0,optimizer_steps=0,images_read=0),exclusive=True)
def gradients_counterfactual(model,x,y,q,mass,mask,w,cfg,proto_enabled=True):
    ns=dict(method.gradient_kernel.__globals__);original=method.target_data
    if not proto_enabled:ns['target_data']=lambda depth:(original(depth)[0],0.,original(depth)[2])
    f=method.gradient_kernel;kernel=types.FunctionType(f.__code__,ns,'same_exact_loss_kernel_counterfactual_coefficient',f.__defaults__,f.__closure__);return kernel(model,x,y,q,mass,mask,w,cfg,cfg['microbatch_size'])
def diagnose(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu and gpu in resources.GPU.values();p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();assert not OUT.exists();claim(OUT.with_name(OUT.name+'.claim.json'),'two_fixed_TRAIN_macro_loss_gradient_diagnosis');slots=resources.slots();assert slots['total']<=5 and slots['counts'][gpu]==1 and any(j['pid']==os.getpid() and j['gpu_uuid']==gpu for j in slots['jobs']);s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4);OUT.mkdir();data=base.load('train');rows=[];model_checks={}
    for depth in (4,6):
        key=f'L{depth}_currentKD';method.select(key);cfg=method.config();model=method.initialize(depth,False).cuda();initial=s.digest(model.state_dict());order,theta=engine.epoch_plan(1,3141,cfg);w=method.old.kd.class_weights(cfg,'cuda');names=[n for n,_ in model.named_parameters()];groups={k:[n for n in names if (n=='raw_router' or n.startswith('phase_sets.'))==flag and (k!='alpha' or n.startswith('fusion_logits.')) and (k!='electronics' or not n.startswith('fusion_logits.'))] for k,flag in [('phase',True),('electronics',False),('alpha',False)]}
        for batch in (0,1):
            idx=order[batch*16:batch*16+16];x,y,mask,crop=method.mixed(data,idx,theta,1,batch,cfg,'cuda');q,mass=method.old.targets(idx,1,batch,cfg,'cuda');seed=method.seed_for(1,batch);vectors={};reference=None;losses={}
            for kind in ('total','no_proto','no_KD','CE_only'):
                c=dict(cfg)
                if kind in ('no_KD','CE_only'):c['KD_weight']=0.
                if kind=='CE_only':
                    for k in ('phase_anchor_weight','phase_smooth_weight','capture_weight','consistency_weight'):c[k]=0.
                model.context(seed,len(idx),cfg);prob,capture,loss=gradients_counterfactual(model,x,y,q,mass,mask,w,c,kind not in ('no_proto','CE_only'));pair=(prob.detach().cpu().numpy(),capture.detach().cpu().numpy())
                if reference is None:reference=pair
                assert max(float(np.max(abs(a-b))) for a,b in zip(pair,reference))<=3e-6;assert s.digest(model.state_dict())==initial;vectors[kind]={n:v.grad.detach().cpu().numpy().copy() for n,v in model.named_parameters()};assert all(np.isfinite(v).all() for v in vectors[kind].values());losses[kind]=float(loss)
            arrays={};report={}
            for group,ns in groups.items():
                total=np.concatenate([vectors['total'][n].ravel() for n in ns]);ce=np.concatenate([vectors['CE_only'][n].ravel() for n in ns]);proto=total-np.concatenate([vectors['no_proto'][n].ravel() for n in ns]);kd=total-np.concatenate([vectors['no_KD'][n].ravel() for n in ns]);other=total-ce-proto-kd
                for name,value in [('total',total),('CE',ce),('prototype',proto),('KD',kd),('other_regularizers',other)]:arrays[group+'_'+name]=value;report.setdefault(group,{})[name]=summarize(value,ce)
            path=OUT/f'L{depth}_macro{batch}_gradients.npz';base.save_np(path,arrays);rows.append(dict(depth=depth,batch=batch,n=16,TRAIN_indices_SHA256=s.digest(idx),source_loss_coefficients=dict(KD=cfg['KD_weight'],prototype=method.target_data(depth)[1]),losses=losses,groups=report,gradient_arrays_SHA256=s.sha(path),state_unchanged=True));model.zero_grad(set_to_none=True)
        model_checks[str(depth)]=dict(parent_SHA256=method.incumbent(depth)[0]['sha256'],before=initial,after=s.digest(model.state_dict()),optimizer_steps=0);assert model_checks[str(depth)]['before']==model_checks[str(depth)]['after'];del model;torch.cuda.empty_cache()
    result=dict(identity(),passed=True,policy=policy,rows=rows,model_checks=model_checks,unique_TRAIN_images_forwarded=32,TRAIN_images_loaded=3141,forward_calls=256,forward_image_units=512,optimizer_steps=0,VALID_images_read=0,TEST_images_read=0,real_models=2,not_a_new_performance_result=True,interpretation='Two local TRAIN macros, not whole-dataset proof or unique-cause attribution');s.save(OUT/'diagnosis.json',result,exclusive=True);print(json.dumps(dict(passed=True,diagnosis_SHA256=s.sha(OUT/'diagnosis.json'),rows=[dict(depth=q['depth'],batch=q['batch'],groups=q['groups']) for q in rows],optimizer_steps=0,TEST_images_read=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','diagnose'],required=True);p.add_argument('--gpu');a=p.parse_args()
    try:cpu() if a.phase=='cpu' else diagnose(a.gpu)
    except BaseException as e:
        f=CPU if a.phase=='cpu' else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
