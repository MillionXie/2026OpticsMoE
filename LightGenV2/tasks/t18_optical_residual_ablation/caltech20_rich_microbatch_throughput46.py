"""TRAIN-only throughput/memory admission; never updates a parameter."""
import argparse,os,subprocess,time,json,statistics
from pathlib import Path
import numpy as np,torch
import caltech20_rich_view_target_consistency46 as method
HERE,ROOT,s,base,engine,resources,old=method.HERE,method.ROOT,method.s,method.base,method.engine,method.resources,method.old
FILES=('caltech20_rich_microbatch_throughput46.py','CALTECH20_RICH_MICROBATCH_THROUGHPUT46_PROTOCOL_20261018.md')
CPU=HERE/'runs/smoke/caltech20_rich_microbatch_throughput46_CPU_20261018';OUT=HERE/'runs/smoke/caltech20_rich_microbatch_throughput46_TRAIN_20261018'
def sources():
    z=method.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def cpu():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not CPU.exists();claim(CPU.with_name(CPU.name+'.claim.json'),'microbatch_linear_accumulation_CPU');r=np.random.default_rng(17);x=torch.tensor(r.normal(size=(16,7)));w=torch.tensor(r.normal(size=(7,3)),requires_grad=True);target=torch.tensor(r.normal(size=(16,3)));expected=x.numpy().T@(x.numpy()@w.detach().numpy()-target.numpy())/16;checks=[]
    for micro in (2,4,8):
        w.grad=None
        for off in range(0,16,micro):(.5*(x[off:off+micro]@w-target[off:off+micro]).square().sum()/16).backward()
        e=float(np.abs(w.grad.numpy()-expected).max());assert e<1e-12;checks.append(dict(micro=micro,error=e))
    assert s.read(method.CPU/'checks.json')['passed'];CPU.mkdir();s.save(CPU/'checks.json',dict(identity(),passed=True,independent_numpy_accumulation=checks,resource_guard=resources.proof(),real_model_constructions=0,real_forward_calls=0,optimizer_steps=0,images_read=0),exclusive=True)
def benchmark(gpu):
    assert gpu==resources.GPU[2] and os.environ['CUDA_VISIBLE_DEVICES']==gpu and not OUT.exists();p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();claim(OUT.with_name(OUT.name+'.claim.json'),'actual_TRAIN_micro2_4_8_throughput');slots=resources.slots();assert slots['total']<=5 and slots['counts'][gpu]==1 and any(j['pid']==os.getpid() and j['gpu_uuid']==gpu for j in slots['jobs']);s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4);OUT.mkdir();method.select('L6_prototype_both');cfg=method.config();model=method.initialize(6,False).cuda();before=s.digest(model.state_dict());data=base.load('train');order,theta=engine.epoch_plan(1,3141,cfg);idx=order[:16];x,y,mask,_=method.mixed(data,idx,theta,1,0,cfg,'cuda');q,mass=method.old.targets(idx,1,0,cfg,'cuda');weights=method.old.kd.class_weights(cfg,'cuda');reference=None;rows=[]
    # Warm FFT plans on the same TRAIN macro without any optimizer/update.
    model.context(method.seed_for(1,0),16,cfg);method.gradients(model,x,y,q,mass,mask,weights,cfg,2);model.zero_grad(set_to_none=True);torch.cuda.synchronize()
    for micro in (2,4,8):
        timings=[];max_mem=0
        try:
            for repeat in range(2):
                model.zero_grad(set_to_none=True);model.context(method.seed_for(1,0),16,cfg);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();prob,capture,loss=method.gradients(model,x,y,q,mass,mask,weights,cfg,micro);torch.cuda.synchronize();timings.append(time.perf_counter()-start);max_mem=max(max_mem,torch.cuda.max_memory_allocated());vectors={n:v.grad.detach().cpu().double().numpy().copy() for n,v in model.named_parameters()};pv=prob.detach().cpu().numpy();cv=capture.detach().cpu().numpy()
                if reference is None:reference=(vectors,pv,cv,loss)
                rel=max(float(np.linalg.norm(vectors[n]-a)/max(np.linalg.norm(a),1e-30)) for n,a in reference[0].items());pe=float(np.abs(pv-reference[1]).max());ce=float(np.abs(cv-reference[2]).max());assert rel<=2e-4 and pe<=3e-6 and ce<=3e-6 and abs(loss-reference[3])<=2e-6*max(1,abs(loss));assert s.digest(model.state_dict())==before
            rows.append(dict(micro=micro,passed=True,seconds=timings,median_seconds=statistics.median(timings),peak_allocated_bytes=max_mem,gradient_max_relative=rel,probability_maxabs=pe,capture_maxabs=ce))
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache();assert s.digest(model.state_dict())==before;rows.append(dict(micro=micro,passed=False,rejected_memory_policy=True,automatic_retry=False));break
    assert rows[0]['passed'];winner=min((row for row in rows if row['passed']),key=lambda row:row['median_seconds']);model.zero_grad(set_to_none=True);after=s.digest(model.state_dict());assert after==before;del model;torch.cuda.empty_cache();result=dict(identity(),passed=True,policy=policy,depth=6,case='prototype_both',rows=rows,fastest_micro=winner['micro'],speedup_over_micro2=rows[0]['median_seconds']/winner['median_seconds'],TRAIN_macro_indices_SHA256=s.digest(idx),unique_TRAIN_images=16,VALID_images_read=0,TEST_images_read=0,optimizer_steps=0,real_models=1,state_before=before,state_after=after,training_source_not_modified=True,interpretation='Local macro timing on3090; not guaranteed whole-epoch4090 speedup, not a new accuracy result');s.save(OUT/'benchmark.json',result,exclusive=True);print(json.dumps(dict(passed=True,benchmark_SHA256=s.sha(OUT/'benchmark.json'),rows=rows,fastest_micro=winner['micro'],speedup=result['speedup_over_micro2'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','benchmark'],required=True);p.add_argument('--gpu');a=p.parse_args()
    try:cpu() if a.phase=='cpu' else benchmark(a.gpu)
    except BaseException as e:
        out=CPU if a.phase=='cpu' else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
