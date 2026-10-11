"""Bounded train-only numerical gate and throughput; never starts a training suite."""
import argparse, copy, gc, json, os, time, sys, subprocess
from pathlib import Path
import fast_optical as execution

o, torch = execution.optical, execution.torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    a.out.mkdir(parents=True, exist_ok=False)
    profile=o.r.read(o.HERE/'butterfly_profile.json')
    o.f.gpu_budget.ALLOWED.update(profile['gpu_uuids'])
    policy=o.f.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=3)
    torch.set_num_threads(4)
    manifest=o.r.read(a.data.parent/'data_manifest.json')
    sources=o.source_identity(o.HERE/'butterfly_profile.json')
    sources.update({str((o.HERE/name).relative_to(o.ROOT)).replace('\\','/'):o.r.sha(o.HERE/name)
                    for name in ['fast_optical.py','probe_fast_optical.py']})
    o.r.save(a.out/'metadata.json',dict(command=sys.argv,pid=os.getpid(),sources=sources,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data_sha256=o.r.sha(a.data),training_reads_test=False,gpu_policy=policy,time=o.r.now()))
    x,y,_=o.load_data(a.data,'train',manifest)
    x,y=x[:8],y[:8]
    amplitude=o.encode(x)
    weights=(len(manifest['split_ids']['train'])/(60*torch.tensor(manifest['supports']['train'],dtype=torch.float32))).cuda()
    results=[]
    for depth in [2,16]:
        for rho in [0.,.3]:
            cfg=o.config(profile,manifest,rho,'base')
            o.r.setseed(17)
            model=o.build(depth,profile,cfg)
            def compute(batch):
                model.zero_grad(set_to_none=True)
                probs=[]
                for start in range(0,8,batch):
                    end=min(start+batch,8)
                    probability,capture=o.forward(model,amplitude[start:end])
                    loss=o.objective(model,probability,capture,y[start:end].cuda(),weights,cfg)
                    (loss*(end-start)/8).backward()
                    probs.append(probability.detach())
                return torch.cat(probs),{n:p.grad.detach().clone() for n,p in model.named_parameters()}
            compute(1)
            torch.cuda.synchronize();baseline_started=time.perf_counter()
            reference,grad=compute(1)
            torch.cuda.synchronize();baseline_seconds=(time.perf_counter()-baseline_started)/8
            for fast,batch in [(False,2),(False,4),(True,1),(True,2),(True,4)]:
                if fast:execution.enable_expert_batching(model)
                torch.cuda.reset_peak_memory_stats()
                try:
                    torch.cuda.synchronize();started=time.perf_counter()
                    prob,g=compute(batch)
                    torch.cuda.synchronize();duration=time.perf_counter()-started
                    error=float((reference-prob).abs().max())
                    relative=max(float((g[n]-v).norm()/v.norm().clamp_min(1e-12)) for n,v in grad.items())
                    cosine=min(float(torch.nn.functional.cosine_similarity(g[n].flatten(),v.flatten(),dim=0)) for n,v in grad.items())
                    finite=all(torch.isfinite(v).all() and float(v.norm())>0 for v in g.values())
                    result=dict(depth=depth,rho=rho,expert_batched=fast,microbatch=batch,seconds_per_sample=duration/8,
                        serial_micro1_seconds_per_sample=baseline_seconds,speedup=baseline_seconds/(duration/8),
                        probability_max_error=error,gradient_max_relative_l2=relative,gradient_min_cosine=cosine,
                        gradients_finite_nonzero=bool(finite),passed=error<=1e-6 and relative<=.02 and cosine>=.999 and finite,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
                    del g,prob
                except torch.cuda.OutOfMemoryError:
                    result=dict(depth=depth,rho=rho,expert_batched=fast,microbatch=batch,passed=False,error='OOM under 3GiB cap')
                    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                results.append(result);print(json.dumps(result),flush=True)
                o.r.save(a.out/'probe.json',dict(scope='Fresh seed17 weights, first8 saved Butterfly train images only, no training/test; serial micro1 reference',
                    gpu_policy=policy,results=results,time=o.r.now()))
            del model,reference,grad
            gc.collect();torch.cuda.empty_cache()
    o.r.save(a.out/'status.json',dict(state='complete',gpu_released_on_exit=True,test_evaluations=0,time=o.r.now()))


if __name__=='__main__':main()
