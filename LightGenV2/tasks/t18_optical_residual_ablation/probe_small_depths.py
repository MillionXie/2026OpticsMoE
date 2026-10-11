"""Fresh train-only deeper execution gates; no fitting or test evaluation."""
import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
import time
import fast_optical as execution
from medicinal_pilot import load_fit_data

o, torch = execution.optical, execution.torch


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--gpu', required=True); a = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == a.gpu
    a.out.mkdir(parents=True, exist_ok=False)
    profile = o.r.read(a.profile); manifest = o.r.read(a.data.parent/'data_manifest.json')
    o.f.gpu_budget.ALLOWED.update(profile['gpu_uuids'])
    policy = o.f.gpu_budget.configure(torch, a.gpu, shared=True, memory_gib=3)
    sources = o.source_identity(a.profile)
    sources.update({str((o.HERE/name).relative_to(o.ROOT)).replace('\\', '/'): o.r.sha(o.HERE/name)
        for name in ['fast_optical.py', 'medicinal_pilot.py', 'prepare_coil20.py', 'probe_small_depths.py']})
    o.r.save(a.out/'metadata.json', dict(command=sys.argv, pid=os.getpid(), sources=sources,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        data_sha256=o.r.sha(a.data), python=sys.version, torch=torch.__version__, cuda_build=torch.version.cuda, time=o.r.now()))
    torch.set_num_threads(4)
    data = load_fit_data(a.data, 'train', manifest)
    amplitude = o.encode(data[0][:8]); labels = data[1][:8].cuda()
    count = len(manifest['classes'])
    weights = (len(data[1])/(count*torch.bincount(data[1], minlength=count).float())).cuda()
    results = []
    for depth in [4, 8, 16]:
        for rho in [0., .3]:
            cfg = o.config(profile, manifest, rho, 'base'); o.r.setseed(17)
            model = execution.build(depth, profile, cfg, batched=False)
            def compute(batch):
                model.zero_grad(set_to_none=True); probabilities = []
                for start in range(0, 8, batch):
                    end = min(start+batch, 8)
                    probability, capture = o.forward(model, amplitude[start:end])
                    (o.objective(model, probability, capture, labels[start:end], weights, cfg)*(end-start)/8).backward()
                    probabilities.append(probability.detach())
                return torch.cat(probabilities), {n: p.grad.detach().clone() for n, p in model.named_parameters()}
            reference, gradients = compute(1)
            initial_sha = o.f.tensor_identity(model)
            execution.enable_expert_batching(model)
            assert o.f.tensor_identity(model) == initial_sha
            for micro in [1, 4]:
                torch.cuda.synchronize(); started = time.perf_counter()
                prob, grad = compute(micro)
                torch.cuda.synchronize(); seconds = (time.perf_counter()-started)/8
                error = float((reference-prob).abs().max())
                relative = max(float((grad[n]-g).norm()/g.norm().clamp_min(1e-12)) for n, g in gradients.items())
                cosine = min(float(torch.nn.functional.cosine_similarity(grad[n].flatten(), g.flatten(), dim=0)) for n, g in gradients.items())
                finite = all(bool(torch.isfinite(g).all()) and float(g.norm()) > 0 for g in grad.values())
                result = dict(depth=depth, rho=rho, expert_batched=True, microbatch=micro,
                    initial_parameter_sha256=initial_sha, parameters=sum(p.numel() for p in model.parameters()),
                    probability_max_error=error, gradient_max_relative_l2=relative, gradient_min_cosine=cosine,
                    gradients_finite_nonzero=finite, passed=error<=1e-6 and relative<=.02 and cosine>=.999 and finite,
                    train_core_seconds_per_sample=seconds,
                    epoch_projection_with_50pct_margin_seconds=len(data[1])*seconds*1.5,
                    projection_scope='8 fresh train samples, encoded inputs already on GPU; 50% heuristic margin, not measured full epoch')
                results.append(result); print(json.dumps(result), flush=True)
                o.r.save(a.out/'probe.json', dict(results=results, gpu_policy=policy, test_evaluations=0,
                    data_sha256=o.r.sha(a.data), source_sha256=o.r.sha(Path(__file__)), scope='Fresh train-only numerical/throughput probe, not accuracy'))
                del prob, grad
            del model, reference, gradients; gc.collect(); torch.cuda.empty_cache()
    o.r.save(a.out/'status.json', dict(state='complete', test_evaluations=0, time=o.r.now()))


if __name__ == '__main__': main()
