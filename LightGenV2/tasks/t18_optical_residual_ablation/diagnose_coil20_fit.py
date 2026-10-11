"""Fresh L16 train-only objective gradients; no checkpoints, fitting, val or test."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import fast_optical as execution
from medicinal_pilot import load_fit_data

o, torch = execution.optical, execution.torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.gpu
    args.out.mkdir(parents=True, exist_ok=False)
    profile = o.r.read(args.profile)
    manifest = o.r.read(args.data.parent/'data_manifest.json')
    assert o.r.sha(args.data) == manifest['cache_sha256']
    o.f.gpu_budget.ALLOWED.update(profile['gpu_uuids'])
    policy = o.f.gpu_budget.configure(torch, args.gpu, shared=True, memory_gib=3)
    torch.set_num_threads(4)
    sources = o.source_identity(args.profile)
    for name in ['fast_optical.py', 'medicinal_pilot.py', 'prepare_coil20.py', Path(__file__).name]:
        sources[str((o.HERE/name).relative_to(o.ROOT)).replace('\\', '/')] = o.r.sha(o.HERE/name)
    assert all(hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:'+p])).hexdigest() == s for p,s in sources.items())
    o.r.save(args.out/'metadata.json', dict(command=sys.argv, pid=os.getpid(), time=o.r.now(),
        sources=sources, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        data_sha256=manifest['cache_sha256'], gpu_policy=policy))
    data = load_fit_data(args.data, 'train', manifest)
    # Four distinct train labels; index selection precedes and is independent of predictions.
    indices = [int(torch.where(data[1] == label)[0][0]) for label in [0, 6, 12, 19]]
    weights = (len(data[1])/(20*torch.bincount(data[1], minlength=20).float())).cuda()
    results, identities = [], {}
    for init_range in [1., .1]:
        for rho in [0., .3]:
            cfg = o.config(profile, manifest, rho, 'base')
            cfg['phase_init_raw_uniform'] = init_range
            o.r.setseed(17)
            model = execution.build(16, profile, cfg, batched=True)
            identity = o.f.tensor_identity(model)
            if init_range in identities: assert identity == identities[init_range]
            else: identities[init_range] = identity
            gradients = {name: None for name in ['nll', 'capture', 'smooth']}
            values = {name: 0. for name in gradients}
            probabilities, energies, totals = [], [], []
            parameters = list(model.parameters())
            for index in indices:
                out = model(o.encode(data[0][index:index+1]))
                e, intensity = out['energies'], out['intensity']
                p = (e+1e-12)/(e.sum(1, keepdim=True)+20e-12)
                capture = (intensity*model.masks.sum(0)).sum((-1,-2))/intensity.sum((-1,-2)).clamp_min(1e-12)
                y = data[1][index:index+1].cuda()
                target = torch.nn.functional.one_hot(y, 20)*(1-cfg['label_smoothing'])+cfg['label_smoothing']/20
                terms = dict(nll=(-(target*p.clamp_min(1e-12).log()).sum(1)*weights[y]).mean(),
                    capture=-cfg['capture_weight']*capture.clamp_min(1e-12).log().mean(),
                    smooth=cfg['phase_smooth_weight']*o.b.phase_smoothness(model))
                for name, loss in terms.items():
                    gs = torch.autograd.grad(loss, parameters, retain_graph=name != 'smooth')
                    vector = torch.cat([g.detach().flatten().cpu() for g in gs])/len(indices)
                    gradients[name] = vector if gradients[name] is None else gradients[name]+vector
                    values[name] += float(loss.detach())/len(indices)
                probabilities.append(p.detach().cpu()); energies.append(float(e.detach().sum()))
                totals.append(float(intensity.detach().sum()))
            ps = torch.cat(probabilities)
            norms = {name: float(g.norm()) for name,g in gradients.items()}
            assert all(bool(torch.isfinite(g).all()) for g in gradients.values())
            results.append(dict(rho=rho, depth=16, phase_init_raw_uniform=init_range,
                initial_parameter_sha256=identity, sample_ids=[str(data[2][i]) for i in indices],
                labels=[int(data[1][i]) for i in indices], objective_values=values, gradient_norms=norms,
                nll_capture_cosine=float(torch.nn.functional.cosine_similarity(gradients['nll'], gradients['capture'], dim=0)),
                probability_min=float(ps.min()), probability_max=float(ps.max()),
                max_pairwise_probability_difference=float((ps[:,None]-ps[None,:]).abs().max()),
                mean_entropy=float(-(ps*ps.clamp_min(1e-12).log()).sum(1).mean()),
                detector_energy_sums=energies, output_intensity_sums=totals,
                epsilon_fraction_max=max(20e-12/(e+20e-12) for e in energies)))
            del model, parameters, gradients, out, terms, gs, vector
            gc.collect(); torch.cuda.empty_cache()
    o.r.save(args.out/'diagnostic.json', dict(results=results, test_evaluations=0,
        scope='Fresh seed17 L16 four prespecified train samples; no fitting, val, checkpoint or test. Not causal proof.'))
    o.r.save(args.out/'status.json', dict(state='complete', test_evaluations=0, time=o.r.now()))


if __name__ == '__main__': main()
