"""Fixed-checkpoint phase and routing diagnostics on train/validation only."""
import argparse,json,subprocess,sys
from pathlib import Path
import bloodmnist_experiment as b
from bloodmnist_experiment import r,torch,np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();meta=r.read(a.run/'metadata.json');cfg=meta['config'];lock=r.read(a.run/'test_lock.json');assert r.sha(a.data)==lock['data_sha256'];torch.set_num_threads(4);train=b.load_data(a.data,'train');val=b.load_data(a.data,'val');reports=[]
    for rel,digest in lock['sources'].items():assert r.sha(b.TASK/rel)==digest,rel
    for spec in lock['models']:
        if spec['arch']=='cnn':continue
        dest=a.run/spec['name'];assert r.sha(dest/'best_checkpoint.pt')==spec['checkpoint_sha256'];r.setseed(spec['seed']);m=b.build(spec['arch'],spec['depth'],cfg);initial={n:p.detach().clone() for n,p in m.named_parameters()};ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);m.eval();phase={}
        for n,param in m.named_parameters():
            difference=param.detach()-initial[n];s=param.detach().sigmoid();phase[n]=dict(elements=param.numel(),changed_elements=int((difference.abs()>1e-7).sum()),rms_update=float(difference.square().mean().sqrt()),saturation_fraction=float(((s<.01)|(s>.99)).float().mean()))
        assert all(v['rms_update']>0 for v in phase.values())
        # Each selected phase must show finite gradients; zero on an individual
        # representative batch is recorded, not silently treated as a failure.
        labels=train[1].cpu().numpy();norms={n:0. for n in phase}
        for offset in [0,2]:
            idx=np.concatenate([np.flatnonzero(labels==k)[offset:offset+2] for k in range(8)]);m.zero_grad(set_to_none=True);prob,c,_=b.forward(m,b.encode(train[0][idx]),spec['arch']);loss=-prob[torch.arange(len(idx)),train[1][idx]].clamp_min(1e-12).log().mean()-.2*c.clamp_min(1e-12).log().mean();loss.backward()
            for n,param in m.named_parameters():
                v=float(param.grad.norm());assert np.isfinite(v);norms[n]+=v
        for n in phase:phase[n]['two_representative_batch_gradient_norm_sum']=norms[n]
        vm,_=b.evaluate(m,val,spec['arch'],cfg['batch_size']);assert vm==r.read(dest/'summary.json')['metrics']['val'];report=dict(model=spec['name'],phase=phase,validation_replayed=True)
        if spec['arch']=='moe':
            prompt=m.net.prompt;original=prompt.routing;probs={}
            with torch.no_grad():
                for split,data in [('train',train),('val',val)]:
                    buf=[]
                    for idx in torch.arange(len(data[1])).split(cfg['batch_size']):buf.append(original(b.encode(data[0][idx]))['probabilities'].cpu())
                    probs[split]=torch.cat(buf)
                mean=probs['train'].mean(0).cuda()
                def fixed(images):
                    out=original(images);weights=mean.expand(len(images),-1);out.update(probabilities=weights,weights=weights,transmission=prompt.transmission(weights),prompt_amplitude=prompt.amplitude_map(weights));return out
                prompt.routing=fixed
                try:fixed_metrics,_=b.evaluate(m,val,'moe',cfg['batch_size'])
                finally:delattr(prompt,'routing')
                q=probs['val'];power=q.square()/q.square().sum(1,keepdim=True);vy=val[1].cpu();report['routing']=dict(dynamic_validation=vm,fixed_training_mean_validation=fixed_metrics,train_mean_probability=mean.cpu().tolist(),class_mean_power={str(k):power[vy==k].mean(0).tolist() for k in range(8)},power_std=power.std(0,unbiased=False).tolist(),mean_normalized_power_entropy=float(-(power*power.clamp_min(1e-12).log()).sum(1).mean()/np.log(9)))
        reports.append(report);del m,initial,ck;torch.cuda.empty_cache()
    r.save(a.run/'selected_optics_audit.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=r.sha(__file__),time=r.now(),test_arrays_read=False,scope='Fixed selected checkpoints; train/validation diagnostics, not another D2NN baseline',models=reports));print(json.dumps(dict(models=len(reports),all_selected_phase_tensors_updated=True)))

if __name__=='__main__':main()
