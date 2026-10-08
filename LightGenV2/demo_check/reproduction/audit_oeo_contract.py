"""CPU structural audit: same on/off initialization and actual per-layer OEO calls.

Only the build function's final Module.cuda() is replaced with identity so the
unchanged optical forward can run on CPU. No physical or numeric operator is mocked.
LayerNorm instrumentation delegates to the original function with unchanged inputs.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import bloodmnist_multiseed as m
from bloodmnist_multiseed import b,r,torch,np,F,PhaseLayer

def main():
    p=argparse.ArgumentParser();p.add_argument('--examples',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not torch.cuda.is_initialized();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);cfg=r.read(m.BASE)
    with np.load(a.examples,allow_pickle=False) as z:x=torch.from_numpy(z['amplitude_100'][:2].copy())
    results=[];norm=F.layer_norm
    for depth in [2,4,6]:
        for arch in ['moe','d2nn_wide']:
            for seed in [17,27,37]:
                models=[]
                with patch.object(torch.nn.Module,'cuda',lambda self,*args,**kwargs:self):
                    for name in [arch,arch+'_nooeo']:r.setseed(seed);models.append(m.build(name,depth,cfg))
                aa,bb=[dict(v.named_parameters()) for v in models];assert aa.keys()==bb.keys();assert all(torch.equal(aa[k],bb[k]) for k in aa);assert all(k.endswith('raw_phase') and v.device.type=='cpu' for k,v in aa.items())
                record=dict(arch=arch,depth=depth,seed=seed,on_off_initial_phases_bitwise_equal=True,parameters=sum(v.numel() for v in aa.values()),all_learned_parameters_are_phases=True)
                if seed==17:
                    traces=[]
                    for name,model,expected in zip([arch,arch+'_nooeo'],models,[depth,0]):
                        events=[];handles=[]
                        for key,module in model.named_modules():
                            if isinstance(module,PhaseLayer):
                                handles.append(module.register_forward_pre_hook(lambda q,inputs,key=key:events.append(dict(event='phase',module=key,shape=list(inputs[0].shape)))))
                        def counted(value,normalized_shape,*args,**kwargs):
                            events.append(dict(event='layer_norm',input_shape=list(value.shape),normalized_shape=list(normalized_shape)));return norm(value,normalized_shape,*args,**kwargs)
                        try:
                            with torch.no_grad(),patch.object(F,'layer_norm',counted):prob,cap,_=m.forward(model,x,name)
                        finally:
                            for handle in handles:handle.remove()
                        calls=[e for e in events if e['event']=='layer_norm'];assert len(calls)==expected,(name,depth,len(calls));assert all(e['normalized_shape']==[498,498] for e in calls);assert torch.isfinite(prob).all() and prob.shape==(2,8)
                        phases=sum(e['event']=='phase' for e in events);assert phases==(1+5*depth if arch=='moe' else depth),(arch,phases)
                        traces.append(dict(arm=name,observed_oeo_calls=len(calls),observed_phase_calls=phases,events=events))
                    record['forward_traces']=traces
                results.append(record);del models,aa,bb
    assert not torch.cuda.is_initialized();r.save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=m.sources(),auditor_sha256=r.sha(__file__),input_examples_sha256=r.sha(a.examples),device='CPU only',torch=torch.__version__,visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),scope=__doc__));r.save(a.out/'results.json',results);r.save(a.out/'status.json',dict(state='complete',initialization_pairs=18,forward_runs=12,time=r.now()));print(json.dumps(dict(initialization_pairs=18,forward_runs=12,passed=True)))
if __name__=='__main__':main()
