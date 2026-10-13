"""CPU fresh-graph static/dynamic native export contract; no dataset/device/test."""
import argparse,math,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from PIL import Image
from caltech_repair234_model import CorrectedMoE,input_amplitude
import export_caltech_repair234 as e
import caltech_repair234 as r

def rejected(fn):
    try:fn()
    except ValueError:return True
    raise AssertionError('Invalid deployment input accepted')

def reverse(path,kind):
    height=1080 if kind=='amplitude' else 1200
    with Image.open(path) as im:
        assert im.size==(1920,height)
        return np.asarray(im.crop((452,(height-1016)//2,1468,(height+1016)//2)).resize((478,478),Image.Resampling.NEAREST)).copy()

def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);checks=[]
    try:
        torch.manual_seed(20261014);probes={'random':torch.randint(0,256,(1,3,150,150),dtype=torch.uint8),'dim':torch.full((1,3,150,150),4,dtype=torch.uint8),'black':torch.zeros((1,3,150,150),dtype=torch.uint8)}
        for depth in (2,3,4):
            for rho in (0.,.3):
                torch.manual_seed(17);model=CorrectedMoE(depth,rho).eval();name=f'L{depth}_rho{rho}';cp=out/(name+'.pt');torch.save(dict(model=model.state_dict(),scope='Synthetic fresh state, not a trained selection'),cp);cfg=dict(depth=depth,residual_rho=rho,scope='Synthetic export check');package=out/(name+'_static');manifest=e.export_selected(model,cfg,package,cp,e.sha(cp),[f'synthetic{k}' for k in range(20)],r.sources())
                for stage in manifest['stages']:
                    s=stage['stage'];gray=reverse(package/f'phase_{s}_native.bmp','phase');assert np.array_equal(gray,np.asarray(Image.open(package/f'phase_{s}_canonical.bmp')))
                with np.load(package/'amplitude_multipliers.npz') as z:
                    for stage in manifest['stages']:
                        s=stage['stage'];assert np.array_equal(z[s],model.plane(s).detach().abs().numpy()[20:498,20:498])
                assert manifest['contract']['graph'].startswith('RGB fixed input') and manifest['stages'][2]['input_size']==478
                rejected(lambda:e.encode_stage(model,torch.zeros(1,224,224),'1',torch.full((1,4),.5)))
                rejected(lambda:e.encode_stage(model,torch.zeros(1,478,478),'0',torch.full((1,4),.5)))
                rejected(lambda:e.encode_stage(model,torch.zeros(1,224,224),'0',torch.tensor([[1.,0.,0.,0.]])))
                rejected(lambda:e.amplitude_from_ccd(model,torch.full((1,478,478),float('nan')),'global'))
                for probe,image in probes.items():
                    amp=input_amplitude(image)
                    with torch.no_grad():reference=model(amp)[0]
                    current=[amp];weights=[None];calls=[];stats=[]
                    def replay(stage,a,p,ids):
                        encoded,phase=e.encode_stage(model,current[0],stage,weights[0]);assert np.array_equal(encoded,np.rint(a*255).astype(np.uint8)) and np.array_equal(phase,p)
                        dest=out/(name+'_'+probe+'_'+stage);receipt=e.save_sample_bmps(model,current[0],stage,dest,weights[0]);qa=reverse(dest/'amplitude_000000_native.bmp','amplitude');qp=reverse(dest/'phase_native.bmp','phase');assert np.array_equal(qa,encoded[0]) and np.array_equal(qp,phase);calls.append(stage);stats.append(receipt['brightness'][0])
                        field=torch.from_numpy(qa.astype(np.float32)/255)[None]*torch.exp(2j*math.pi*torch.from_numpy(qp.astype(np.float32)/256))[None];field=torch.nn.functional.pad(field,(20,20,20,20));intensity=model.propagator(field).abs().square()[:,20:498,20:498]
                        if stage=='router':weights[0]=e.routing_from_ccd(model,intensity)
                        elif int(stage)<depth-1:current[0]=e.amplitude_from_ccd(model,intensity,model.stages[int(stage)+1])
                        return intensity.numpy()
                    with torch.no_grad():actual=model(amp,callback=replay,ids=[probe])[0]
                    error=float((reference-actual).abs().max());assert error<=1e-6 and calls==['router']+[str(k) for k in range(depth)];assert torch.equal(weights[0],model.routing_weights)
                    if probe=='black':assert all(s['active']['maximum']==0 for s in stats) and torch.allclose(actual,torch.full_like(actual,.05),atol=1e-8,rtol=0)
                    checks.append(dict(depth=depth,rho=rho,probe=probe,passed=True,captures=len(calls),probability_max_error=error,native_static_dynamic_exact=True,routing_and_ccd_helpers_match=True,brightness=stats))
        r.save(out/'checks.json',dict(passed=True,checks=checks,source_sha256={n:r.sha(Path(__file__).with_name(n)) for n in ('export_caltech_repair234.py','check_caltech_repair234_export.py')},training_reference_sources=r.sources(),synthetic_static_packages=6,model_forward_calls=36,test_arrays_read=0,test_evaluations=0,devices_opened=0,gpu_used=False,scope='Fresh synthetic states, not selected trained weights; matching future trained-weight check still required',wrong_sizes_top2_and_nonfinite_rejected=True),exclusive=True)
    except BaseException as x:r.save(out/'failure.json',dict(error=repr(x),automatic_retry=False),exclusive=True);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);check(p.parse_args().out)
