"""GPU six-stage consistency test without opening any optical hardware."""
import argparse,json
from pathlib import Path
import torch
from .bounded_amplitude import encode
from .ccd_bridge import attach
from ..sealed_editor import build_sealed

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    for kind in ('rational','tanh'):
        spec=dict(kind=kind,scale=.5)
        z=torch.tensor([0.,.01,.1,.5,1.,5.,10.],requires_grad=True)
        amplitude=encode(z,spec)
        assert amplitude[0]==0 and bool((amplitude>=0).all()) and bool((amplitude<=1).all())
        assert bool((amplitude[1:]>=amplitude[:-1]).all())
        amplitude.sum().backward();assert bool(torch.isfinite(z.grad).all()) and bool((z.grad[1:4]>0).all())
        reconstructed=(amplitude.detach()*255).round()/255
        assert float((reconstructed-amplitude.detach()).abs().max())<=.5/255+1e-7
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);saved['bounded_amplitude']=spec
        model=build_sealed(saved).cuda().eval();x=torch.load(a.inputs,map_location='cpu',weights_only=False)
        values=[x[k][:1].cuda() for k in ('reference','embeddings','mask','noise')];values[1]=values[1].float()
        with torch.no_grad():expected=model(*values)
        stages=[]
        def callback(stage,amplitude,phase,ideal):
            assert float(amplitude.min())>=0 and float(amplitude.max())<=1.000001
            assert bool(torch.isfinite(amplitude).all())
            stages.append(stage);return ideal
        restore=attach(model,callback)
        with torch.no_grad():actual=model(*values)
        restore();assert len(stages)==6 and float((actual-expected).abs().max())<1e-6
        print(json.dumps(dict(mapping=kind,stages=stages,max_output_error=float((actual-expected).abs().max()),status='passed')),flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
