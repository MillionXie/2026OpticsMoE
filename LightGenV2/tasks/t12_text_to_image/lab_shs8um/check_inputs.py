"""Read-only six-layer amplitude quantization audit; does not open hardware."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from .ccd_bridge import attach
from ..sealed_editor import build_sealed

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    model=build_sealed(torch.load(a.project/'assets/small.pt',map_location='cpu',weights_only=False)).cuda().eval()
    x=torch.load(a.inputs,map_location='cpu',weights_only=False)
    rows=[]
    def callback(stage,amplitude,phase,ideal):
        active=amplitude.detach().cpu().numpy();scale=float(active.max())
        stats=[]
        for i,value in enumerate(active):
            gray=np.rint(np.clip(value/scale,0,1)*255)
            stats.append(dict(index=i,max=int(gray.max()),mean=float(gray.mean()),p99=float(np.percentile(gray,99)),
                nonzero_fraction=float((gray>0).mean()),bright_fraction=float((gray>=128).mean()),
                sample_max_over_batch_max=float(value.max()/scale),mean_squared_amplitude=float(np.mean((gray/255)**2))))
        rows.append(dict(stage=stage,batch_amplitude_max=scale,images=stats))
        return ideal
    restore=attach(model,callback)
    values=[x[k].cuda() for k in ('reference','embeddings','mask','noise')];values[1]=values[1].float()
    with torch.inference_mode():model(*values)
    restore();print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
