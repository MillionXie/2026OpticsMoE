"""Locate detector signal loss before/after OEO and inspect classification gradients."""
import argparse
import json
import subprocess
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import OpticalOEO, TextEncoder, encode, loss
from .run import load_data, setseed
from .prepare import save,digest

def main():
    p=argparse.ArgumentParser()
    for key in ['data','run','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--oeo-activation',choices=['relu','softplus','centered_leaky_relu','intensity_softsign'],default='relu')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    vocab=json.loads((a.data/'vocab.json').read_text());data=load_data(a.data,'val',vocab,'cuda')
    cfg=json.loads((a.run/'metadata.json').read_text())['config']
    front=TextEncoder(len(vocab),'fixed').cuda()
    field=encode(data['images'][data['index'][:32]],front(data['ids'][:32]),cfg['input_layout'])
    labels=data['labels'][:32];results={}
    for arch in ['moe','d2nn']:
        for state in ['initial','best','last']:
            setseed(17);model=OpticalOEO(arch,17,input_layout=cfg['input_layout'],oeo_activation=a.oeo_activation).cuda();model.eval()
            sha=None
            if state!='initial':
                path=a.run/'fixed'/arch/(state+'_checkpoint.pt');sha=digest(path.read_bytes())
                model.load_state_dict(torch.load(path,weights_only=False)['model'])
            original=model.oeo;stages=[]
            def trace(x):
                output=original(x)
                with torch.no_grad():
                    intensity=x.abs().square()
                    active=intensity[:,20:498,20:498]
                    z=F.layer_norm(active/active.mean((-2,-1),keepdim=True).clamp_min(1e-20),(478,478),eps=1e-5)
                    before=model.detect(intensity,model.class_centers,64)
                    after=model.detect(output.abs().square(),model.class_centers,64)
                    maxima=[];positive=[]
                    for y,xx in model.class_centers:
                        roi=z[:,y-20-32:y-20+32,xx-20-32:xx-20+32]
                        maxima.append(float(roi.max()));positive.append(float((roi>0).float().mean()))
                    stages.append(dict(layer=len(stages)+1,raw_window_energy_mean=before.mean(0).tolist(),
                         post_oeo_window_energy_mean=after.mean(0).tolist(),
                         pre_oeo_zero_window_fraction=float((before.sum(1)==0).float().mean()),
                         post_oeo_zero_window_fraction=float((after.sum(1)==0).float().mean()),
                         normalized_window_max=maxima,positive_window_pixel_fraction=positive))
                return output
            model.oeo=trace
            output=model(field);cost=loss(output,labels);cost.backward()
            results[arch+'/'+state]=dict(checkpoint_sha256=sha,stages=stages,nll=float(cost),
                 yes_probability_std=float(output['probabilities'][:,1].std()),
                 route_mean=output['route_power'].mean(0).detach().tolist(),
                 gradient_norms={name:float(param.grad.norm()) if param.grad is not None else None for name,param in model.named_parameters()})
            del model;torch.cuda.empty_cache()
    save(a.out/'probe.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         source_sha256=digest(Path(__file__).read_bytes()),data_manifest_sha256=digest((a.data/'manifest.json').read_bytes()),
         protocol='First 32 validation questions in stored order; no test access; no weight updates',oeo_activation=a.oeo_activation,results=results))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
