"""Paired class-balanced TRAIN gradients, no fitting or TEST access."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .static_four import build_static_four,CONTRACT
from .diagnose_frontend_transfer import checked_features
from .sensor_neutral import amplitude
from .train_unified_static import load_arrays
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha,gpu_gate


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('run','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.run/'protocol.json').read_text());receipt=json.loads((a.run/'completion.json').read_text());path=a.run/'best_checkpoint.pt';digest=sha(path);assert digest==receipt['best_sha256'] and receipt['contract']==CONTRACT
    cfg=dict(protocol['config'],per_process_GPU_memory_fraction=.2);gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');cp=torch.load(path,map_location='cpu',weights_only=False);m=build_static_four(cp['architecture']).to(device);m.load_state_dict(cp['model'],strict=True);m.eval();before={n:tensor_sha(v) for n,v in m.state_dict().items()};data=load_arrays(a.data,'train');front=checked_features(data,a.frontend,cp['frontend_sha256'])
    rgb=np.concatenate([np.flatnonzero((data['labels']==c)&(data['domains']==0))[:5] for c in range(10)])
    sar=rgb+1;assert np.array_equal(data['labels'][rgb],data['labels'][sar]) and np.all(data['domains'][sar]==1)
    ids=data['ids'].astype(str);assert [v.rsplit(':',1)[0] for v in ids[rgb]]==[v.rsplit(':',1)[0] for v in ids[sar]]
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();gradients=[];losses=[]
    for domain,indices in enumerate((rgb,sar)):
        acc={n:torch.zeros_like(v) for n,v in m.named_parameters() if v.requires_grad};total=0.
        for chunk in np.array_split(indices,10):
            m.zero_grad(set_to_none=True);result=m(amplitude(data,front,chunk,device));loss=F.cross_entropy(result['logits'],torch.as_tensor(data['labels'][chunk],device=device));loss.backward();total+=float(loss.detach())*len(chunk)
            for n,v in m.named_parameters():
                if v.requires_grad:
                    assert v.grad is not None and torch.isfinite(v.grad).all();acc[n]+=v.grad.detach()*len(chunk)/len(indices)
        gradients.append(acc);losses.append(total/len(indices));put(a.out/'status.json',dict(completed_domains=domain+1,updated=time.time()))
    groups={n:dict(RGB_norm=float(v.norm()),SAR_norm=float(gradients[1][n].norm()),cosine=float(F.cosine_similarity(v.flatten(),gradients[1][n].flatten(),dim=0,eps=1e-20))) for n,v in gradients[0].items()}
    assert sha(path)==digest and all(tensor_sha(v)==before[n] for n,v in m.state_dict().items())
    put(a.out/'report.json',dict(parent_SHA256=digest,TRAIN_unique_locations=50,TRAIN_views=100,ids=data['ids'][np.stack((rgb,sar),1).reshape(-1)].tolist(),TRAIN_losses=losses,gradient_groups=groups,TEST_loaded=False,parameter_updates=False,SDK=False,gpu=gpu,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seconds=time.time()-started))


if __name__=='__main__':main()
