"""TRAIN/validation controlled input layout comparison; never loads TEST."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .unified_native import build_unified
from .train_unified_static import load_arrays,cache_frontend,batch_amplitude
from .sensor_neutral import frontend,amplitude
from .diagnose_unified import ridge_weights,class_scores
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import gpu_gate,put,sha


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('parent','data','frontend','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False);protocol=json.loads((a.parent/'protocol.json').read_text());receipt=json.loads((a.parent/'completion.json').read_text());digest=sha(a.parent/'best_checkpoint.pt');assert digest==receipt['best_sha256']
    gpu_gate(protocol['config'],a.gpu_uuid);device=torch.device('cuda:0');model=build_unified(cp['architecture']).to(device);model.load_state_dict(cp['model'],strict=True);model.eval();a.out.mkdir(parents=True,exist_ok=False);scores=[];inputs={}
    for split in ('train','validation'):
        data=load_arrays(a.data,split);old=cache_frontend(data,a.frontend);new=frontend(data,a.frontend);pred={k:[] for k in ('old','sensor_neutral')};fields={k:[] for k in pred}
        for start in range(0,len(data['labels']),8):
            ix=np.arange(start,min(start+8,len(data['labels'])))
            for label,amp in (('old',batch_amplitude(data,old,ix,device)),('sensor_neutral',amplitude(data,new,ix,device))):
                pred[label].extend(model(amp)['logits'].cpu().tolist());fields[label].append(torch.nn.functional.adaptive_avg_pool2d(amp[:,None],(28,28))[:,0].flatten(1).cpu().numpy())
            if start%800==0:put(a.out/'status.json',dict(state='layout_comparison',split=split,seen=start+len(ix),updated=time.time()))
        inputs[split]={n:np.concatenate(v) for n,v in fields.items()};inputs[split].update(labels=data['labels'],domains=data['domains']);scores.append(dict(split=split,locked_parent_predictions={n:class_scores(v,data['labels'],data['domains']) for n,v in pred.items()}))
    probes=[]
    for encoding in ('old','sensor_neutral'):
        for domain in (0,1):
            tr=inputs['train'];va=inputs['validation'];mask=tr['domains']==domain
            for reg in (.001,.01,.1):
                weight=ridge_weights(tr[encoding][mask],tr['labels'][mask],reg);probes.append(dict(encoding=encoding,training_domain=domain,regularization=reg,validation=class_scores(va[encoding]@weight.T,va['labels'],va['domains'])))
    assert sha(a.parent/'best_checkpoint.pt')==digest
    put(a.out/'report.json',dict(passed=True,parent_SHA256=digest,controlled_scores=scores,input_linear_probes=probes,TEST_loaded=False,SDK=False,parent_unchanged=True,physical_graph_unchanged=True))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
