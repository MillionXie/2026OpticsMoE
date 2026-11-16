"""Identity-aware evaluation and routing statistics for the MLP reference."""
from contextlib import contextmanager
from pathlib import Path
import csv
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from LightGenV2.tasks.t22_public_temporal_mlp.tools.data import groups,batch


@torch.inference_mode()
def evaluate(model,payload,device,batch_fields=8,directory=None,slot_shift=0):
    indices,valid=groups(payload,'test')
    predictions=[]
    targets=[]
    identities=[]
    selected={name:[] for name in ('vision','language')}
    counts={name:torch.zeros(16,4,dtype=torch.int64) for name in selected}
    for start in range(0,len(indices),batch_fields):
        ii=indices[start:start+batch_fields]
        vv=valid[start:start+batch_fields]
        if slot_shift:ii=ii.roll(slot_shift,1);vv=vv.roll(slot_shift,1)
        item=batch(payload,ii,vv,device)
        output=model.predict_from_embeddings(item['vision'])
        keep=item['valid']
        predictions.extend(output['prediction'][keep].cpu().tolist())
        targets.extend(item['target'][keep].cpu().tolist())
        identities.extend(payload['ids'][index] for index in ii[vv].tolist())
        for name,route in output['routing'].items():
            selection=route['selected']
            if name=='vision':value=(selection & keep[:,:,None,None]).sum((0,2))
            else:value=(selection & keep[:,:,None]).sum(0)
            counts[name]+=value.cpu()
            selected[name].append(selection[keep].cpu().reshape(-1,4))
    if len(identities)!=558 or len(set(identities))!=558:
        raise ValueError('Evaluation requires 558 distinct unpadded video identities')
    result=dict(videos=558,srcc=float(spearmanr(predictions,targets).statistic))
    if not np.isfinite(result['srcc']):raise ValueError('Nonfinite MLP evaluation score')
    for name,pieces in selected.items():
        choices=torch.cat(pieces)
        share=choices.float().mean(0)/2
        by_slot=counts[name].double()/counts[name].sum(-1,keepdim=True)
        codes=(choices.long()*torch.tensor([1,2,4,8])).sum(-1)
        codes=torch.bincount(codes,minlength=16)
        result[name+'_routing']=dict(selection_share=share.tolist(),selection_share_by_slot=by_slot.tolist(),
              maximum_global_share_deviation=float((share-.25).abs().max()),
              maximum_slot_share_deviation=float((by_slot-.25).abs().max()),
              unique_top2_pairs=int((codes>0).sum()),selection_variation=float(1-codes.max()/len(choices)))
    records=[dict(sample_id=name,target_MOS=target,predicted_MOS=prediction)
             for name,target,prediction in zip(identities,targets,predictions)]
    if directory is not None:
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'predictions.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
        (directory/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result,records


def qualifies(metrics,minimum=.790,maximum=.799):
    return minimum<=metrics['srcc']<=maximum and all(
        metrics[name+'_routing']['maximum_global_share_deviation']<=.05
        and metrics[name+'_routing']['maximum_slot_share_deviation']<=.10
        and metrics[name+'_routing']['unique_top2_pairs']==6 for name in ('vision','language'))


@contextmanager
def feature_noise(model,fraction,reference,seed):
    handles=[]
    for index,name in enumerate(('vision','language')):
        generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed+100*index)
        def perturb(module,arguments,output,name=name,generator=generator):
            noise=torch.randn(output.shape,dtype=output.dtype,device=output.device,generator=generator)
            return output+fraction*reference[name]*noise
        handles.append(getattr(model,name+'_global').register_forward_hook(perturb))
    try:yield
    finally:
        for handle in handles:handle.remove()
