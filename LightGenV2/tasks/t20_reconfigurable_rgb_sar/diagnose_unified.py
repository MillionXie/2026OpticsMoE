"""TRAIN/validation-only localization of information and optimization bottlenecks.

Read-only parent graph; no TEST loading, SDK, source/model mutation or new layers.
Ridge probes are diagnostic linear decoders, not reported optical model results.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_unified_static import load_arrays,cache_frontend,batch_amplitude,balanced_batches
from .unified_native import build_unified,CONTRACT
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import gpu_gate,put,sha,tensor_sha


def ridge_weights(x,y,regularization):
    """RMS-preconditioned ridge; maps directly to existing no-bias Linear head."""
    x=np.asarray(x,dtype=np.float64);scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-6)
    z=x/scale;target=np.eye(10,dtype=np.float64)[np.asarray(y)]
    gram=z.T@z/len(z);rhs=z.T@target/len(z)
    w=np.linalg.solve(gram+regularization*np.eye(z.shape[1]),rhs)/scale[:,None]
    return w.T.astype(np.float32)


def class_scores(logits,y,domains):
    p=np.asarray(logits).argmax(1);result={}
    for d,name in ((0,'RGB'),(1,'SAR')):
        ix=np.asarray(domains)==d
        if not ix.any():continue
        cm=np.zeros((10,10),dtype=int);np.add.at(cm,(np.asarray(y)[ix],p[ix]),1)
        result[name]=dict(n=int(ix.sum()),accuracy=float((p[ix]==np.asarray(y)[ix]).mean()),confusion_matrix=cm.tolist())
    return result


@torch.no_grad()
def extract(model,data,front,device,out,split):
    rows={n:[] for n in ('input','first','final')};route=[];stats={};logits=[]
    if model.architecture=='moe':rows['router']=[]
    for start in range(0,len(data['labels']),8):
        ix=np.arange(start,min(start+8,len(data['labels'])));amp=batch_amplitude(data,front,ix,device)
        r=model(amp,return_debug=True);logits.extend(r['logits'].cpu().tolist())
        rows['input'].append(F.adaptive_avg_pool2d(amp[:,None],(28,28))[:,0].flatten(1).cpu().numpy())
        for stage,ccd in r['ccd'].items():rows[stage].append(model.features_from_ccd(ccd).cpu().numpy())
        if r['route_power'] is not None:route.extend(r['route_power'].cpu().tolist())
        for stage,s in r['display_stats'].items():
            v=stats.setdefault(stage,dict(clipped=[],nonzero=[]));v['clipped'].append(s['clipped_fraction']);v['nonzero'].append(s['nonzero_fraction'])
        if start%800==0:put(out/'status.json',dict(state='extracting',split=split,seen=int(ix[-1]+1),updated=time.time()))
    values={k:np.concatenate(v) for k,v in rows.items()};values.update(labels=data['labels'],domains=data['domains'],ids=data['ids'],logits=np.asarray(logits,dtype=np.float32))
    np.savez_compressed(out/(split+'_features.npz'),**values)
    return values,dict(scores=class_scores(values['logits'],data['labels'],data['domains']),display={n:{k:float(np.mean(v)) for k,v in row.items()} for n,row in stats.items()},routing_by_domain={name:dict(mean_power=np.asarray(route)[data['domains']==d].mean(0).tolist(),winner_fraction=(np.bincount(np.asarray(route)[data['domains']==d].argmax(1),minlength=16)/(data['domains']==d).sum()).tolist()) for d,name in ((0,'RGB'),(1,'SAR'))} if route else None)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('run','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.run/'protocol.json').read_text());receipt=json.loads((a.run/'completion.json').read_text());cfg=protocol['config']
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');path=a.run/'best_checkpoint.pt';parent_sha=sha(path)
    if parent_sha!=receipt['best_sha256']:raise ValueError('Parent changed')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['contract']!=CONTRACT:raise ValueError('Changed graph')
    model=build_unified(state['architecture']).to(device);model.load_state_dict(state['model'],strict=True);model.eval();a.out.mkdir(parents=True,exist_ok=False)
    original={n:tensor_sha(v) for n,v in model.state_dict().items()};train=load_arrays(a.data,'train');val=load_arrays(a.data,'validation');front=cache_frontend(train,a.frontend);vf=cache_frontend(val,a.frontend)
    tr,ts=extract(model,train,front,device,a.out,'train');va,vs=extract(model,val,vf,device,a.out,'validation');gradients=[]
    for label,d in (('RGB',[0]),('SAR',[1]),('joint',[0,1])):
        ix=next(balanced_batches(train,d,8,42));model.zero_grad(set_to_none=True);r=model(batch_amplitude(train,front,ix,device));loss=F.cross_entropy(r['logits'],torch.as_tensor(train['labels'][ix],device=device));loss.backward()
        gradients.append(dict(domain=label,loss=float(loss.detach()),groups={n:dict(norm=float(v.grad.norm()),maximum=float(v.grad.abs().max()),nonzero=float((v.grad!=0).float().mean())) for n,v in model.named_parameters() if v.requires_grad and v.grad is not None}))
    probes=[]
    for stage in ('input','first','final'):
        for domains in ([0,1],[0],[1]):
            ix=np.isin(tr['domains'],domains)
            for reg in (.001,.01,.1,1.):
                w=ridge_weights(tr[stage][ix],tr['labels'][ix],reg)
                probes.append(dict(stage=stage,training_domains=domains,regularization=reg,train=class_scores(tr[stage]@w.T,tr['labels'],tr['domains']),validation=class_scores(va[stage]@w.T,va['labels'],va['domains'])))
    if sha(path)!=parent_sha or any(tensor_sha(v)!=original[n] for n,v in model.state_dict().items()):raise RuntimeError('Read-only diagnostic changed parent')
    put(a.out/'report.json',dict(passed=True,parent_SHA256=parent_sha,training=ts,validation=vs,gradient_groups=gradients,linear_probes=probes,TEST_loaded=False,SDK=False,original_graph_unchanged=True,gpu=gpu,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
