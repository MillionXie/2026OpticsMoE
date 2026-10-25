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
from .static_four import build_static_four,CONTRACT as FOUR_CONTRACT
from .sensor_neutral import amplitude as neutral_amplitude
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
def extract(model,data,front,device,out,split,amplitude_fn=batch_amplitude):
    rows={n:[] for n in ('input','first','final')};route=[];stats={};logits=[]
    if model.architecture=='moe':rows['router']=[]
    for start in range(0,len(data['labels']),8):
        ix=np.arange(start,min(start+8,len(data['labels'])));amp=amplitude_fn(data,front,ix,device)
        r=model(amp,return_debug=True);logits.extend(r['logits'].cpu().tolist())
        rows['input'].append(F.adaptive_avg_pool2d(amp[:,None],(28,28))[:,0].flatten(1).cpu().numpy())
        for stage,ccd in r['ccd'].items():rows[stage].append(model.features_from_ccd(ccd).cpu().numpy())
        if r['route_power'] is not None:route.extend(r['route_power'].cpu().tolist())
        for stage,s in r['display_stats'].items():
            v=stats.setdefault(stage,dict(clipped=[],nonzero=[]));v['clipped'].append(s['clipped_fraction']);v['nonzero'].append(s['nonzero_fraction'])
        if start%800==0:put(out/'status.json',dict(state='extracting',split=split,seen=int(ix[-1]+1),updated=time.time()))
    values={k:np.concatenate(v) for k,v in rows.items()};values['frontend']=front.float().cpu().numpy();values.update(labels=data['labels'],domains=data['domains'],ids=data['ids'],logits=np.asarray(logits,dtype=np.float32))
    np.savez_compressed(out/(split+'_features.npz'),**values)
    return values,dict(scores=class_scores(values['logits'],data['labels'],data['domains']),display={n:{k:float(np.mean(v)) for k,v in row.items()} for n,row in stats.items()},routing_by_domain={name:dict(mean_power=np.asarray(route)[data['domains']==d].mean(0).tolist(),winner_fraction=(np.bincount(np.asarray(route)[data['domains']==d].argmax(1),minlength=len(model.active_indices))/(data['domains']==d).sum()).tolist()) for d,name in ((0,'RGB'),(1,'SAR'))} if route else None)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('run','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.run/'protocol.json').read_text());receipt=json.loads((a.run/'completion.json').read_text());cfg=protocol['config']
    cfg=dict(cfg,per_process_GPU_memory_fraction=.2);gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');path=a.run/'best_checkpoint.pt';parent_sha=sha(path)
    if parent_sha!=receipt['best_sha256']:raise ValueError('Parent changed')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['contract'] not in (CONTRACT,FOUR_CONTRACT):raise ValueError('Changed graph')
    four=state['contract']==FOUR_CONTRACT;factory=build_static_four if four else build_unified;amplitude_fn=neutral_amplitude if four else batch_amplitude;frontend_fn=cache_frontend
    if four:
        # Late import avoids the ridge-helper import cycle and pins THIS PT's
        # frozen frontend instead of silently assuming the first FB7D revision.
        from .diagnose_frontend_transfer import checked_features
        def frontend_fn(data,path):return checked_features(data,path,state['frontend_sha256'])
    model=factory(state['architecture']).to(device);model.load_state_dict(state['model'],strict=True);model.eval();a.out.mkdir(parents=True,exist_ok=False)
    original={n:tensor_sha(v) for n,v in model.state_dict().items()};train=load_arrays(a.data,'train');val=load_arrays(a.data,'validation');front=frontend_fn(train,a.frontend);vf=frontend_fn(val,a.frontend)
    tr,ts=extract(model,train,front,device,a.out,'train',amplitude_fn);va,vs=extract(model,val,vf,device,a.out,'validation',amplitude_fn);gradients=[]
    for label,d in (('RGB',[0]),('SAR',[1]),('joint',[0,1])):
        ix=next(balanced_batches(train,d,8,42));model.zero_grad(set_to_none=True);r=model(amplitude_fn(train,front,ix,device));loss=F.cross_entropy(r['logits'],torch.as_tensor(train['labels'][ix],device=device));loss.backward()
        gradients.append(dict(domain=label,loss=float(loss.detach()),groups={n:dict(norm=float(v.grad.norm()),maximum=float(v.grad.abs().max()),nonzero=float((v.grad!=0).float().mean())) for n,v in model.named_parameters() if v.requires_grad and v.grad is not None}))
    probes=[]
    for stage in ('frontend','input','first','final'):
        for domains in ([0,1],[0],[1]):
            ix=np.isin(tr['domains'],domains)
            for reg in (.001,.01,.1,1.):
                w=ridge_weights(tr[stage][ix],tr['labels'][ix],reg)
                probes.append(dict(stage=stage,training_domains=domains,regularization=reg,train=class_scores(tr[stage]@w.T,tr['labels'],tr['domains']),validation=class_scores(va[stage]@w.T,va['labels'],va['domains'])))
    if sha(path)!=parent_sha or any(tensor_sha(v)!=original[n] for n,v in model.state_dict().items()):raise RuntimeError('Read-only diagnostic changed parent')
    pairing={}
    if four:
        assert np.array_equal(train['domains'],np.tile([0,1],3000)) and np.array_equal(train['labels'][::2],train['labels'][1::2])
        ids=train['ids'].astype(str);assert [x.rsplit(':',1)[0] for x in ids[::2]]==[x.rsplit(':',1)[0] for x in ids[1::2]]
        for stage in ('frontend','input','first','final'):
            x=tr[stage].astype(np.float64);x-=x.mean(0,keepdims=True);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            same=(x[::2]*x[1::2]).sum(1);other=(x[::2]*np.roll(x[1::2],1,axis=0)).sum(1)
            pairing[stage]=dict(TRAIN_pair_cosine_mean=float(same.mean()),TRAIN_shuffled_cosine_mean=float(other.mean()),pair_cosine_std=float(same.std()))
    put(a.out/'report.json',dict(passed=True,contract=state['contract'],parent_SHA256=parent_sha,frontend_SHA256=state.get('frontend_sha256'),training=ts,validation=vs,gradient_groups=gradients,linear_probes=probes,TRAIN_pair_alignment=pairing,TEST_loaded=False,SDK=False,original_graph_unchanged=True,gpu=gpu,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
