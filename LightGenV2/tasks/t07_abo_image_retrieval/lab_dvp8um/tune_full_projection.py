"""Adapt the original last projection only, using replayed TRAIN CCD inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=50)
    a=p.parse_args()
    import four_image_flow as flow
    import full_query_flow as pipe
    from transformers import AutoProcessor
    a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42); torch.set_num_threads(4)
    protocol=json.loads((flow.PROJECT/'protocol.json').read_text())
    train=[r for r in protocol['rows'] if r['split']=='train']
    query=[r for r in protocol['rows'] if r['split']=='query'];rows=train+query
    cache=torch.load(a.capture/'features.pt',map_location='cpu',weights_only=True)
    assert cache['ids']==[r['sample_id'] for r in rows]
    source=flow.PROJECT/'assets/best.pt';assert flow.sha(source)==flow.BEST
    payload=torch.load(source,map_location='cpu',weights_only=True)
    model=flow.OpticalRetrieval(payload['metadata']).cuda().eval().requires_grad_(False)
    model.load_state_dict(payload['state_dict'],strict=True)
    assert isinstance(model.readout.projection,torch.nn.Linear)
    processor=AutoProcessor.from_pretrained(str(flow.PROJECT/'assets/processor'),local_files_only=True)
    pipe.STAGE_CALIBRATION.update(json.loads((a.capture/'run_contract.json').read_text())['stage_calibration'])
    phase_paths={s:a.capture/'phase'/(s+'.bmp') for s in pipe.STAGE_CALIBRATION}
    def capture(bench,out,stage,phase,active,ids,orientation):
        x=np.stack([np.asarray(Image.open(a.capture/'ccd'/stage/(i+'.png'))) for i in ids])
        return torch.from_numpy(x).cuda().float(),{},[]
    pipe.capture_stage=capture
    def sim(model,batch):
        n=len(batch['input_ids'])
        return {**{s:torch.zeros(n,478,478) for s in pipe.STAGE_CALIBRATION},'descriptor':torch.zeros(n,64)}
    pipe.snapshot_simulation=sim
    inputs=[];descriptors=[]
    hook=model.readout.projection.register_forward_pre_hook(lambda module,args: inputs.append(args[0].detach().cpu().clone()))
    for start in range(0,len(rows),4):
        descriptors.append(pipe.process_batch(model,processor,None,a.output,phase_paths,rows[start:start+4])['descriptor'])
        if start%160==0:print(json.dumps(dict(replay_completed=start+4,total=len(rows))),flush=True)
    hook.remove()
    original_error=float((torch.cat(descriptors)-cache['descriptors']).abs().max())
    assert original_error<0.0001,original_error
    features=torch.cat(inputs).float().cuda()
    head=copy.deepcopy(model.readout.projection).requires_grad_(True)
    classes={v:i for i,v in enumerate(sorted({r['product_id'] for r in train}))}
    y=torch.tensor([classes[r['product_id']] for r in train],device='cuda')
    fit=[];val=[]
    for product in classes:
        idx=[i for i,r in enumerate(train) if r['product_id']==product]
        idx.sort(key=lambda i:hashlib.sha256(('physical-head42:'+train[i]['sample_id']).encode()).hexdigest())
        assert len(idx)==8;fit+=idx[:6];val+=idx[6:]
    fit_t=torch.tensor(fit,device='cuda');val_t=torch.tensor(val,device='cuda')
    positive=y[fit_t,None].eq(y[fit_t][None,:]);diagonal=torch.eye(len(fit),device='cuda',dtype=torch.bool)
    positive=positive&~diagonal
    anchors=cache['descriptors'][fit].float().cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=0.0001,weight_decay=0)
    def validate():
        z=F.normalize(head(features[:1600]),dim=-1)
        top=(z[val_t]@z[fit_t].T).argmax(1)
        return float(y[fit_t][top].eq(y[val_t]).float().mean())
    with torch.no_grad():best_score=validate()
    best_epoch=0;best=copy.deepcopy(head.state_dict());history=[dict(epoch=0,validation_r1=best_score)]
    for epoch in range(1,a.epochs+1):
        optimizer.zero_grad();z=F.normalize(head(features[fit_t]),dim=-1)
        logits=(z@z.T/0.1).masked_fill(diagonal,-1e4)
        loss=-(logits.log_softmax(1)*positive).sum(1).div(positive.sum(1)).mean()+0.05*(1-(z*anchors).sum(1)).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);optimizer.step()
        with torch.no_grad():score=validate()
        row=dict(epoch=epoch,loss=float(loss.detach()),validation_r1=score);history.append(row);print(json.dumps(row),flush=True)
        if score>best_score:best_score=score;best_epoch=epoch;best=copy.deepcopy(head.state_dict())
    def save(name,state):
        out=dict(payload,state_dict=dict(payload['state_dict']))
        for k,v in state.items():out['state_dict']['readout.projection.'+k]=v.cpu()
        torch.save(out,a.output/name)
    save('last.pt',head.state_dict());save('best.pt',best);head.load_state_dict(best)
    with torch.no_grad():verified=F.normalize(head(features),dim=-1).float().cpu()
    predictions,metrics=pipe.evaluate(verified[1600:],query,protocol,{'ids':[r['sample_id'] for r in train],'vectors':verified[:1600]})
    flow.write(a.output/'report.json',dict(status='complete',metrics=metrics,predictions=predictions,best_epoch=best_epoch,
        validation_r1=best_score,history=history,train_ids=[train[i]['sample_id'] for i in fit],validation_ids=[train[i]['sample_id'] for i in val],
        test_used_for_selection=False,physical_gallery=True,query_count=800,gallery_count=1600,
        source_sha256=flow.sha(source),best_sha256=flow.sha(a.output/'best.pt'),original_replay_max_error=original_error,
        changed_parameters=['readout.projection.weight','readout.projection.bias'],
        tuned_parameter_count=sum(p.numel() for p in head.parameters()),added_inference_parameters=0))
    torch.save(dict(ids=cache['ids'],descriptors=verified),a.output/'verified_features.pt')
    print(json.dumps(dict(status='complete',metrics=metrics)),flush=True)


if __name__=='__main__':main()
