"""Replay exact six CCD stages; adapt final projection on physical TRAIN only."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import layerwise as lab

def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    root=Path(__file__).resolve().parent;capture=root/'runs/layerwise_physical2400';out=root/'runs/readout50_trainonly'
    out.mkdir(exist_ok=False)
    payload=torch.load(root/'assets/best.pt',map_location='cpu',weights_only=True)
    assert lab.flow.sha(root/'assets/best.pt')==lab.SHA
    model=lab.OpticalRetrieval(payload['metadata']).cuda().eval().requires_grad_(False);model.load_state_dict(payload['state_dict'],strict=True)
    assert isinstance(model.readout.projection,torch.nn.Linear)
    protocol=json.loads((lab.OLD/'protocol.json').read_text());train=[r for r in protocol['rows'] if r['split']=='train'];query=[r for r in protocol['rows'] if r['split']=='query'];rows=train+query
    assert len(train)==1600 and len(query)==800
    assert not set(r['sample_id'] for r in train)&set(r['sample_id'] for r in query)
    # Source image identity audit, not just ID strings.
    def image_sha(r):return lab.flow.sha(lab.OLD/'data'/r['image_path'])
    tsha=[image_sha(r) for r in train];qsha=[image_sha(r) for r in query]
    assert not set(tsha)&set(qsha),'TRAIN/query image overlap'
    geometry=json.loads((lab.OLD/'runs/abo_i2i_20260926/shs_geometry.json').read_text())
    lab.flow.BASE_CORNERS=np.asarray(geometry['base_corners_screen_TL_TR_BR_BL'],np.float32)
    lab.pipeline.STAGE_CALIBRATION.clear();lab.pipeline.STAGE_CALIBRATION.update({s:(r['phase_candidate'],r['camera_orientation']) for s,r in geometry['stage_calibration'].items()})
    phases={s:capture/'phase'/(s+'.bmp') for s in lab.STAGES}
    def replay(bench,unused,stage,phase,active,ids,orientation):
        arrays=[]
        for sid in ids:
            folder=capture/'ccd'/stage;r=json.loads((folder/(sid+'.json')).read_text())
            assert r['phase_sha256']==lab.flow.sha(phases[stage]) and r['wait_ms']==240 and r['exposure']['exposure_us']==400
            arrays.append(np.array(Image.open(folder/(sid+'.png'))))
        return torch.from_numpy(np.stack(arrays)).cuda().float(),{},1.
    lab.pipeline.stage_active=lab.stage_active;lab.pipeline.capture_stage=replay
    processor=lab.AutoProcessor.from_pretrained(str(lab.OLD/'assets/processor'),local_files_only=True)
    inputs=[];vectors=[]
    hook=model.readout.projection.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().cpu().clone()))
    for i in range(0,len(rows),4):
        with torch.inference_mode():v=lab.pipeline.process_batch(model,processor,None,out,phases,rows[i:i+4])['descriptor']
        vectors.append(v.cpu())
        if i%160==0:print(json.dumps({'replayed':i+4,'total':2400}),flush=True)
    hook.remove();x=torch.cat(inputs).cuda();baseline=torch.cat(vectors)
    bank=torch.load(capture/'physical_features.pt',map_location='cpu',weights_only=True)
    assert bank['ids']==[r['sample_id'] for r in rows]
    err=float((baseline-bank['vectors']).abs().max());assert err<1e-5,err
    classes={p:i for i,p in enumerate(sorted({r['product_id'] for r in train}))};y=torch.tensor([classes[r['product_id']] for r in train],device='cuda');fit=[];val=[]
    for product in classes:
        ids=[i for i,r in enumerate(train) if r['product_id']==product];ids.sort(key=lambda i:hashlib.sha256(('physical-head42:'+train[i]['sample_id']).encode()).hexdigest());assert len(ids)==8;fit+=ids[:6];val+=ids[6:]
    assert not set(tsha[i] for i in fit)&set(tsha[i] for i in val),'Fit/VAL image overlap'
    fi=torch.tensor(fit,device='cuda');vi=torch.tensor(val,device='cuda');head=copy.deepcopy(model.readout.projection).requires_grad_(True)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=0)
    positive=y[fi,None].eq(y[fi][None]);diagonal=torch.eye(len(fit),device='cuda',dtype=torch.bool);positive&=~diagonal
    anchors=baseline[fit].cuda()
    def validate():
        z=F.normalize(head(x[:1600]),dim=-1);nearest=(z[vi]@z[fi].T).argmax(1);return float(y[fi][nearest].eq(y[vi]).float().mean())
    with torch.no_grad():best_score=validate()
    best=copy.deepcopy(head.state_dict());epoch_best=0;history=[{'epoch':0,'validation_r1':best_score}]
    for ep in range(1,51):
        opt.zero_grad();z=F.normalize(head(x[fi]),dim=-1);logits=(z@z.T/.1).masked_fill(diagonal,-1e4)
        loss=-(logits.log_softmax(1)*positive).sum(1).div(positive.sum(1)).mean()+.05*(1-(z*anchors).sum(1)).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
        with torch.no_grad():score=validate()
        row={'epoch':ep,'loss':float(loss.detach()),'validation_r1':score};history.append(row);print(json.dumps(row),flush=True)
        if score>best_score:best_score=score;best=copy.deepcopy(head.state_dict());epoch_best=ep
    protected={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in payload['state_dict'].items() if not k.startswith('readout.projection.')}
    def save(name,state):
        result=dict(payload,state_dict=dict(payload['state_dict']))
        for k,v in state.items():result['state_dict']['readout.projection.'+k]=v.cpu()
        assert all(hashlib.sha256(result['state_dict'][k].numpy().tobytes()).hexdigest()==h for k,h in protected.items())
        torch.save(result,out/name)
    save('last.pt',head.state_dict());save('best.pt',best);head.load_state_dict(best)
    with torch.no_grad():final=F.normalize(head(x),dim=-1).cpu()
    predictions,metrics=lab.pipeline.evaluate(final[1600:],query,protocol,{'ids':bank['ids'],'vectors':final})
    report={'status':'complete','source_sha256':lab.SHA,'best_sha256':lab.flow.sha(out/'best.pt'),'baseline_replay_max_error':err,'metrics':metrics,'predictions':predictions,'best_epoch':epoch_best,'validation_r1':best_score,'fit_ids':[train[i]['sample_id'] for i in fit],'validation_ids':[train[i]['sample_id'] for i in val],'test_ids':[r['sample_id'] for r in query],'test_used_for_training_or_selection':False,'train_count':1200,'validation_count':400,'test_count':800,'physical_train_captures_reused':True,'physical_gallery':1600,'protected_unchanged':True,'tuned_parameters':sum(p.numel() for p in head.parameters()),'history':history}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':'complete','metrics':metrics}),flush=True)
if __name__=='__main__':main()
