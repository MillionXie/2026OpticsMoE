"""TRAIN-only terminal decoder adaptation, VAL selection, single full TEST replay."""
import argparse,copy,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .export_samples import ssim
from ..sealed_editor import build_sealed
from ..product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from ..qwen_mini_small import PromptEmbeddingLookup
from .ccd_bridge import attach
from .physical_decoder_boundary import freeze_for_adaptation,protected_hash

STAGES=('language_router','language_expert','language_global','vision_router','vision_expert','vision_global')
def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(value,target):
    a=value.detach().cpu().float().add(1).mul(.5).clamp(0,1).permute(1,2,0).numpy()
    b=target.detach().cpu().float().add(1).mul(.5).clamp(0,1).permute(1,2,0).numpy()
    mse=float(np.mean((a-b)**2))
    return dict(mse_0_1=mse,mae_0_1=float(np.mean(abs(a-b))),psnr_db=float(-10*np.log10(max(mse,1e-15))),ssim=ssim(torch.from_numpy(a).permute(2,0,1),torch.from_numpy(b).permute(2,0,1)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--train',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(927)
    payload=torch.load(a.project/'assets/small.pt',map_location='cpu',weights_only=False)
    contract=json.loads((a.project/'assets/contract.json').read_text());assert sha(a.project/'assets/small.pt')==contract['checkpoint_sha256']
    model=build_sealed(payload);selected=freeze_for_adaptation(model);protected=protected_hash(model)
    selection=json.loads(a.selection.read_text());assert selection['test_product_overlap']==selection['test_source_hash_overlap']==0
    roles={role:[i for i,row in enumerate(selection['records']) if row['role']==role] for role in ('fit','validation')}
    assert len(roles['fit'])==1296 and len(roles['validation'])==432
    if a.preflight:
        print(json.dumps(dict(trainable_parameters=sum(v.numel() for _,v in selected),prefixes=sorted({n.split('.')[1] for n,_ in selected}),protected_sha256=protected,fit=len(roles['fit']),validation=len(roles['validation']))),flush=True);return
    # Waiting is CPU-only: never allocate another bench GPU during acquisition.
    deadline=time.monotonic()+10800
    while not (a.train/'report.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('TRAIN capture not complete')
        time.sleep(15)
    for root,count in ((a.train,1728),(a.test,2304)):
        report=json.loads((root/'report.json').read_text());assert report['status']=='complete' and report['sample_count']==count and report['contract']['checkpoint_sha256']==contract['checkpoint_sha256']
        for stage in STAGES:
            files=list((root/'ccd'/stage).glob('*.png'));assert len(files)==count,(stage,len(files))
            for file in files:
                receipt=json.loads(file.with_suffix('.json').read_text());assert receipt['exposure']['exposure_us']==400 and receipt['wait_ms']==240 and receipt['canonical_orientation']=='flip_v'
    assert not a.output.exists();a.output.mkdir(parents=True)
    write(a.output/'selection.json',selection)
    model=model.cuda();lookup=PromptEmbeddingLookup(a.project/'assets/datasets/abo_unified_expanded_qwen_embeddings_v2.pt')
    data=a.project/'assets/datasets';datasets={split:ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1',split,256,data/'abo_unified_expanded_instructions_qwen2_v2.pt') for split in ('train','test')}
    state={};restore=None
    def inputs(split,positions):
        indices=[selection['indices'][i] for i in positions] if split=='train' else positions
        rows=[datasets[split][i] for i in indices];emb,mask,_=lookup.batch([r['prompt'] for r in rows],torch.device('cpu'))
        reference=torch.stack([r['reference'] for r in rows]);noise=torch.stack([torch.randn(reference[0].shape,generator=torch.Generator().manual_seed(1042+i)) for i in indices])
        root=a.train if split=='train' else a.test;prefix='train' if split=='train' else 'test'
        state['ccd']={stage:torch.from_numpy(np.stack([np.asarray(Image.open(root/'ccd'/stage/f'{prefix}_{i:05d}.png')).copy() for i in positions])).cuda().float() for stage in STAGES}
        x=[reference.cuda(),emb.cuda().float(),mask.cuda(),noise.cuda()]
        return x,torch.stack([r['target'] for r in rows]).cuda(),rows
    restore=attach(model,lambda stage,amplitude,phase,ideal:state['ccd'][stage])
    # Numerical contract: replay unmodified TEST once on a fixed six-row audit,
    # not for selection. Must reproduce the previously sealed FP32 outputs.
    x,_,_=inputs('test',list(range(6)))
    with torch.no_grad():replayed=model(*x)
    recorded=torch.load(a.test/'batch_00000_00006/outputs.pt',map_location='cpu',weights_only=False)['actual'].cuda()
    replay_error=float((recorded-replayed).abs().max());assert replay_error<1e-5,replay_error
    write(a.output/'baseline_replay_audit.json',dict(max_error=replay_error,protected_sha256=protected,trainable_parameters=sum(v.numel() for _,v in selected)))
    @torch.no_grad()
    def evaluate(positions):
        model.eval();total=0.
        for start in range(0,len(positions),6):
            batch=positions[start:start+6];x,target,_=inputs('train',batch);total+=float(F.mse_loss(model(*x),target))*len(batch)
        return total/len(positions)
    baseline_val=evaluate(roles['validation']);best=baseline_val;best_epoch=0;history=[]
    def save(name,epoch):
        saved={k:v for k,v in payload.items() if k!='model'};saved['model']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        saved['physical_adaptation']=dict(epoch=epoch,source_sha256=contract['checkpoint_sha256'],protected_sha256=protected,scope=selection['scope'],decoder_only=True)
        torch.save(saved,a.output/name)
    save('best.pt',0);optimizer=torch.optim.AdamW([v for _,v in selected],lr=1e-5,weight_decay=.01)
    for epoch in range(1,a.epochs+1):
        # Keep all modules in eval mode; no dropout/RNG change to frozen upstream.
        model.eval();order=torch.randperm(len(roles['fit']),generator=torch.Generator().manual_seed(927+epoch)).tolist();total=0.
        for start in range(0,len(order),6):
            batch=[roles['fit'][i] for i in order[start:start+6]];x,target,_=inputs('train',batch);optimizer.zero_grad(set_to_none=True);prediction=model(*x);loss=F.mse_loss(prediction,target);loss.backward();torch.nn.utils.clip_grad_norm_([v for _,v in selected],1.);optimizer.step();total+=float(loss.detach())*len(batch)
        val=evaluate(roles['validation']);row=dict(epoch=epoch,fit_mse_minus1_1=total/len(order),validation_mse_minus1_1=val);history.append(row);print(json.dumps(row),flush=True);write(a.output/'history.json',history)
        assert protected_hash(model)==protected,'Frozen upstream changed'
        if val<best:best=val;best_epoch=epoch;save('best.pt',epoch)
        save('last.pt',epoch);write(a.output/'progress.json',dict(status='training',epoch=epoch,best_epoch=best_epoch))
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model']);assert protected_hash(model)==protected
    rows_out=[]
    for start in range(0,2304,6):
        ids=list(range(start,min(start+6,2304)));x,target,rows=inputs('test',ids)
        with torch.no_grad():physical=model(*x)
        restore()
        with torch.no_grad():simulation=model(*x)
        restore=attach(model,lambda stage,amplitude,phase,ideal:state['ccd'][stage])
        folder=a.output/f'batch_{start:05d}_{start+len(ids):05d}';folder.mkdir();torch.save(dict(actual=physical.cpu(),simulation=simulation.cpu()),folder/'outputs.pt')
        for j,index in enumerate(ids):
            row=dict(test_index=index,source_id=rows[j]['sample_id'],prompt=rows[j]['prompt'],category=rows[j]['category'],mode=rows[j]['mode'])
            for label,value in (('reference',x[0]),('target',target),('physical_tuned',physical),('simulation_tuned',simulation)):
                path=folder/f'test_{index:05d}_{label}.png';Image.fromarray(value[j].detach().cpu().add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).numpy()).save(path);row[label+'_image']=str(path.relative_to(a.output));row[label+'_sha256']=sha(path)
            for label,value in (('physical',physical),('simulation',simulation)):
                row.update({label+'_'+k:v for k,v in metrics(value[j],target[j]).items()})
            rows_out.append(row)
        write(a.output/'progress.json',dict(status='test_replay',completed=start+len(ids),total=2304))
    restore();write(a.output/'sample_metrics.json',rows_out)
    with (a.output/'sample_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows_out[0]));w.writeheader();w.writerows(rows_out)
    totals={key:sum(r[key] for r in rows_out)/2304 for key in rows_out[0] if key.startswith(('physical_','simulation_')) and isinstance(rows_out[0][key],float)}
    write(a.output/'report.json',dict(status='complete',best_epoch=best_epoch,baseline_validation_mse=baseline_val,best_validation_mse=best,protected_unchanged=True,checkpoint_sha256=sha(a.output/'best.pt'),test_samples=2304,metrics=totals,scope='VAL selected; TEST used only once after selection; same original physical CCD; continuation validation warm-start caveat'))
if __name__=='__main__':main()
