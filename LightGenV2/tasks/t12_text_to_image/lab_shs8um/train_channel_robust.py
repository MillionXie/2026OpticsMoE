"""TRAIN-only clean-anchored channel adaptation, VAL-only selection."""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from ..sealed_editor import build_sealed
from ..audited_unified import architecture_report
from ..product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from ..qwen_mini_small import PromptEmbeddingLookup
from .robust_channel import RobustChannel, PROFILES


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def ssim_per_image(a, b):
    x = torch.arange(11, device=a.device, dtype=a.dtype)-5
    g = torch.exp(-x.square()/(2*1.5**2)); g = g/g.sum()
    kernel = (g[:,None]*g[None,:]).expand(3,1,11,11)
    blur = lambda v: F.conv2d(v,kernel,groups=3)
    ma, mb = blur(a), blur(b)
    va, vb = blur(a*a)-ma*ma, blur(b*b)-mb*mb
    cov = blur(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean((1,2,3))


class IndexedDataset:
    def __init__(self, dataset): self.dataset=dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return dict(self.dataset[index], index=index)


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint','assets','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=('camera','combined'),default='camera')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--val-samples',type=int,default=96)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--seed',type=int,default=1042)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=build_sealed(saved).cuda().eval()
    report=architecture_report(model)
    if report['counted_parameters']>20_000_000: raise ValueError('20M budget exceeded')
    channel=RobustChannel(model)
    data=a.assets/'datasets'
    cache=data/'abo_unified_expanded_instructions_qwen2_v2.pt'
    lookup=PromptEmbeddingLookup(data/'abo_unified_expanded_qwen_embeddings_v2.pt')
    def dataset(split):
        return IndexedDataset(ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1',split,256,cache))
    valset=dataset('val')
    indices=torch.linspace(0,len(valset)-1,min(a.val_samples,len(valset))).long().tolist()
    val=DataLoader(Subset(valset,indices),batch_size=a.batch_size)
    execution=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   command=sys.argv,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                   source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                   data_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (cache,data/'abo_cleanrender_lamp_table_pillow_256_v1/val.jsonl',data/'abo_cleanrender_lamp_table_pillow_256_v1/train.jsonl')})
    write(a.output/'protocol.json',dict(execution=execution,config=vars(a)|{'checkpoint':str(a.checkpoint),'assets':str(a.assets),'output':str(a.output)},
          counted_parameters=report['counted_parameters'],profiles=PROFILES,
          amplitude_contract=saved['bounded_amplitude'],validation_indices=indices,
          selection='minimize mean camera/combined VAL MSE; clean VAL mean per-image PSNR drop <=0.2dB, SSIM drop <=0.002; TEST only after VAL selection',
          leakage='nominal coherent branch POWER fraction; sqrt(1-eta)*E_mod + sqrt(eta)*exp(i phi)*E_unmod; same bounded incident field; no image constants',
          detector='bounded-field intensity proxy, not calibrated camera electrons; no TEST CCD fitting'))

    def inputs(batch):
        ref=batch['reference'].cuda();target=batch['target'].cuda()
        emb,mask,_=lookup.batch(list(batch['prompt']),torch.device('cuda'))
        # Identical per-index seed to deployment full TEST, independent of batch size.
        noise=torch.stack([torch.randn(ref[0].shape,generator=torch.Generator().manual_seed(1042+int(i))) for i in batch['index']]).cuda()
        return ref,emb.float(),mask,noise,target

    @torch.no_grad()
    def evaluate(loader, profiles):
        model.eval(); results={}
        for profile in profiles:
            sums=dict(mse_0_1=0.,psnr_db=0.,ssim=0.); count=0; rows=[]
            for batch_index,batch in enumerate(loader):
                ref,emb,mask,noise,target=inputs(batch)
                # Fixed perturbation seed per batch for reproducible VAL ranking.
                torch.manual_seed(9000+int(batch['index'][0]))
                channel.configure(None if profile=='clean' else profile)
                output=model(ref,emb,mask,noise)
                pred=output.float().add(1).div(2).clamp(0,1)
                gt=target.float().add(1).div(2).clamp(0,1)
                mse=(pred-gt).square().mean((1,2,3));psnr=-10*torch.log10(mse.clamp_min(1e-15))
                ss=ssim_per_image(pred,gt)
                for j in range(len(ref)):
                    row=dict(index=int(batch['index'][j]),sample_id=batch['sample_id'][j],mode=batch['mode'][j],category=batch['category'][j],mse_0_1=float(mse[j]),psnr_db=float(psnr[j]),ssim=float(ss[j]))
                    rows.append(row)
                    for k in sums:sums[k]+=row[k]
                count+=len(ref)
                if a.evaluate_only and batch_index%200==0:
                    print(json.dumps(dict(profile=profile,completed=count)),flush=True)
            results[profile]=dict(samples=count,**{k:v/count for k,v in sums.items()})
            if a.evaluate_only: write(a.output/(profile+'_per_image.json'),rows)
        return results

    if a.evaluate_only:
        loader=DataLoader(dataset(a.split),batch_size=a.batch_size)
        write(a.output/'evaluation.json',dict(split=a.split,execution=execution,metrics=evaluate(loader,('clean','camera','combined','stress')),
              caveat='perturbed simulations are NOT new physical CCD measurements'))
        channel.restore();return

    teacher=build_sealed(saved).cuda().eval().requires_grad_(False)
    baseline=evaluate(val,('clean','camera','combined'))
    best_score=(baseline['camera']['mse_0_1']+baseline['combined']['mse_0_1'])/2
    baseline_clean=baseline['clean']; selected=0; history=[]
    def save(name,step):
        payload=copy.copy(saved)
        payload['model']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        payload['channel_robust_training']=dict(step=step,profile=a.profile,execution=execution)
        torch.save(payload,a.output/name)
    save('best_checkpoint.pt',0)
    write(a.output/'baseline_val.json',baseline)
    groups=[dict(params=[v for n,v in model.named_parameters() if 'raw_phase' in n or 'raw_router_phase' in n],lr=2e-5),
            dict(params=[v for n,v in model.named_parameters() if 'raw_phase' not in n and 'raw_router_phase' not in n],lr=2e-6)]
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    loader=DataLoader(dataset('train'),batch_size=a.batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(a.seed))
    step=0
    while step<a.steps:
        for batch in loader:
            model.train();optimizer.zero_grad(set_to_none=True)
            ref,emb,mask,noise,target=inputs(batch)
            with torch.no_grad(): anchor=teacher(ref,emb,mask,noise)
            channel.configure(None);clean=model(ref,emb,mask,noise)
            strength=.2+.8*min(1.,step/max(1,a.steps*.5))
            channel.configure(a.profile,strength);noisy=model(ref,emb,mask,noise)
            quality=lambda pred:F.mse_loss(pred,target)+.1*F.l1_loss(pred,target)
            loss=.5*quality(clean)+.5*quality(noisy)+.1*F.mse_loss(clean,anchor)+.05*F.mse_loss(noisy,anchor)
            if not bool(loss.isfinite()): raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
            if step%25==0:print(json.dumps(dict(step=step,loss=float(loss),strength=strength)),flush=True)
            if step%100==0 or step==a.steps:
                metrics=evaluate(val,('clean','camera','combined'))
                clean_val=metrics['clean']
                eligible=(clean_val['psnr_db']>=baseline_clean['psnr_db']-.2 and clean_val['ssim']>=baseline_clean['ssim']-.002)
                score=(metrics['camera']['mse_0_1']+metrics['combined']['mse_0_1'])/2
                record=dict(step=step,metrics=metrics,eligible=eligible,score=score)
                if eligible and score<best_score:best_score=score;selected=step;save('best_checkpoint.pt',step)
                history.append(record);write(a.output/'history.json',history);print(json.dumps(record),flush=True)
            if step>=a.steps:break
    save('last_checkpoint.pt',step)
    write(a.output/'report.json',dict(status='complete',selected_step=selected,baseline_val=baseline,history=history,
                                    counted_parameters=report['counted_parameters'],execution=execution))
    channel.restore()


if __name__=='__main__':main()
