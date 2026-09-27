"""Fixed VAL optical-channel visualizations and aggregate live routing audit."""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
import torch
import numpy as np
from PIL import Image,ImageDraw
from torch.utils.data import DataLoader,Subset
from ..sealed_editor import build_sealed
from ..qwen_mini_small import PromptEmbeddingLookup
from ..product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from ..audited_unified import optical_diagnostics,architecture_report
from .robust_channel import RobustChannel
from .train_channel_robust import IndexedDataset,ssim_per_image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=96)
    p.add_argument('--profiles',nargs='+',default=['clean','combined','stress','severe','extreme'])
    a=p.parse_args()
    if len(a.labels)!=len(a.checkpoints):raise ValueError('Labels/checkpoints mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    data=a.assets/'datasets'
    dataset=IndexedDataset(ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1','val',256,data/'abo_unified_expanded_instructions_qwen2_v2.pt'))
    # First item for each category/mode, no metric-driven cherry picking.
    examples={}
    for i in range(len(dataset)):
        item=dataset[i];key=(item['category'],item['mode'])
        if key not in examples:examples[key]=i
        if len(examples)==9:break
    preview=list(examples.values())
    indices=sorted(set(torch.linspace(0,len(dataset)-1,min(a.samples,len(dataset))).long().tolist()+preview))
    lookup=PromptEmbeddingLookup(data/'abo_unified_expanded_qwen_embeddings_v2.pt')
    saved_images={};reference={};targets={};results={}
    def pil(x):return Image.fromarray((x.detach().cpu().add(1).div(2).clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8))
    for label,checkpoint in zip(a.labels,a.checkpoints):
        model=build_sealed(torch.load(checkpoint,map_location='cpu',weights_only=False)).cuda().eval()
        channel=RobustChannel(model)
        stats={};clean_masks={}
        for profile in a.profiles:
            rows=[];routing={b:dict(selected=[],weights=[],probabilities=[],changed=0) for b in ('language','vision')}
            for batch in DataLoader(Subset(dataset,indices),batch_size=4):
                ref=batch['reference'].cuda();gt=batch['target'].cuda()
                emb,mask,_=lookup.batch(list(batch['prompt']),torch.device('cuda'))
                noise=torch.stack([torch.randn(ref[0].shape,generator=torch.Generator().manual_seed(1042+int(i))) for i in batch['index']]).cuda()
                torch.manual_seed(9000+int(batch['index'][0]));channel.configure(None if profile=='clean' else profile)
                with torch.no_grad():
                    out=model(ref,emb.float(),mask,noise)
                    x=out.add(1).div(2).clamp(0,1);y=gt.add(1).div(2).clamp(0,1)
                    mse=(x-y).square().mean((1,2,3));ss=ssim_per_image(x,y)
                for j,index in enumerate(batch['index'].tolist()):
                    rows.append(dict(index=index,mse=float(mse[j]),psnr=float(-10*torch.log10(mse[j].clamp_min(1e-15))),ssim=float(ss[j])))
                    if index in preview:
                        image=pil(out[j]);saved_images[label,profile,index]=image
                        image.save(a.output/f'{label}_{profile}_{index}.png')
                        reference[index]=pil(ref[j]);targets[index]=pil(gt[j])
                for branch,obj in (('language',model.text),('vision',model.editor.bottleneck)):
                    r=obj.last_routing
                    selected=r['selected_mask'].detach().cpu().bool()
                    for key in ('weights','probabilities'):routing[branch][key].extend(r[key].detach().cpu().tolist())
                    routing[branch]['selected'].extend(selected.tolist())
                    for j,index in enumerate(batch['index'].tolist()):
                        if profile=='clean':clean_masks[branch,index]=selected[j]
                        else:routing[branch]['changed']+=int(not torch.equal(clean_masks[branch,index],selected[j]))
            for branch,r in routing.items():
                selected=torch.tensor(r.pop('selected')).float()
                r['selection_rate']=selected.mean(0).tolist()
                r['weight_mean']=torch.tensor(r.pop('weights')).mean(0).tolist()
                r['probability_mean']=torch.tensor(r.pop('probabilities')).mean(0).tolist()
                r['top2_changed_fraction']=r.pop('changed')/len(rows)
                r['top2_pair_counts']={str(pair):int(count) for pair,count in zip(*torch.unique(selected,dim=0,return_counts=True))}
            stats[profile]=dict(samples=len(rows),psnr=sum(r['psnr'] for r in rows)/len(rows),ssim=sum(r['ssim'] for r in rows)/len(rows),mse=sum(r['mse'] for r in rows)/len(rows),routing=routing)
            print(json.dumps(dict(label=label,profile=profile,metrics=stats[profile])),flush=True)
        results[label]=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),architecture=architecture_report(model),alpha=optical_diagnostics(model),metrics=stats)
        channel.restore();del model,channel;torch.cuda.empty_cache()
    # Three category exemplars, full resolution; all nine also exported individually.
    for mode in ('background','object','joint'):
        selected=[(category,index) for (category,item_mode),index in examples.items() if item_mode==mode]
        for label in a.labels:
            cols=['input','GT','clean','stress','severe','extreme']
            sheet=Image.new('RGB',(256*len(cols),280*len(selected)),(255,255,255));draw=ImageDraw.Draw(sheet)
            for row,(category,index) in enumerate(selected):
                for col,name in enumerate(cols):
                    img=reference[index] if name=='input' else targets[index] if name=='GT' else saved_images[label,name,index]
                    sheet.paste(img,(col*256,row*280+24));draw.text((col*256+4,row*280+4),f'{category} {name}',fill='black')
            sheet.save(a.output/f'{label}_{mode}_contact.png')
    (a.output/'audit.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),split='VAL',indices=indices,preview_indices=preview,selection='first category/mode; no cherry-picking',models=results),indent=2))


if __name__=='__main__':main()
