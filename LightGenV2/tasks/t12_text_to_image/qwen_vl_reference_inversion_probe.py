"""Controlled TRAIN reference-reconstruction probe of three Qwen feature taps.

This is NOT an image editor or a deployment baseline. It tests whether a small
readout can recover reference appearance from frozen Qwen features. Resolution
and feature stage both differ; do not claim a single-factor causal attribution.
"""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw

from .qwen_vl_hybrid_run import checked_features,load_data
from .qwen_vl_turbo_cache import image_float,save_with_digest
from .qwen_vl_turbo_data import sha256,write_json
from .qwen_vl_turbo_run import tensor_image


def native_vision_grid(tokens):
    if tokens.ndim!=3 or tokens.shape[1]!=256:raise ValueError('Native 16x16 grid required')
    # Original Qwen unmerged order is tile-row/tile-col/local-row/local-col.
    b,_,c=tokens.shape
    return tokens.reshape(b,8,8,2,2,c).permute(0,1,3,2,4,5).reshape(b,16,16,c)


class InverseReadout(nn.Module):
    def __init__(self,dimension,grid):
        super().__init__();self.grid=grid
        if grid not in (8,16):raise ValueError('Only audited native grids')
        self.norm=nn.LayerNorm(dimension);self.project=nn.Linear(dimension,128)
        self.up1=nn.Conv2d(128,64,3,padding=1)
        self.up2=nn.Conv2d(64,32,3,padding=1) if grid==8 else None
        self.output=nn.Conv2d(32 if grid==8 else 64,4,1)

    def forward(self,tokens):
        x=self.project(self.norm(tokens.float()))
        x=native_vision_grid(x) if self.grid==16 else x.reshape(len(x),8,8,128)
        x=x.permute(0,3,1,2)
        x=F.silu(self.up1(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
        if self.up2 is not None:x=F.silu(self.up2(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
        return self.output(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','features','merger','sd','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--steps',type=int,default=600);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid or a.output.exists() or a.steps<4:raise ValueError('Approved GPU, new run and real training required')
    payload=load_data(a.data);features=checked_features(a.features,payload)
    ready=json.loads((a.merger/'report.json').read_text());path=a.merger/'native_vision.pt'
    if not ready['complete'] or ready['sha256']!=sha256(path) or ready['source_cache_sha256']!=json.loads((a.data/'cache_complete.json').read_text())['sha256'] or ready['terminal_feature_sha256']!=json.loads((a.features/'report.json').read_text())['sha256']:
        raise ValueError('Native merger provenance differs')
    merger=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if merger['sample_ids']!=[r['sample_id'] for r in payload['rows']]:raise ValueError('Native merger sample order differs')
    selected=[];seen=set()
    for i,row in enumerate(payload['rows']):
        digest=hashlib.sha256(tensor_image(payload['reference'][i]).tobytes()).hexdigest()
        key=row['split'],digest
        if key not in seen:seen.add(key);selected.append(i)
    train=[j for j,i in enumerate(selected) if payload['rows'][i]['split']=='train'];dev=[j for j,i in enumerate(selected) if payload['rows'][i]['split']=='val']
    if len(train)!=76 or len(dev)!=16:raise ValueError('Unexpected unique original-reference cohort')
    torch.manual_seed(1042);rng=random.Random(1042);torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.8,0)
    a.output.mkdir(parents=True);write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),'started_unix':time.time(),'TRAIN_gradients_only':True})
    inputs={'LM_image64':torch.stack([payload['hidden'][i][4:68] for i in selected]).to(device),
            'merger64':torch.stack([merger['vision'][i] for i in selected]).to(device),
            'vision23_256':torch.cat([features[i]['vision_23'] for i in selected]).to(device)}
    models=nn.ModuleDict({name:InverseReadout(value.shape[-1],16 if name=='vision23_256' else 8).to(device) for name,value in inputs.items()})
    optimizer=torch.optim.AdamW(models.parameters(),lr=1e-3,weight_decay=.001)
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained(a.sd,subfolder='vae',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval().requires_grad_(False)
    reference=image_float(payload['reference'][selected],device)
    with torch.no_grad():goals=torch.cat([vae.encode(reference[j:j+16].to(vae.dtype)).latent_dist.mode().float()*vae.config.scaling_factor for j in range(0,len(reference),16)])
    import lpips
    metric=lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False);history=[]
    @torch.no_grad()
    def evaluate(step):
        result={'step':step,'branches':{}}
        figure_ids={s:sum(([j for j in ids if payload['rows'][selected[j]]['category']==c][:3] for c in ('lamp','pillow')),[]) for s,ids in [('TRAIN',train),('DEV',dev)]}
        panels={s:Image.new('RGB',(1280,278*len(ids))) for s,ids in figure_ids.items()}
        for name,model in models.items():
            model.eval();rows=[]
            for start in range(0,len(reference),8):
                with torch.autocast('cuda',dtype=torch.bfloat16):z=model(inputs[name][start:start+8]);rgb=vae.decode(z.to(vae.dtype)/vae.config.scaling_factor).sample.float().clamp(-1,1)
                target=reference[start:start+8];mse=(rgb-target).square().mean((1,2,3))/4;perceptual=metric(rgb,target).flatten()
                for k in range(len(rgb)):
                    j=start+k;split='TRAIN' if j in train else 'DEV';ids=train if split=='TRAIN' else dev
                    rows.append({'split':split,'sample_id':payload['rows'][selected[j]]['sample_id'],'PSNR':float(-10*mse[k].clamp_min(1e-12).log10()),'LPIPS':float(perceptual[k]),'latent_MSE':float((z[k].float()-goals[j]).square().mean())})
                    if j in figure_ids[split]:
                        pos=figure_ids[split].index(j);y=pos*278;draw=ImageDraw.Draw(panels[split]);col=list(models).index(name)+2
                        draw.text((0,y),'REFERENCE RECONSTRUCTION ONLY '+split,fill='white');draw.text((col*256,y+12),name,fill='white')
                        panels[split].paste(tensor_image(target[k]),(0,y+22));panels[split].paste(tensor_image(rgb[k]),(col*256,y+22))
                        if name=='LM_image64':
                            oracle=vae.decode(goals[j:j+1].to(vae.dtype)/vae.config.scaling_factor).sample[0]
                            draw.text((256,y+12),'VAE ORACLE (not editor)',fill='white');panels[split].paste(tensor_image(oracle),(256,y+22))
            result['branches'][name]={s:{'n':sum(v['split']==s for v in rows),**{key:sum(v[key] for v in rows if v['split']==s)/sum(v['split']==s for v in rows) for key in ('PSNR','LPIPS','latent_MSE')}} for s in ('TRAIN','DEV')}
            result['branches'][name]['images']=rows
        for split,panel in panels.items():panel.save(a.output/(split+'_reference_reconstruction_'+str(step)+'.png'))
        return result
    for step in range(1,a.steps+1):
        batch=rng.sample(train,16);models.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=1e-4+9e-4*.5*(1+math.cos(math.pi*step/a.steps))
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=sum(F.mse_loss(model(inputs[name][batch]).float(),goals[batch]) for name,model in models.items())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reference inversion loss')
        loss.backward();norm=nn.utils.clip_grad_norm_(models.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite reconstruction gradients')
        optimizer.step()
        if step%25==0:write_json(a.output/'progress.json',{'step':step,'steps':a.steps,'TRAIN_latent_loss':float(loss.detach())})
        if step in {100,300,a.steps}:history.append(evaluate(step));write_json(a.output/'history.json',history)
    state_sha=save_with_digest({'readouts':models.state_dict(),'steps':a.steps,'sample_ids':[payload['rows'][i]['sample_id'] for i in selected]},a.output/'diagnostic_readouts.pt')
    write_json(a.output/'report.json',{'complete':True,'reference_reconstruction_only':True,'generative_editor':False,'model_selection_performed':False,
        'TRAIN_unique_references':len(train),'DEV_unique_references':len(dev),'independent_TEST_opened':False,'TRAIN_only_gradients':True,
        'reference_VAE_encoder_role':'TRAIN reference-latent targets and explicitly labelled reconstruction oracle, never a deployed editor input',
        'steps':a.steps,'final':history[-1],'parameters':{n:sum(v.numel() for v in m.parameters()) for n,m in models.items()},'checkpoint_sha256':state_sha,
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),
        'limits':'Feature stage and grid differ; readout parameter counts differ. CPU merger replay had previously measured small BF16 differences. This diagnostic neither establishes text-editing success nor proposes a reference-VAE deployment bypass.'})


if __name__=='__main__':main()
