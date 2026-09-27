"""Official pix2pix-Turbo generator, paired ABO fine-tuning; no TEST selection.

External upstream is pinned by git SHA, never silently vendored or modified.
Local pretrained path redirection changes loading only. Generator/LoRA/skips and
single-step scheduler are upstream. This task wrapper adds our split, PSNR
selection and best/last policy; combined G update differs from upstream's two.
"""
import argparse, csv, hashlib, importlib.metadata, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from .export_matched_summary import sha, per_image_metrics, save_png


def load_generator(upstream, base, checkpoint=None):
    from transformers import AutoTokenizer, CLIPTextModel
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    sys.path.insert(0, str(upstream/'src'))
    classes=(AutoTokenizer,CLIPTextModel,AutoencoderKL,UNet2DConditionModel,DDPMScheduler)
    originals={cls:cls.from_pretrained for cls in classes}
    def redirect(cls):
        original=originals[cls]
        def loading(path,*pos,**kw):
            if path=='stabilityai/sd-turbo':
                path=str(base);kw['local_files_only']=True
                if cls in (AutoencoderKL,UNet2DConditionModel,CLIPTextModel):kw['variant']='fp16'
            return original(path,*pos,**kw)
        return loading
    for cls in classes:cls.from_pretrained=staticmethod(redirect(cls))
    try:
        from pix2pix_turbo import Pix2Pix_Turbo
        model=Pix2Pix_Turbo(pretrained_path=str(checkpoint) if checkpoint else None,lora_rank_unet=8,lora_rank_vae=4)
    finally:
        for cls in classes:cls.from_pretrained=originals[cls]
    # Upstream set_train does not freeze untouched tensors; official optimizer
    # filters them. Explicit freezing here gives exactly that optimizer subset.
    model.unet.requires_grad_(False);model.vae.requires_grad_(False)
    model.set_train()
    model.unet.enable_gradient_checkpointing()
    return model


def architecture(model):
    parts={name:sum(p.numel() for p in getattr(model,name).parameters()) for name in ('text_encoder','unet','vae')}
    total=sum(p.numel() for p in model.parameters())
    embedding=model.text_encoder.get_input_embeddings().weight.numel()
    return {'components_total':parts,'total_inference_parameters':total,
            'word_embedding_parameters_separately_listed':embedding,
            'counted_inference_parameters_excluding_word_embedding':total-embedding,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'trainable_by_component':{n:sum(p.numel() for p in getattr(model,n).parameters() if p.requires_grad) for n in parts},
            'lora_ranks':{'unet':8,'vae':4},'unet_calls':1,'image_size':[256,256],
            'qwen_used':False,'pca_condition_used':False,'gt_mask_or_retrieval_at_inference':False,
            'training_only_networks_excluded':'VGG LPIPS, CLIP image/text similarity, vision-aided discriminator'}


def main():
    parser=argparse.ArgumentParser()
    for name in ('assets','upstream','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--learning-rate',type=float,default=5e-6)
    parser.add_argument('--validation-every',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=927)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--evaluate',choices=('val','test'))
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.evaluate and (args.output/'protocol.json').exists():raise ValueError('Use a new output; do not overwrite a sealed run')
    upstream_sha=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if upstream_sha!='86f54146590ffb4543c8cf85b5a36657da670924':raise ValueError('Unexpected upstream revision')
    if subprocess.check_output(['git','-C',str(args.upstream),'status','--porcelain'],text=True).strip():raise ValueError('Upstream must be clean')
    assets=args.assets;model=load_generator(args.upstream,assets/'models/sd-turbo-fp16',args.checkpoint)
    data=assets/'datasets/abo_cleanrender_lamp_table_pillow_256_v1'
    instructions=assets/'datasets/abo_unified_expanded_instructions_qwen2_v2.pt'
    protocol={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'upstream_git_commit':upstream_sha,'upstream_repository':'https://github.com/GaParmar/img2img-turbo',
              'argv':sys.argv,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
              'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'architecture':architecture(model),'pretraining':'Local SD-Turbo pretrained backbone; fresh LoRA/skips, not from-scratch backbone training',
              'data_sha256':{s:sha(data/(s+'.jsonl')) for s in ('train','val','test')},'instructions_sha256':sha(instructions),
              'base_model_files':{str(p.relative_to(assets/'models/sd-turbo-fp16')):sha(p) for p in sorted((assets/'models/sd-turbo-fp16').rglob('*')) if p.is_file()},
              'metric_protocol':'RGB[0,1], per-image PSNR then mean; full VAL selection, fixed TEST only after selection',
              'loss':'1 RGB MSE + 5 VGG LPIPS + 5 CLIP text similarity + .5 vision-aided CLIP GAN',
              'wrapper_difference':'Single combined G backward/update instead of upstream separate reconstruction/GAN updates; effective batch8; full VAL PSNR; best/last only',
              'precision':'FP32 weights with BF16 autocast'}
    def write(name,value):(args.output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    write('protocol.json',protocol);write('architecture.json',protocol['architecture'])
    def dataset(split):return ExpandedUnifiedProductEditDataset(data,split,256,instructions)
    def forward(x,prompts):
        with torch.autocast('cuda',dtype=torch.bfloat16):return model(x,prompt=prompts,deterministic=True)
    @torch.no_grad()
    def evaluate(split,export=False):
        model.unet.eval();model.vae.eval();rows=[];ds=dataset(split)
        for i in range(12 if args.smoke else len(ds)):
            row=ds[i]
            # Official deterministic=True still samples VAE posterior. Preserve
            # official forward, fix posterior randomness per sample, not batch.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(1042+i);pred=forward(row['reference'][None].cuda(),[row['prompt']]).float()
            metrics=per_image_metrics(pred,row['target'][None].cuda())[0]
            sample_id=f'{split}_{i:05d}'
            record={'test_index':i,'sample_id':sample_id,'source_id':row['sample_id'],
                    'product_id':row['source_id'],'prompt':row['prompt'],**metrics}
            for key in ('mode','category','target_id','source_scene','target_scene'):
                record[key]=row.get(key,'unknown')
            if export:
                for name,value in (('reference',row['reference']),('target',row['target']),('generated',pred[0])):
                    relative=f'images/{name}/{sample_id}.png'
                    record[name+'_image']=relative
                    record[name+'_png_sha256']=save_png(value,args.output/relative)
            rows.append(record)
        model.unet.train();model.vae.train()
        summary={'count':len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in ('psnr_db','ssim','mse_0_1','mae_0_1')}}
        return summary,rows
    if args.evaluate:
        if args.checkpoint is None:raise ValueError('Evaluation requires selected checkpoint')
        metrics,rows=evaluate(args.evaluate,True);write('sample_metrics.json',rows)
        with (args.output/'sample_metrics.csv').open('w',encoding='utf-8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write('report.json',{'status':'complete','split':args.evaluate,'checkpoint_sha256':sha(args.checkpoint),'metrics':metrics,'protocol':protocol});return
    import lpips,vision_aided_loss
    sys.path.insert(0,str(assets/'vendor/CLIP'))
    import clip
    perceptual=lpips.LPIPS(net='vgg').cuda().eval().requires_grad_(False)
    clip_model,_=clip.load('ViT-B/32',device='cuda');clip_model=clip_model.float().eval().requires_grad_(False)
    disc=vision_aided_loss.Discriminator(cv_type='clip',loss_type='multilevel_sigmoid_s',device='cuda').cuda()
    disc.requires_grad_(True);disc.cv_ensemble.requires_grad_(False)
    gp=[p for p in model.parameters() if p.requires_grad];dp=[p for p in disc.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(gp,lr=args.learning_rate,weight_decay=.01)
    dop=torch.optim.AdamW(dp,lr=args.learning_rate,weight_decay=.01)
    train=dataset('train');assert len(train)==20736
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed))
    history=[];step=0;best=-float('inf');best_step=0;started=time.monotonic()
    def check(epoch):
        nonlocal best,best_step
        val,_=evaluate('val');entry={'epoch':epoch,'step':step,'val':val,'elapsed_seconds':time.monotonic()-started};history.append(entry)
        if val['psnr_db']>best:
            best=val['psnr_db'];best_step=step;model.save_model(args.output/'best_checkpoint.pt')
        model.save_model(args.output/'last_checkpoint.pt')
        write('history.json',history);write('progress.json',{'status':'training','step':step,'best_step':best_step,'best_val_psnr':best,'last_val':val})
        print(json.dumps(entry),flush=True)
    check(0);opt.zero_grad();dop.zero_grad()
    for epoch in range(1,args.epochs+1):
        for bi,batch in enumerate(loader):
            x=batch['reference'].cuda();gt=batch['target'].cuda();pred=forward(x,batch['prompt']).float()
            disc.requires_grad_(False)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                resized=F.interpolate(pred*.5+.5,(224,224),mode='bilinear',align_corners=False)
                mean=pred.new_tensor((.48145466,.4578275,.40821073))[None,:,None,None]
                std=pred.new_tensor((.26862954,.26130258,.27577711))[None,:,None,None]
                logits,_=clip_model((resized-mean)/std,clip.tokenize(batch['prompt'],truncate=True).cuda())
                gloss=F.mse_loss(pred,gt)+5*perceptual(pred,gt).mean()+5*(1-logits.mean()/100)+.5*disc(pred,for_G=True).mean()
            if not torch.isfinite(gloss):raise ValueError('Nonfinite generator loss')
            (gloss/args.accumulation).backward()
            disc.requires_grad_(True);disc.cv_ensemble.requires_grad_(False)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                dloss=.5*(disc(gt,for_real=True).mean()+disc(pred.detach(),for_real=False).mean())
            if not torch.isfinite(dloss):raise ValueError('Nonfinite discriminator loss')
            (dloss/args.accumulation).backward();step+=1
            if (bi+1)%args.accumulation==0 or bi+1==len(loader):
                torch.nn.utils.clip_grad_norm_(gp,1);torch.nn.utils.clip_grad_norm_(dp,1)
                opt.step();dop.step();opt.zero_grad(set_to_none=True);dop.zero_grad(set_to_none=True)
            if step%100==0:
                write('progress.json',{'status':'training','epoch':epoch,'step':step,'best_step':best_step,'best_val_psnr':best,'g_loss':float(gloss),'d_loss':float(dloss)})
                print('train',epoch,step,float(gloss),float(dloss),flush=True)
            if step%args.validation_every==0:check(epoch)
            if args.smoke and step>=4:break
        check(epoch)
        if args.smoke:break
    write('report.json',{'status':'smoke_complete' if args.smoke else 'complete','best_step':best_step,'best_val_psnr':best,
                         'best_checkpoint_sha256':sha(args.output/'best_checkpoint.pt'),'protocol':protocol,'test':'not used; separate selected-checkpoint evaluation pending'})
    print('complete',best_step,best,flush=True)


if __name__=='__main__':main()
