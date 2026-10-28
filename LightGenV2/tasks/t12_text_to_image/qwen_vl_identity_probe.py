"""TRAIN-only identity-fit diagnostic using the complete existing raw editor.

Unlike a tiny inverse readout, this preserves the actual Qwen/UNet/VAE graph.
It is not editing performance, an independent test, or an accepted teacher.
"""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw

from .qwen_vl_hybrid_run import completed_teacher,load_data,load_qwen
from .qwen_vl_joint_adapt import RawQwenHead,install_joint_lora,joint_adapter_state,load_joint_adapters
from .qwen_vl_spatial import build_spatial_head
from .qwen_vl_turbo_cache import image_float,save_with_digest
from .qwen_vl_turbo_data import sha256,write_json
from .qwen_vl_turbo_run import tensor_image


def select_identity_references(rows,fit_per_category=6,transfer_per_category=3):
    fit=[];transfer=[]
    for category in ('lamp','pillow'):
        selected=[];seen=set()
        for i,row in enumerate(rows):
            if row['split']!='train' or row['category']!=category:continue
            product=row['source']['product_id']
            if product not in seen:seen.add(product);selected.append(i)
            if len(selected)==fit_per_category+transfer_per_category:break
        if len(selected)!=fit_per_category+transfer_per_category:raise ValueError('Insufficient original TRAIN products')
        fit+=selected[:fit_per_category];transfer+=selected[fit_per_category:]
    return fit,transfer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','parent','data','qwen','sd','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid'] or a.output.exists():raise ValueError('Authorized single GPU and new diagnostic run required')
    parent=completed_teacher(a.parent);state=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if state.get('schema')!='qwen_spatial_joint_lora_v1':raise ValueError('Use the verified complete raw multimodal editor')
    payload=load_data(a.data);fit,transfer=select_identity_references(payload['rows'],cfg['fit_per_category'],cfg['transfer_per_category'])
    ids=fit+transfer;nfit=len(fit)
    if any(payload['rows'][i]['split']!='train' for i in ids):raise ValueError('Neither DEV nor TEST may enter this diagnostic')
    inference={'reference':payload['reference'][ids],'rows':[{'prompt':cfg['identity_prompt']} for _ in ids]}
    torch.manual_seed(cfg['seed']);rng=random.Random(cfg['seed']);torch.set_num_threads(4);device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],0);a.output.mkdir(parents=True)
    write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256':sha256(Path(__file__)),'started_unix':time.time(),'config':cfg,'parent_checkpoint_sha256':parent['checkpoint_sha256'],
        'data_cache_sha256':json.loads((a.data/'cache_complete.json').read_text())['sha256'],
        'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0)})
    write_json(a.output/'selected_references.json',{'original_cohort':'TRAIN','identity_prompt':cfg['identity_prompt'],
        'items':[{'sample_id':payload['rows'][i]['sample_id'],'product_id':payload['rows'][i]['source']['product_id'],
            'category':payload['rows'][i]['category'],'group':'fit' if j<nfit else 'transfer_within_original_TRAIN',
            'reference_RGB_sha256':hashlib.sha256(tensor_image(inference['reference'][j]).tobytes()).hexdigest()} for j,i in enumerate(ids)]})
    qwen,processor=load_qwen(a.qwen,device);install_joint_lora(qwen,state['config'],state['qwen_joint_lora'],train=True)
    qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    head=build_spatial_head(a.sd,state,state['config'],state['visual_layout']['positions'],device)
    head.unet.enable_gradient_checkpointing();model=RawQwenHead(head,qwen,processor,state['visual_layout'])
    references=image_float(inference['reference'],device)
    # VAE encoder produces TRAIN loss targets only; forward below takes RGB
    # through Qwen and never receives an encoded reference or a target latent.
    with torch.no_grad():goal=torch.cat([model.vae.encode(references[j:j+6].to(model.vae.dtype)).latent_dist.mode().float()*model.vae.config.scaling_factor for j in range(0,len(ids),6)])
    import lpips
    metric=lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    injection=list(model.unet.conv_in.spatial.parameters());injection_ids={id(v) for v in injection}
    optimizer=torch.optim.AdamW([{'params':model.adapter.parameters(),'lr':cfg['readout_lr']},
        {'params':model.spatial.parameters(),'lr':cfg['readout_lr']},
        {'params':[v for v in model.unet.parameters() if v.requires_grad and id(v) not in injection_ids],'lr':cfg['unet_lr']},
        {'params':injection,'lr':cfg['readout_lr']},
        {'params':[v for v in qwen.parameters() if v.requires_grad],'lr':cfg['qwen_lr']}],weight_decay=.01)
    params=[v for v in model.parameters() if v.requires_grad];rates=[g['lr'] for g in optimizer.param_groups]
    generator=torch.Generator(device=device).manual_seed(cfg['seed']);history=[];best=float('inf');before_images=None
    chosen=[]
    for category in ('lamp','pillow'):
        chosen += [j for j,i in enumerate(ids) if payload['rows'][i]['category']==category and j<nfit][:2]
        chosen += [j for j,i in enumerate(ids) if payload['rows'][i]['category']==category and j>=nfit][:1]
    @torch.no_grad()
    def evaluate(step):
        model.eval();records=[];outputs=[]
        for j,i in enumerate(ids):
            h,valid=model.raw_features(inference,[j],device)
            # Exactly the same noise for every reference: product-specific
            # random seeds cannot identify the reconstruction target here.
            noise=torch.randn(1,4,32,32,device=device,generator=torch.Generator(device=device).manual_seed(cfg['evaluation_seed']))
            z,rgb=model(h,valid,noise);rgb=rgb.clamp(-1,1);target=references[j:j+1]
            mse=float((rgb-target).square().mean()/4)
            records.append({'sample_id':payload['rows'][i]['sample_id'],'category':payload['rows'][i]['category'],
                'group':'identity_fit_TRAIN' if j<nfit else 'identity_transfer_within_original_TRAIN',
                'PSNR':-10*math.log10(max(mse,1e-12)),'LPIPS':float(metric(rgb,target).mean()),
                'latent_MSE':float(F.mse_loss(z,goal[j:j+1]))})
            outputs.append(tensor_image(rgb[0]))
        result={'step':step,'groups':{},'images':records}
        for group in {v['group'] for v in records}:
            rows=[v for v in records if v['group']==group]
            result['groups'][group]={'n':len(rows),**{k:sum(v[k] for v in rows)/len(rows) for k in ('PSNR','LPIPS','latent_MSE')}}
        panel=Image.new('RGB',(768,310*len(chosen)),'white');draw=ImageDraw.Draw(panel)
        for pos,j in enumerate(chosen):
            y=pos*310;r=records[j]
            draw.text((4,y+2),'IDENTITY DIAGNOSTIC ONLY | '+r['group'],fill='black')
            draw.text((4,y+15),cfg['identity_prompt']+f" | step{step} P{r['PSNR']:.2f} L{r['LPIPS']:.3f}",fill='black')
            draw.text((4,y+34),'INPUT                             INHERITED EDITOR                   IDENTITY-TRAINED',fill='black')
            panel.paste(tensor_image(inference['reference'][j]),(0,y+54))
            panel.paste(outputs[j] if before_images is None else before_images[j],(256,y+54));panel.paste(outputs[j],(512,y+54))
        panel.save(a.output/'identity_latest.png')
        return result,outputs
    initial,before_images=evaluate(0);write_json(a.output/'before.json',initial);(a.output/'identity_before.png').write_bytes((a.output/'identity_latest.png').read_bytes())
    def save(step,result,path):
        return save_with_digest({'schema':'qwen_identity_diagnostic_v1','step':step,'config':cfg,
            'generator_config':state['config'],'visual_layout':state['visual_layout'],
            'parent_checkpoint_sha256':parent['checkpoint_sha256'],'qwen_joint_lora':joint_adapter_state(qwen),
            'adapter':model.adapter.state_dict(),'spatial':model.spatial.state_dict(),
            'unet_trainable':{n:v.detach().cpu().clone() for n,v in model.unet.named_parameters() if v.requires_grad},'metrics':result},path)
    for step in range(1,cfg['steps']+1):
        model.train();model.vae.eval();optimizer.zero_grad(set_to_none=True);batch=rng.sample(range(nfit),cfg['batch_size'])
        factor=min(1.,step/cfg['warmup_steps'])*(.1+.9*.5*(1+math.cos(math.pi*step/cfg['steps'])))
        for g,rate in zip(optimizer.param_groups,rates):g['lr']=rate*factor
        h,valid=model.raw_features(inference,batch,device)
        noise=torch.randn(len(batch),4,32,32,device=device,generator=generator)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,rgb=model(h,valid,noise);target=references[batch]
            latent=F.mse_loss(z.float(),goal[batch]);pixel=F.l1_loss(rgb.float(),target)
            perceptual=metric(rgb.float().clamp(-1,1),target).mean()
            loss=cfg['latent_weight']*latent+cfg['pixel_weight']*pixel+cfg['lpips_weight']*perceptual
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite identity diagnostic loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite identity gradients')
        optimizer.step()
        if step==4:
            gradients={branch:sum(float(v.grad.abs().sum()) for n,v in qwen.named_parameters() if branch in n and v.requires_grad and v.grad is not None) for branch in ('model.visual.','model.language_model.')}
            if not all(v>0 and math.isfinite(v) for v in gradients.values()):raise ValueError('Both raw modalities must receive real gradients')
            write_json(a.output/'smoke_report.json',{'real_steps':4,'finite_loss_gradient':True,'Qwen_LoRA_gradients':gradients,'TRAIN_only':True})
        if step%10==0:write_json(a.output/'progress.json',{'step':step,'steps':cfg['steps'],'TRAIN_loss':float(loss.detach())})
        if step%cfg['evaluate_every']==0 or step==cfg['steps']:
            result,images=evaluate(step);history.append(result);write_json(a.output/'history.json',history)
            save(step,result,a.output/'last_checkpoint.pt');score=result['groups']['identity_fit_TRAIN']['LPIPS']
            if score<best:
                best=score;best_result=result;best_digest=save(step,result,a.output/'best_checkpoint.pt')
                (a.output/'identity_best.png').write_bytes((a.output/'identity_latest.png').read_bytes())
                for j,image in enumerate(images):
                    folder=a.output/'images'/'identity_best';folder.mkdir(parents=True,exist_ok=True);image.save(folder/f'sample_{j:04d}.png')
    selected=torch.load(a.output/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    load_joint_adapters(qwen,selected['qwen_joint_lora']);model.adapter.load_state_dict(selected['adapter']);model.spatial.load_state_dict(selected['spatial'])
    with torch.no_grad():
        for name,value in model.unet.named_parameters():
            if name in selected['unet_trainable']:value.copy_(selected['unet_trainable'][name].to(value))
    reloaded,images=evaluate(selected['step']);checks=[]
    for j,image in enumerate(images):
        expected=Image.open(a.output/'images'/'identity_best'/f'sample_{j:04d}.png').convert('RGB')
        checks.append(image.tobytes()==expected.tobytes())
    write_json(a.output/'selected_reload_identity.json',{'passed':all(checks),'raw_RGB_images':len(checks),'checkpoint_sha256':best_digest})
    if not all(checks):raise ValueError('Saved diagnostic adapters do not reproduce selected raw RGB')
    write_json(a.output/'report.json',{'complete':True,'diagnostic_only':True,'editing_quality_accepted':False,
        'independent_TEST_opened':False,'DEV_used':False,'fit_examples':nfit,'transfer_examples':len(transfer),
        'transfer_is_original_TRAIN_not_independent_test':True,'before':initial,'best':best_result,'best_selection':'identity fit TRAIN LPIPS only',
        'checkpoint_sha256':best_digest,'parent_checkpoint_sha256':parent['checkpoint_sha256'],'source_commit':os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256':sha256(Path(__file__)),'fresh_random_TRAIN_noise':True,'shared_noise_across_all_evaluation_references':cfg['evaluation_seed'],
        'forward_inputs':'Reference RGB and identity instruction only through actual full Qwen; no reference VAE input, GT latent, masks or identifiers',
        'limits':'Small reference-identity fit is a capacity diagnostic, not editing success or evidence of unique causation. Transfer references belong to the original TRAIN cohort.'})


if __name__=='__main__':main()
