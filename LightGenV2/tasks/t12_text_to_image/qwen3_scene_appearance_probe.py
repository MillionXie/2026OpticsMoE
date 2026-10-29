"""No-recolor furnished scenes, matched existing vs text-readout adaptation.

Original Qwen/SDXL/VAE/image reader remain fixed. Native CLIP is TRAIN-only
conditioning supervision. This is a development pilot, not paired diffusion
training or proof that the old model learned color editing.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import traceback
import torch
import torch.nn.functional as F
from PIL import Image
from .appearance_scene_protocol import PROFILE
from .qwen3_scene_conditioning import QwenSDXLConditions,cache_inputs,batch,fresh_qwen,raw_features,qwen_asset_manifest
from .white_reference_scene_probe import GPU,digest,write,model_manifest


def freeze_appearance(head):
    head.requires_grad_(False)
    head.text.requires_grad_(True);head.pooled.requires_grad_(True)
    return head


def text_loss(pred,target,lengths):
    position=torch.arange(77,device=pred['text'].device)[None,:]
    valid=position<torch.tensor(lengths,device=position.device)[:,None]
    difference=(pred['text'].float()-target['text'].float()).square().mean(-1)
    scale=target['text'].float().square().mean().clamp_min(1e-6)
    active=(difference*valid).sum()/valid.sum().clamp_min(1)/scale
    padding=(difference*(~valid)).sum()/(~valid).sum().clamp_min(1)/scale
    pooled=(pred['pooled'].float()-target['pooled'].float()).square().mean()/target['pooled'].float().square().mean().clamp_min(1e-6)
    cosine=(1-F.cosine_similarity(pred['pooled'].float(),target['pooled'].float(),dim=-1)).mean()
    return active+.15*padding+pooled+.1*cosine


def fit(a,cache,parent):
    def create():
        head=freeze_appearance(QwenSDXLConditions().cuda());head.load_state_dict(parent['state_dict'],strict=True)
        return head,torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    def evaluate(head):
        values=[];head.eval()
        with torch.no_grad():
            for start in range(0,len(cache['records']),8):
                indices=list(range(start,min(start+8,len(cache['records']))));h,v,m,t=batch(cache,indices)
                lengths=[max(cache['text_targets'][cache['records'][i]['text_key']]['token_lengths']) for i in indices]
                values.append(text_loss(head(h,v,m),t,lengths).item())
        head.train();return sum(values)/len(values)
    head,opt=create();initial=evaluate(head);smoke=[]
    for step in range(4):
        indices=list(range(step*2,step*2+2));h,v,m,t=batch(cache,indices)
        lengths=[max(cache['text_targets'][cache['records'][i]['text_key']]['token_lengths']) for i in indices]
        loss=text_loss(head(h,v,m),t,lengths);opt.zero_grad();loss.backward()
        grad=sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
        assert torch.isfinite(loss) and grad>0 and all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        assert all(p.grad is None for p in list(head.image.parameters())+list(head.project.parameters()))
        opt.step();smoke.append({'step':step+1,'loss':loss.item(),'gradient_absolute_sum':grad})
    write(a.output/'smoke_report.json',{'complete':True,'steps':smoke,'image_and_shared_projection_frozen':True})
    del head,opt;gc.collect();torch.cuda.empty_cache();head,opt=create();rng=random.Random(20261029);best=float('inf');history=[]
    for step in range(1,a.steps+1):
        indices=rng.sample(range(len(cache['records'])),8);h,v,m,t=batch(cache,indices)
        lengths=[max(cache['text_targets'][cache['records'][i]['text_key']]['token_lengths']) for i in indices]
        loss=text_loss(head(h,v,m),t,lengths);assert torch.isfinite(loss);opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad],1.);opt.step()
        if step%20==0:write(a.output/'progress.json',{'stage':'TRAIN_text_readout_adaptation','step':step,'total':a.steps,'loss':loss.item()})
        if step%100==0 or step==a.steps:
            value=evaluate(head);history.append({'step':step,'TRAIN_text_condition_loss':value});write(a.output/'history.json',history)
            for n,p in head.state_dict().items():
                if n.startswith(('image.','project.')):assert torch.equal(p.cpu(),parent['state_dict'][n])
            state={**parent,'state_dict':head.state_dict(),'step':step,'source_commit':os.environ['T12_SOURCE_COMMIT'],
              'prompt_profile':PROFILE,'TRAIN_text_condition_loss':value,'parent_sha256':digest(a.parent/'best_checkpoint.pt'),
              'cache_sha256':digest(a.output/'TRAIN_condition_cache.pt'),'image_and_project_unchanged':True}
            torch.save(state,a.output/'last_checkpoint.pt')
            if value<best:best=value;torch.save(state,a.output/'best_checkpoint.pt')
    write(a.output/'adaptation_report.json',{'complete':True,'initial_TRAIN_text_loss':initial,'best_TRAIN_text_loss':best,'steps':a.steps,
      'best_sha256':digest(a.output/'best_checkpoint.pt'),'trainable_parameters':sum(p.numel() for p in head.parameters() if p.requires_grad),
      'selection':'TRAIN native text conditions only, not generated image quality/DEV/TEST','image_and_shared_projection_unchanged':True,
      'no_added_deployment_parameters':True,'paired_diffusion_training_performed':False})
    del head,opt;gc.collect();torch.cuda.empty_cache()


def render(a,cache,parent):
    from diffusers import StableDiffusionXLPipeline
    qwen,processor=fresh_qwen(a.qwen)
    pipe=StableDiffusionXLPipeline.from_pretrained(a.model_path,variant='fp16',torch_dtype=torch.float16,
      text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,image_encoder=None,local_files_only=True).to('cuda')
    pipe.load_ip_adapter(str(a.adapter_path),subfolder='sdxl_models',weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',image_encoder_folder=None,local_files_only=True)
    assert pipe.text_encoder is None and pipe.text_encoder_2 is None and pipe.image_encoder is None
    pipe.set_ip_adapter_scale(.25);pipe.set_progress_bar_config(disable=True)
    for c in pipe.components.values():
        if isinstance(c,torch.nn.Module):c.eval().requires_grad_(False)
    chosen=[r['row'] for r in cache['records'] if r['row']['condition_id'] in ['0','1']];assert len(chosen)==16
    focused=torch.load(a.output/'best_checkpoint.pt',map_location='cpu',weights_only=False);variants={}
    for name,state in [('existing',parent),('focused',focused)]:
        head=QwenSDXLConditions().cuda().eval();head.load_state_dict(state['state_dict'],strict=True);out=a.output/name;out.mkdir();records=[]
        for i,row in enumerate(chosen):
            src=a.data/row['reference'];dst=out/(row['reference_id']+'.png')
            if not dst.exists():shutil.copyfile(src,dst)
            with Image.open(src) as im:h,v,ids=raw_features(qwen,processor,im.convert('RGB'),row['prompt'])
            old=next(r for r in cache['records'] if r['row']['sample_id']==row['sample_id'])
            assert torch.equal(h,old['hidden']) and torch.equal(v,old['visual']) and torch.equal(ids,old['input_ids'])
            with torch.inference_mode():
                pred=head(h.unsqueeze(0).cuda().float(),v.unsqueeze(0).cuda().float())
                image_condition=torch.cat([parent['negative_image_condition'].cuda().half(),pred['image'].half()],dim=0).unsqueeze(1)
                generator=torch.Generator(device='cuda').manual_seed(row['seed']);noise=torch.randn((1,4,128,128),generator=generator,device='cuda',dtype=torch.float16)
                noise_sha=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                image=pipe(prompt_embeds=pred['text'].half(),pooled_prompt_embeds=pred['pooled'].half(),negative_prompt_embeds=torch.zeros_like(pred['text']).half(),
                  negative_pooled_prompt_embeds=torch.zeros_like(pred['pooled']).half(),ip_adapter_image_embeds=[image_condition],latents=noise,generator=generator,
                  width=1024,height=1024,num_inference_steps=30,guidance_scale=7.5).images[0]
            filename=row['sample_id']+'.png';image.save(out/filename)
            records.append({**row,'generated_PNG':filename,'generated_sha256':digest(out/filename),'input_sha256':digest(src),
              'initial_latent_sha256':noise_sha,'raw_Qwen_cache_maximum_error':0.,'checkpoint_origin':name})
            write(out/'images.json',records);write(a.output/'progress.json',{'stage':'raw_Qwen_1024_generation','variant':name,'done':i+1,'total':16})
        write(out/'report.json',{'complete':True,'prompt_profile':PROFILE,'source_commit':os.environ['T12_SOURCE_COMMIT'],'generated_count':16,
          'training_performed':False,'rendering_weights_frozen':True,'condition_training_origin':'previous1200_color_condition_warmup' if name=='existing' else 'same_parent_plus_TRAIN_text_readout_adaptation',
          'checkpoint_sha256':digest(a.parent/'best_checkpoint.pt') if name=='existing' else digest(a.output/'best_checkpoint.pt'),
          'independent_TEST_opened':False,'CLIP_encoders_loaded_at_inference':False,'reference_VAE_encoder_called':False,
          'input_whitelist':['white reference RGB','complete no-recolor prompt','source-derived seed'],'quality_accepted':False,'resolution':1024})
        variants[name]=records;del head;gc.collect();torch.cuda.empty_cache()
    assert [(r['sample_id'],r['prompt'],r['initial_latent_sha256']) for r in variants['existing']]==[(r['sample_id'],r['prompt'],r['initial_latent_sha256']) for r in variants['focused']]
    del qwen,pipe;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','parent','qwen','model-path','adapter-path','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--steps',type=int,default=600);a=p.parse_args();commit=os.environ.get('T12_SOURCE_COMMIT','')
    if a.output.exists() or len(commit)!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('New run/published source/GPU6 required')
    assert json.loads((a.data/'report.json').read_text())['prompt_profile']==PROFILE
    rows=[r for r in json.loads((a.data/'pilot_train_requests.json').read_text()) if r['seed_index']==0]
    assert len(rows)==64 and all(r['split']=='train' for r in rows)
    parent_report=json.loads((a.parent/'alignment_report.json').read_text());assert parent_report['complete'] and digest(a.parent/'best_checkpoint.pt')==parent_report['best_sha256']
    parent=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.7,0);a.use_native_caption_targets=True
    write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':commit,'source_sha256':digest(Path(__file__)),
      'request_sha256':digest(a.data/'requests.json'),'parent_sha256':parent_report['best_sha256'],'prompt_profile':PROFILE,
      'Qwen_UNet_VAE_base_unchanged':True,'added_parameters':0,'paired_scene_GT_used':False})
    try:
        assets={'Qwen':qwen_asset_manifest(a.qwen),'SDXL':model_manifest(a.model_path),'IPAdapter':model_manifest(a.adapter_path)}
        assert assets==json.loads((a.parent/'model_assets.json').read_text()),'Public pretrained model assets changed'
        write(a.output/'model_assets.json',assets)
        cache=cache_inputs(a,rows);fit(a,cache,parent);render(a,cache,parent)
        write(a.output/'report.json',{'complete':True,'source_commit':commit,'prompt_profile':PROFILE,'generated_count':32,
          'TRAIN_condition_rows':64,'independent_TEST_opened':False,'paired_diffusion_training_performed':False,'same_initial_noise_verified':True,'quality_accepted':False})
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
