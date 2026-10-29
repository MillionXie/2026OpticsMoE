"""Fresh complete Qwen3-VL -> pretrained SDXL conditions, TRAIN-only pilot.

No synthetic scene GT is needed for this interface warm-up. Native CLIP text and
image encoders are teachers only, deleted before raw Qwen generation. Low feature
loss does not establish generation quality. This does not yet perform paired
diffusion fine-tuning, nor use DEV/TEST to choose a checkpoint.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from .white_reference_scene_probe import GPU,digest,write,checked_tokens,model_manifest


class QueryCondition(nn.Module):
    def __init__(self,queries,output,width=512):
        super().__init__();self.query=nn.Parameter(torch.randn(1,queries,width)*.02)
        self.attention=nn.MultiheadAttention(width,8,batch_first=True)
        self.norm=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.out=nn.Linear(width,output)
    def forward(self,x,mask=None):
        q=self.query.expand(x.shape[0],-1,-1)
        q=q+self.attention(q,x,x,key_padding_mask=mask,need_weights=False)[0]
        q=q+self.ff(self.norm(q));return self.out(q),q


class QwenSDXLConditions(nn.Module):
    def __init__(self,input_width=2048,width=512,text_width=2048,image_width=1280,pooled_width=1280):
        super().__init__();self.project=nn.Sequential(nn.LayerNorm(input_width),nn.Linear(input_width,width))
        self.text=QueryCondition(77,text_width,width);self.image=QueryCondition(257,image_width,width)
        self.pooled=nn.Linear(width,pooled_width)
    def forward(self,hidden,visual,padding=None):
        text,q=self.text(self.project(hidden),padding)
        image,_=self.image(self.project(visual))
        return {'text':text,'image':image,'pooled':self.pooled(q.mean(dim=1))}


def alignment_loss(pred,target):
    losses={}
    for name in ['text','image','pooled']:
        scale=target[name].float().square().mean().clamp_min(1e-6)
        mse=(pred[name].float()-target[name].float()).square().mean()/scale
        cosine=(1-F.cosine_similarity(pred[name].float(),target[name].float(),dim=-1)).mean()
        losses[name]=mse+.1*cosine
    return sum(losses.values()),losses


def fresh_qwen(path):
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    model=Qwen3VLForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
    assert model.config.vision_config.depth==24 and model.config.text_config.num_hidden_layers==28
    processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=65536,max_pixels=65536)
    return model,processor


def raw_features(model,processor,image,prompt):
    # All native final states are visible to the text resampler, including the
    # instruction states that attended to the reference. Image conditions come
    # from the SAME full Qwen visual merger, with no external RGB encoder.
    content=[{'type':'image','image':image},{'type':'text','text':prompt}]
    text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=[text],images=[image],return_tensors='pt').to('cuda');native=[]
    if not inputs['attention_mask'].all() or inputs['image_grid_thw'].tolist()!=[[1,16,16]]:raise ValueError('Native B1/grid changed')
    hook=model.model.visual.register_forward_hook(lambda m,a,out:native.append(out[0]))
    try:
        with torch.inference_mode():hidden=model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
    finally:hook.remove()
    if len(native)!=1 or native[0].shape!=(64,2048):raise ValueError('Complete Qwen visual merger changed')
    return hidden[0].cpu().bfloat16(),native[0].cpu().bfloat16(),inputs['input_ids'].cpu()


def cache_inputs(a,rows):
    from transformers import CLIPTokenizer,CLIPTextModel,CLIPTextModelWithProjection,CLIPVisionModelWithProjection,CLIPImageProcessor
    qwen,processor=fresh_qwen(a.qwen)
    encoders=[cls.from_pretrained(a.model_path/f'text_encoder{suffix}',variant='fp16',torch_dtype=torch.float16,local_files_only=True).cuda().eval().requires_grad_(False)
      for cls,suffix in [(CLIPTextModel,''),(CLIPTextModelWithProjection,'_2')]]
    tokenizers=[CLIPTokenizer.from_pretrained(a.model_path/f'tokenizer{suffix}',local_files_only=True) for suffix in ['','_2']]
    vision=CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',torch_dtype=torch.float16,local_files_only=True).cuda().eval().requires_grad_(False)
    image_processor=CLIPImageProcessor.from_pretrained(a.model_path/'feature_extractor',local_files_only=True) if (a.model_path/'feature_extractor').exists() else CLIPImageProcessor()
    text_targets={};image_targets={};cache=[];negative=None
    for i,row in enumerate(rows):
        path=a.data/row['reference'];assert digest(path)==row['reference_sha256']
        with Image.open(path) as im:im=im.convert('RGB');hidden,visual,ids=raw_features(qwen,processor,im,row['prompt'])
        key=hashlib.sha256(row['prompt'].encode()).hexdigest()
        if key not in text_targets:
            features=[];lengths=[]
            for tokenizer,encoder in zip(tokenizers,encoders):
                lengths.append(checked_tokens(tokenizer,row['prompt'])['full_token_count'])
                tokens=tokenizer(row['prompt'],padding='max_length',max_length=77,truncation=False,return_tensors='pt').input_ids.cuda()
                with torch.inference_mode():output=encoder(tokens,output_hidden_states=True)
                features.append(output.hidden_states[-2].cpu().half())
            text_targets[key]={'text':torch.cat(features,dim=-1)[0],'pooled':output.text_embeds[0].cpu().half(),'token_lengths':lengths}
        if row['reference_id'] not in image_targets:
            with Image.open(path) as im:pixels=image_processor(images=im.convert('RGB'),return_tensors='pt').pixel_values.cuda().half()
            with torch.inference_mode():
                features=vision(pixels,output_hidden_states=True).hidden_states[-2]
                if negative is None:negative=vision(torch.zeros_like(pixels),output_hidden_states=True).hidden_states[-2].cpu().half()
            assert features.shape==(1,257,1280)
            image_targets[row['reference_id']]=features[0].cpu().half()
        cache.append({'row':row,'hidden':hidden,'visual':visual,'input_ids':ids,'text_key':key,
          'input_fingerprint':hashlib.sha256(path.read_bytes()+b'\0'+row['prompt'].encode()).hexdigest()})
        write(a.output/'progress.json',{'stage':'fresh_full_Qwen_and_native_TRAIN_teacher_cache','done':i+1,'total':len(rows)})
    counts={'Qwen_all_parameters_frozen':sum(p.numel() for p in qwen.parameters()),'vision_layers':24,'language_layers':28}
    result={'records':cache,'text_targets':text_targets,'image_targets':image_targets,'negative_image':negative,'counts':counts}
    path=a.output/'TRAIN_condition_cache.pt';torch.save(result,path)
    write(a.output/'cache_report.json',{'complete':True,'n':len(cache),'sha256':digest(path),'TRAIN_only':True,'scene_GT_or_mask_used':False,
      'input_whitelist':['reference RGB','full prompt'],'CLIP_teacher_in_deployment':False,**counts})
    del qwen,encoders,vision,processor;gc.collect();torch.cuda.empty_cache()
    # Reload removes inference-tensor status before real backward through heads.
    return torch.load(path,map_location='cpu',weights_only=False)


def batch(cache,indices):
    rows=[cache['records'][i] for i in indices];length=max(r['hidden'].shape[0] for r in rows)
    h=torch.zeros(len(rows),length,2048);mask=torch.ones(len(rows),length,dtype=torch.bool)
    for i,r in enumerate(rows):h[i,:len(r['hidden'])]=r['hidden'].float();mask[i,:len(r['hidden'])]=False
    target={n:torch.stack([cache['text_targets'][r['text_key']][n] for r in rows]).cuda().float() for n in ['text','pooled']}
    target['image']=torch.stack([cache['image_targets'][r['row']['reference_id']] for r in rows]).cuda().float()
    return h.cuda(),torch.stack([r['visual'] for r in rows]).cuda().float(),mask.cuda(),target


def fit(a,cache):
    def new():
        torch.manual_seed(20261029);model=QwenSDXLConditions().cuda();return model,torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    model,opt=new();smoke=[]
    for step in range(4):
        h,v,m,t=batch(cache,list(range(step*2,step*2+2)));loss,_=alignment_loss(model(h,v,m),t)
        opt.zero_grad();loss.backward();grad=sum(p.grad.float().abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert torch.isfinite(loss) and grad>0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step();smoke.append({'step':step+1,'loss':loss.item(),'absolute_gradient_sum':grad})
    write(a.output/'smoke_report.json',{'complete':True,'actual_TRAIN_backward':smoke,'teachers_frozen_and_unloaded':True})
    del model,opt;gc.collect();torch.cuda.empty_cache();model,opt=new();history=[];best=float('inf');rng=random.Random(20261029)
    for step in range(1,a.steps+1):
        h,v,m,t=batch(cache,rng.sample(range(len(cache['records'])),8));loss,parts=alignment_loss(model(h,v,m),t)
        if not torch.isfinite(loss):raise ValueError('Nonfinite condition training loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%20==0:write(a.output/'progress.json',{'stage':'TRAIN_condition_alignment','step':step,'total':a.steps,'loss':loss.item()})
        if step%200==0 or step==a.steps:
            model.eval();values=[]
            with torch.no_grad():
                for start in range(0,len(cache['records']),8):
                    h,v,m,t=batch(cache,list(range(start,min(start+8,len(cache['records'])))));val,parts=alignment_loss(model(h,v,m),t);values.append(val.item())
            mean=sum(values)/len(values);history.append({'step':step,'TRAIN_condition_loss':mean});write(a.output/'history.json',history)
            state={'schema':'fresh_qwen3_sdxl_conditions_v1','state_dict':model.state_dict(),'step':step,'TRAIN_condition_loss':mean,
              'source_commit':os.environ['T12_SOURCE_COMMIT'],'cache_sha256':digest(a.output/'TRAIN_condition_cache.pt'),
              'negative_image_condition':cache['negative_image'],
              'inference_parameters':sum(p.numel() for p in model.parameters()),'not_yet_diffusion_finetuned':True}
            torch.save(state,a.output/'last_checkpoint.pt')
            if mean<best:best=mean;torch.save(state,a.output/'best_checkpoint.pt')
            model.train()
    write(a.output/'alignment_report.json',{'complete':True,'steps':a.steps,'TRAIN_condition_loss_best':best,
      'best_sha256':digest(a.output/'best_checkpoint.pt'),'best_selected_by':'TRAIN condition loss only, not image quality or independent generalization',
      'learned_parameters':sum(p.numel() for p in model.parameters()),'scene_GT_used':False})
    del model,opt;gc.collect();torch.cuda.empty_cache()


def raw_generation(a,cache):
    from diffusers import StableDiffusionXLPipeline
    state=torch.load(a.output/'best_checkpoint.pt',map_location='cpu',weights_only=False);assert state['step']>0
    head=QwenSDXLConditions().cuda().eval();head.load_state_dict(state['state_dict'],strict=True)
    qwen,processor=fresh_qwen(a.qwen)
    # No native text or image encoders are loaded at inference. Both conditions
    # now come from the complete Qwen entry point; VAE is decoder only.
    pipe=StableDiffusionXLPipeline.from_pretrained(a.model_path,variant='fp16',torch_dtype=torch.float16,
      text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,image_encoder=None,local_files_only=True).to('cuda')
    pipe.load_ip_adapter(str(a.adapter_path),subfolder='sdxl_models',weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',image_encoder_folder=None,local_files_only=True)
    assert pipe.text_encoder is None and pipe.text_encoder_2 is None and pipe.image_encoder is None
    pipe.set_ip_adapter_scale(.25);pipe.set_progress_bar_config(disable=True)
    for c in pipe.components.values():
        if isinstance(c,nn.Module):c.eval().requires_grad_(False)
    selected=[r for r in json.loads((a.data/'pilot_train_requests.json').read_text()) if r['seed_index']==0 and r['attributes']['product_color'] in ['deep blue','terracotta']]
    assert len(selected)==16
    out=a.output/'raw_TRAIN_generation';out.mkdir();records=[]
    for i,row in enumerate(selected):
        with Image.open(a.data/row['reference']) as im:h,v,ids=raw_features(qwen,processor,im.convert('RGB'),row['prompt'])
        previous=next(r for r in cache['records'] if r['row']['sample_id']==row['sample_id'])
        error=max((h.float()-previous['hidden'].float()).abs().max().item(),(v.float()-previous['visual'].float()).abs().max().item())
        if error!=0 or not torch.equal(ids,previous['input_ids']):raise ValueError('Actual raw frontend cache identity failed')
        with torch.inference_mode():
            cond=head(h.unsqueeze(0).cuda().float(),v.unsqueeze(0).cuda().float())
            image_condition=torch.cat([state['negative_image_condition'].cuda().half(),cond['image'].half()],dim=0).unsqueeze(1)
            generator=torch.Generator(device='cuda').manual_seed(row['seed'])
            noise=torch.randn((1,4,128,128),generator=generator,device='cuda',dtype=torch.float16)
            noise_sha=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            image=pipe(prompt_embeds=cond['text'].half(),pooled_prompt_embeds=cond['pooled'].half(),
              negative_prompt_embeds=torch.zeros_like(cond['text']).half(),negative_pooled_prompt_embeds=torch.zeros_like(cond['pooled']).half(),
              ip_adapter_image_embeds=[image_condition],latents=noise,generator=generator,width=1024,height=1024,
              num_inference_steps=30,guidance_scale=7.5).images[0]
        name=row['sample_id']+'.png';image.save(out/name)
        records.append({**row,'generated_PNG':name,'generated_sha256':digest(out/name),'raw_Qwen_cache_maximum_error':error,
          'initial_latent_sha256':noise_sha,'best_condition_checkpoint_sha256':digest(a.output/'best_checkpoint.pt')})
        write(out/'images.json',records);write(a.output/'progress.json',{'stage':'actual_raw_Qwen_only_TRAIN_generation','done':i+1,'total':16})
    write(out/'report.json',{'complete':True,'generated_count':len(records),'CLIP_encoders_loaded':False,'reference_VAE_encoder_called':False,
      'DEV_or_TEST_inference':False,'quality_accepted':False,'GT_or_mask_used':False,'input_whitelist':['reference RGB','full prompt','source-derived seed'],
      'Qwen_full_frozen_parameters':cache['counts']['Qwen_all_parameters_frozen'],'adapter_parameters':state['inference_parameters'],
      'loaded_SDXL_parameters':sum(p.numel() for p in {id(p):p for c in pipe.components.values() if isinstance(c,nn.Module) for p in c.parameters()}.values()),
      'fixed_negative_image_buffer_elements':state['negative_image_condition'].numel(),'IPAdapter_scale':.25,
      'limitation':'TRAIN pilot after condition-only alignment; not paired diffusion training or independent performance.'})
    del head,qwen,pipe;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','qwen','model-path','adapter-path','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200);a=p.parse_args();commit=os.environ.get('T12_SOURCE_COMMIT','')
    if a.output.exists() or len(commit)!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('New run/published source/GPU6 required')
    audit=json.loads((a.data/'data_audit.json').read_text());assert audit['role_disjoint']
    refs={r['reference_id'] for r in json.loads((a.data/'pilot_train_requests.json').read_text())}
    rows=[r for r in json.loads((a.data/'requests.json').read_text()) if r['split']=='train' and r['reference_id'] in refs]
    assert len(rows)==256 and len(refs)==8
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.65,0)
    write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':commit,'source_sha256':digest(Path(__file__)),
      'request_sha256':digest(a.data/'requests.json'),'phase':'TRAIN-only input-condition warmup, not completed baseline image training'})
    try:
        write(a.output/'model_assets.json',{'Qwen':model_manifest(a.qwen),'SDXL':model_manifest(a.model_path),'IPAdapter':model_manifest(a.adapter_path)})
        cache=cache_inputs(a,rows);fit(a,cache);raw_generation(a,cache)
        write(a.output/'report.json',{'complete':True,'source_commit':commit,'n_TRAIN_conditions':len(rows),
          'paired_scene_training_performed':False,'DEV_or_TEST_inference':False,'generation_quality_established':False,'raw_TRAIN_generated':16,
          'next_stage':'Review all raw Qwen-only generations, then audited TRAIN paired diffusion fine-tuning.'})
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
