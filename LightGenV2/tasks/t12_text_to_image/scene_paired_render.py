"""Raw paired-model DEV generation; never opens pairs.json/target latent cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import torch
from PIL import Image
from .qwen3_scene_conditioning import QwenSDXLConditions
from .qwen3_scene_paired_train import new_qwen, raw_scene_features, pipeline
from .white_reference_scene_probe import GPU, digest, write, checked_tokens


def inference_inputs(data):
    inputs = json.loads((data/'inference_dev_inputs.json').read_text())
    safe = {'sample_id','reference','prompt','seed','reference_id','reference_sha256','category',
            'condition_id','native_caption','reward_caption','prompt_profile'}
    if not inputs or not all(set(row) == safe for row in inputs):
        raise ValueError('Only reference/text/seed and public request metadata allowed at inference')
    return inputs


def render(a):
    from peft import LoraConfig
    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
    state = torch.load(a.checkpoint/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    a.qwen_side=state['Qwen_input_side'];resolution=state['resolution']
    previous = json.loads((a.checkpoint/'report.json').read_text())
    assert previous['complete'] and digest(a.checkpoint/'best_checkpoint.pt') == previous['best_sha256']
    inputs = inference_inputs(a.data)
    rows = [r for r in inputs if r['condition_id'] in ['0','1']]
    qwen = processor = head = identity = None
    if state['variant'] == 'qwen':
        qwen, processor = new_qwen(a)
        head = QwenSDXLConditions().cuda().eval().requires_grad_(False)
        head.load_state_dict(state['head'], strict=True)
        pipe = pipeline(a)
        identity = torch.load(a.cache/'DEV_frontend_identity_only.pt', map_location='cpu', weights_only=False)
        assert not identity['contains_targets_or_masks']
    else:
        encoder = CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',
            torch_dtype=torch.bfloat16, local_files_only=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(a.model_path, variant='fp16',
            torch_dtype=torch.bfloat16, image_encoder=encoder, local_files_only=True).to('cuda')
        pipe.load_ip_adapter(str(a.adapter_path), subfolder='sdxl_models',
            weight_name='ip-adapter-plus_sdxl_vit-h.safetensors', image_encoder_folder=None, local_files_only=True)
        pipe.set_ip_adapter_scale(.25); pipe.set_progress_bar_config(disable=True)
    pipe.unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian',
        target_modules=['to_q','to_k','to_v','to_out.0']), adapter_name='scene_pair')
    named = dict(pipe.unet.named_parameters())
    assert set(state['unet_lora']) == {n for n in named if 'lora_' in n}
    with torch.no_grad():
        for n, value in state['unet_lora'].items(): named[n].copy_(value)
    for c in pipe.components.values():
        if isinstance(c,torch.nn.Module): c.eval().requires_grad_(False)
    records = []
    for i,row in enumerate(rows):
        src = a.data/row['reference']; assert digest(src) == row['reference_sha256']
        dst = a.output/(row['reference_id']+'.png')
        if not dst.exists(): shutil.copyfile(src,dst)
        with Image.open(src) as im: reference = im.convert('RGB').copy()
        generator = torch.Generator(device='cuda').manual_seed(row['seed'])
        noise = torch.randn((1,4,resolution//8,resolution//8), generator=generator, device='cuda', dtype=torch.bfloat16)
        noise_sha = hashlib.sha256(noise.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
        kwargs = dict(width=resolution,height=resolution,num_inference_steps=30,guidance_scale=7.5,latents=noise,generator=generator)
        with torch.inference_mode():
            if head is not None:
                h,v,ids = raw_scene_features(qwen,processor,reference,row['prompt'],a.qwen_side)
                key = hashlib.sha256(src.read_bytes()+b'\0'+row['prompt'].encode()+f'\0Qwen{a.qwen_side}'.encode()).hexdigest()
                old = identity['features'][key]
                assert torch.equal(h,old['hidden']) and torch.equal(v,old['visual']) and torch.equal(ids,old['input_ids'])
                cond = head(h.unsqueeze(0).cuda().float(),v.unsqueeze(0).cuda().float())
                images = torch.cat([state['negative_image'].cuda().bfloat16(),cond['image'].bfloat16()],dim=0).unsqueeze(1)
                kwargs.update(prompt_embeds=cond['text'].bfloat16(),pooled_prompt_embeds=cond['pooled'].bfloat16(),
                    negative_prompt_embeds=torch.zeros_like(cond['text']).bfloat16(),
                    negative_pooled_prompt_embeds=torch.zeros_like(cond['pooled']).bfloat16(),ip_adapter_image_embeds=[images])
            else:
                for t in [pipe.tokenizer,pipe.tokenizer_2]: checked_tokens(t,row['native_caption'])
                kwargs.update(prompt=row['native_caption'],ip_adapter_image=reference)
            image = pipe(**kwargs).images[0]
        name = row['sample_id']+'.png'; image.save(a.output/name)
        records.append({**row,'split':'dev','generated_PNG':name,'generated_sha256':digest(a.output/name),
            'input_sha256':digest(src),'initial_latent_sha256':noise_sha,'model':state['variant'],
            'raw_Qwen_identity_exact':head is not None,'resolution':resolution,'sampling_steps':30})
        write(a.output/'images.json',records)
        write(a.output/'progress.json',{'stage':'independent_raw_DEV_render','done':i+1,'total':len(rows)})
    generator_count = sum(p.numel() for p in {id(p):p for c in pipe.components.values()
        if isinstance(c,torch.nn.Module) for p in c.parameters()}.values())
    write(a.output/'report.json',{'complete':True,'scope':'DEV','source_commit':os.environ['T12_SOURCE_COMMIT'],
        'generated_count':len(records),'prompt_profile':rows[0]['prompt_profile'],'training_performed':False,
        'checkpoint_sha256':previous['best_sha256'],'paired_diffusion_training_performed':True,
        'independent_TEST_opened':False,'GT_mask_or_target_latent_files_opened':False,
        'raw_Qwen_identity_checked':head is not None,'image_feedback':False,'quality_accepted':False,
        'CLIP_encoders_loaded_at_inference':head is None,'precision':'BF16','resolution':resolution,'sampling_steps':30,
        'generator_loaded_parameters':generator_count,'Qwen_parameters':0 if qwen is None else sum(p.numel() for p in qwen.parameters()),
        'head_parameters':0 if head is None else sum(p.numel() for p in head.parameters()),
        'factual_evaluation_captions':True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['data','checkpoint','qwen','cache','model-path','adapter-path','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT','')) != 40 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('Fresh run, published source and GPU6 required')
    a.output.mkdir(parents=True); torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ['T12_SOURCE_COMMIT'],
                                  'source_sha256':digest(Path(__file__))})
    try: render(a)
    except Exception:
        write(a.output/'failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__': main()
