"""Matched, target-free DEV comparison; Qwen once, diffusion sampling allowed.

The Qwen and native encoders feed the same frozen SDXL/IP-Adapter generator.
A second native reference uses the public Canny ControlNet. All variants use
the same white RGB, complete canonical request, fixed caption translation,
initial latent bytes, 1024 resolution and 30 internal denoising steps. No
generated image is fed back, no rotation or TEST prediction is performed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from PIL import Image
from .appearance_scene_protocol import PROFILE, conditions
from .white_reference_scene_probe import GPU, digest, write, checked_tokens, model_manifest


def select_requests(rows):
    selected = [r for r in rows if r['split'] == 'dev' and r['condition_id'] in ['0', '1']]
    if len(selected) != 32 or len({r['reference_id'] for r in selected}) != 16:
        raise ValueError('Prespecified sixteen DEV views x two scenes required')
    if len({(r['reference'], r['prompt'], r['seed']) for r in selected}) != 32:
        raise ValueError('Duplicate model inputs')
    for r in selected:
        expected = conditions(r['category'])[int(r['condition_id'])]
        if r['prompt_profile'] != PROFILE or any(r[k] != expected[k] for k in ['prompt', 'native_caption']):
            raise ValueError('Canonical no-recolor request changed')
    return selected


def run(a):
    import torch
    import numpy as np
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel
    from transformers import CLIPVisionModelWithProjection
    from .qwen3_scene_conditioning import QwenSDXLConditions, fresh_qwen, raw_features, qwen_asset_manifest

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    source_report = json.loads((a.data/'report.json').read_text())
    assert source_report['prompt_profile'] == PROFILE and source_report['complete']
    rows = select_requests(json.loads((a.data/'requests.json').read_text()))
    state = torch.load(a.checkpoint/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    identity = json.loads((a.checkpoint/'adaptation_report.json').read_text())
    assert digest(a.checkpoint/'best_checkpoint.pt') == identity['best_sha256']
    assets = {'Qwen': qwen_asset_manifest(a.qwen), 'SDXL': model_manifest(a.model_path),
              'IPAdapter': model_manifest(a.adapter_path)}
    assert assets == json.loads((a.checkpoint/'model_assets.json').read_text()), 'Model assets changed'
    assets['ControlNet'] = model_manifest(a.controlnet_path)
    write(a.output/'model_assets.json', assets)
    write(a.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(),
          'source_commit': os.environ['T12_SOURCE_COMMIT'], 'source_sha256': digest(Path(__file__)),
          'request_sha256': digest(a.data/'requests.json'), 'head_sha256': identity['best_sha256'],
          'scope': 'DEV selection/diagnosis, not independent TEST', 'image_feedback': False})
    noise_identities = {}
    for variant in ['qwen_single_forward', 'native_sdxl_ip', 'native_sdxl_ip_canny']:
        out = a.output/variant
        out.mkdir()
        qwen = processor = head = None
        common = dict(variant='fp16', torch_dtype=torch.float16, local_files_only=True)
        if variant == 'qwen_single_forward':
            qwen, processor = fresh_qwen(a.qwen)
            head = QwenSDXLConditions().cuda().eval().requires_grad_(False)
            head.load_state_dict(state['state_dict'], strict=True)
            pipe = StableDiffusionXLPipeline.from_pretrained(a.model_path, **common,
                text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, image_encoder=None).to('cuda')
        else:
            encoder = CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',
                torch_dtype=torch.float16, local_files_only=True)
            if variant.endswith('canny'):
                control = ControlNetModel.from_pretrained(a.controlnet_path, **common)
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(a.model_path, controlnet=control,
                    image_encoder=encoder, **common).to('cuda')
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(a.model_path, image_encoder=encoder, **common).to('cuda')
        pipe.load_ip_adapter(str(a.adapter_path), subfolder='sdxl_models',
            weight_name='ip-adapter-plus_sdxl_vit-h.safetensors', image_encoder_folder=None, local_files_only=True)
        pipe.set_ip_adapter_scale(.25)
        pipe.set_progress_bar_config(disable=True)
        for component in pipe.components.values():
            if isinstance(component, torch.nn.Module):
                component.eval().requires_grad_(False)
        records = []
        for i, row in enumerate(rows):
            src = a.data/row['reference']
            assert digest(src) == row['reference_sha256']
            dst = out/(row['reference_id']+'.png')
            if not dst.exists(): shutil.copyfile(src, dst)
            with Image.open(src) as im: reference = im.convert('RGB').copy()
            generator = torch.Generator(device='cuda').manual_seed(row['seed'])
            noise = torch.randn((1, 4, 128, 128), device='cuda', dtype=torch.float16, generator=generator)
            noise_sha = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            previous = noise_identities.setdefault(row['sample_id'], noise_sha)
            assert previous == noise_sha, 'Initial latent mismatch between models'
            kwargs = dict(width=1024, height=1024, num_inference_steps=30, guidance_scale=7.5,
                          generator=generator, latents=noise)
            with torch.inference_mode():
                if variant == 'qwen_single_forward':
                    h, v, ids = raw_features(qwen, processor, reference, row['prompt'])
                    pred = head(h.unsqueeze(0).cuda().float(), v.unsqueeze(0).cuda().float())
                    image_condition = torch.cat([state['negative_image_condition'].cuda().half(),
                                                 pred['image'].half()], dim=0).unsqueeze(1)
                    kwargs.update(prompt_embeds=pred['text'].half(), pooled_prompt_embeds=pred['pooled'].half(),
                        negative_prompt_embeds=torch.zeros_like(pred['text']).half(),
                        negative_pooled_prompt_embeds=torch.zeros_like(pred['pooled']).half(),
                        ip_adapter_image_embeds=[image_condition])
                    assert pipe.text_encoder is None and pipe.text_encoder_2 is None and pipe.image_encoder is None
                    token_audit = {'Qwen_complete_input_tokens': int(ids.numel()), 'truncation': False}
                else:
                    token_audit = [checked_tokens(t, row['native_caption']) for t in [pipe.tokenizer, pipe.tokenizer_2]]
                    kwargs.update(prompt=row['native_caption'], ip_adapter_image=reference)
                    if variant.endswith('canny'):
                        import cv2
                        resized = np.asarray(reference.resize((1024, 1024), Image.Resampling.LANCZOS))
                        edges = cv2.Canny(resized, 100, 200)
                        kwargs.update(image=Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2)),
                                      controlnet_conditioning_scale=.35, control_guidance_end=.5)
                result = pipe(**kwargs).images[0]
            assert result.size == (1024, 1024)
            filename = row['sample_id']+'.png'
            result.save(out/filename)
            records.append({**row, 'generated_PNG': filename, 'generated_sha256': digest(out/filename),
                'input_sha256': digest(src), 'initial_latent_sha256': noise_sha,
                'model_prompt': row['prompt'] if variant == 'qwen_single_forward' else row['native_caption'],
                'token_audit': token_audit, 'model': variant, 'resolution': 1024, 'sampling_steps': 30})
            write(out/'images.json', records)
            write(a.output/'progress.json', {'stage': variant, 'done': i+1, 'total': len(rows)})
        count = sum(p.numel() for p in {id(p): p for c in pipe.components.values()
            if isinstance(c, torch.nn.Module) for p in c.parameters()}.values())
        write(out/'report.json', {'complete': True, 'generated_count': len(records), 'scope': 'DEV',
            'prompt_profile': PROFILE, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
            'training_performed': False, 'independent_TEST_opened': False,
            'loaded_generator_parameters': count, 'Qwen_parameters': 0 if qwen is None else sum(p.numel() for p in qwen.parameters()),
            'head_parameters': 0 if head is None else sum(p.numel() for p in head.parameters()),
            'image_feedback': False, 'rotation': False, 'resolution': 1024, 'sampling_steps': 30,
            'native_caption_translation_recorded': True, 'unique_GT_or_mask_used': False,
            'CLIP_encoders_loaded_at_inference': variant != 'qwen_single_forward', 'quality_accepted': False})
        del pipe, qwen, processor, head
        if variant != 'qwen_single_forward': del encoder
        if variant.endswith('canny'): del control
        gc.collect()
        torch.cuda.empty_cache()
    write(a.output/'report.json', {'complete': True, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
        'DEV_requests': 32, 'generated_count': 96, 'identical_initial_latents_verified': True,
        'independent_TEST_opened': False, 'quality_accepted': False, 'training_performed': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['data', 'checkpoint', 'qwen', 'model-path', 'adapter-path', 'controlnet-path', 'output']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT', '')) != 40 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('Fresh run, published source and GPU6 required')
    a.output.mkdir(parents=True)
    try: run(a)
    except Exception:
        write(a.output/'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
