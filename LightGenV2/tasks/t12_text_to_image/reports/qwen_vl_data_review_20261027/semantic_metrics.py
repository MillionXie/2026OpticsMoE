"""Offline CLIP editing diagnostics for saved VAL PNGs; never a training input."""
import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def caption(description, scene):
    room,tone,brightness,direction=scene
    return f'{description}. {room.replace("_", " ")}, {tone} tones, {brightness}, {direction} window light.'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','data_run','clip_repo','clip_weights','output'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Explicit authorized GPU required')
    report=read(args.run/'report.json');manifest=read(args.run/'overview/image_manifest.json')
    if not report['complete'] or report['test_evaluated'] or report['best_step']<=0:raise ValueError('Require VAL-selected completed model')
    if manifest['checkpoint_sha256']!=report['checkpoint_sha256']:raise ValueError('Image/model identity differs')
    descriptions=read(args.data_run/'product_descriptions.json')
    pairs={r['sample_id']:r for r in read(args.run/'active_pairs.json')['val']}
    if set(pairs)!={r['sample_id'] for r in report['best_val']['images']}:raise ValueError('VAL identity mismatch')
    vendor_sha=subprocess.check_output(['git','-C',str(args.clip_repo),'rev-parse','HEAD']).decode().strip()
    if vendor_sha!='d05afc436d78f1c48dc0dbf8e5980a9d471f35f6':raise ValueError('Unreviewed CLIP upstream')
    if not args.clip_weights.is_file():raise ValueError('Local CLIP checkpoint required; do not download silently')
    sys.path.insert(0,str(args.clip_repo))
    import clip
    torch.set_num_threads(4);torch.manual_seed(1042)
    torch.cuda.set_per_process_memory_fraction(.25,0)
    model,preprocess=clip.load(str(args.clip_weights),device='cuda',jit=False)
    model.eval().requires_grad_(False);records=[]
    with torch.inference_mode():
        for i,score in enumerate(report['best_val']['images']):
            r=pairs[score['sample_id']];images=[]
            for role in ('reference','generated','target'):
                name=f'{role}/sample_{i:04d}.png';path=args.run/'images/val_best'/name
                if sha(path)!=manifest['files'][name]['sha256']:raise ValueError('PNG bytes changed')
                images.append(preprocess(Image.open(path).convert('RGB')))
            visual=F.normalize(model.encode_image(torch.stack(images).cuda()).float(),dim=-1)
            source=caption(descriptions[r['source']['sample_id']],r['source_scene'])
            target=caption(descriptions[r['target']['sample_id']],r['target_scene'])
            texts=[source,target];lengths=[len(clip.clip._tokenizer.encode(s))+2 for s in texts]
            semantic=F.normalize(model.encode_text(clip.tokenize(texts,truncate=True).cuda()).float(),dim=-1)
            di=visual[1]-visual[0];dt=semantic[1]-semantic[0]
            valid=float(di.norm())>1e-6 and float(dt.norm())>1e-6
            direction=float(F.cosine_similarity(di[None],dt[None])) if valid else None
            records.append({**score,'source_caption':source,'target_caption':target,
                'clip_source_text_tokens':lengths[0],'clip_target_text_tokens':lengths[1],
                'clip_text_truncated':max(lengths)>77,
                'clip_target_cosine':float(visual[1]@semantic[1]),
                'clip_direction_cosine':direction,'clip_direction_valid':valid,
                'clip_gt_target_cosine':float(visual[2]@semantic[1]),
                'clip_reference_target_cosine':float(visual[0]@semantic[1]),
                'clip_output_gt_image_cosine':float(visual[1]@visual[2])})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'sample_metrics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    with (args.output/'sample_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    groups=collections.defaultdict(list)
    for r in records:
        for key in ('all',r['mode'],r['category']+'/'+r['mode']):groups[key].append(r)
    keys=('psnr','lpips','object_crop_lpips','edited_psnr','retained_psnr','clip_target_cosine','clip_direction_cosine','clip_output_gt_image_cosine')
    summary={g:{'n':len(rows),**{k:float(np.mean(v)) if (v:=[r[k] for r in rows if r.get(k) is not None]) else None for k in keys}} for g,rows in groups.items()}
    payload={'complete':True,'scope':'Offline development VAL editing evaluation; semantic captions are scoring metadata, never generator inputs',
        'checkpoint_sha256':report['checkpoint_sha256'],'report_sha256':sha(args.run/'report.json'),
        'clip_upstream_sha':vendor_sha,'clip_weight_sha256':sha(args.clip_weights),'clip_model':'OpenAI ViT-B/32',
        'source_sha256':sha(Path(__file__)),'groups':summary,
        'protocol':'Native 256 PNG -> official CLIP preprocess. L2-normalized embeddings; raw cosine in [-1,1], not rescaled CLIPScore. Direction = cosine(output-reference, target_caption-source_caption). All rows included; degenerate direction reports null.',
        'limitations':'CLIP is imperfect for exact color/count/geometry. Caption-matched GT and unedited-source controls are diagnostics, not model outputs. LPIPS was also a training loss. Small 4-product VAL is not independent TEST. No FID/KID claimed.'}
    (args.output/'report.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
