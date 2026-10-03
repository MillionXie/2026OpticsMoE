"""Cache the aligned teacher decoder input for TRAIN images only (~0.75 GB)."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .aligned_baseline import AlignedReadout
from .modeling import load_vision_backbone, sha256_file
from .settings import load_settings
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import (
    prepare_salicon,SALICONSaliencyDataset,collate_salicon,_annotation_path)
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.training import preprocess_vision
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.partial').exists():raise FileExistsError(args.output)
    if args.batch_size < 1:raise ValueError('Positive batch size required')
    s=load_settings(args.config)
    if sha256_file(args.checkpoint)!=s.distillation_teacher_sha256:raise ValueError('Teacher SHA mismatch')
    bundle=prepare_salicon(s,persist=False);records=bundle.train_records
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded=load_vision_backbone(s,device);s.resolve_architecture(loaded.model)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if payload['architecture']!='frozen_qwen24_adapter192_identical_progressive_decoder_v1':raise ValueError('Wrong teacher')
    loaded.model.requires_grad_(False).eval()
    head=AlignedReadout(s.vision_hidden_size).to(device);head.load_state_dict(payload['head'],strict=True)
    model=FrozenQwenVisionTeacher(loaded,head).eval()
    captured=[]
    hook=head.decoder.register_forward_pre_hook(lambda _module,inputs:captured.append(inputs[0].detach()))
    loader=DataLoader(SALICONSaliencyDataset(records,s,training=False),batch_size=args.batch_size,
                      shuffle=False,num_workers=s.num_workers,collate_fn=collate_salicon)
    values=torch.empty(len(records),192,14,14,dtype=torch.float16);ids=[];offset=0
    try:
        for batch in loader:
            captured.clear();inputs=preprocess_vision(loaded.processor,batch['images'],device)
            model(inputs['pixel_values'],inputs['image_grid_thw'])
            n=len(batch['sample_ids'])
            if len(captured)!=1 or captured[0].shape!=(n,192,14,14):raise ValueError('Unexpected teacher feature hook')
            values[offset:offset+n].copy_(captured[0].cpu().half());offset+=n;ids.extend(batch['sample_ids'])
            if offset % 512 < n or offset==len(records):print(f'[teacher train features] {offset}/{len(records)}',flush=True)
    finally:
        hook.remove();model.close()
    if ids!=[r.sample_id for r in records] or any(not k.startswith('train/') for k in ids):raise ValueError('Invalid identities')
    if not torch.isfinite(values).all():raise ValueError('Nonfinite features')
    manifest={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'checkpoint_sha256':sha256_file(args.checkpoint),'checkpoint':str(args.checkpoint.resolve()),
        'feature_contract':'aligned_decoder_input_192x14x14_row_major_v1','split':'train_only',
        'samples':len(ids),'augmentation':False,'image_size':s.image_size,'dtype':'float16',
        'teacher_selected_on_public_test':True,'student_inference_requires_teacher':False,
        'image_manifest_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'train_annotations_sha256':sha256_file(_annotation_path(s.data_root,'train'))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.partial')
    torch.save({'manifest':manifest,'sample_ids':ids,'features':values},temporary);temporary.replace(args.output)
    manifest['cache_sha256']=sha256_file(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
