"""Offline teacher predictions for a SHA-pinned, SALICON-disjoint COCO pool."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader

from .aligned_baseline import AlignedReadout
from .modeling import load_vision_backbone, sha256_file
from .settings import load_settings
from .unlabeled_data import AuditedUnlabeledImages, collate_unlabeled, PREPROCESSING
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.training import preprocess_vision
from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher


def checked_cpu_logits(logits, count):
    if logits.shape != (count,1,224,224) or not torch.isfinite(logits).all():
        raise ValueError('Unexpected/nonfinite teacher output')
    value = logits.detach().cpu().half()
    if not torch.isfinite(value).all():
        raise ValueError('Teacher output overflows float16 cache')
    return value


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--image-manifest', type=Path, required=True)
    parser.add_argument('--image-manifest-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    temporary = args.output.with_suffix('.partial')
    sidecar = args.output.with_suffix('.json')
    if args.output.suffix != '.pt' or any(p.exists() for p in (args.output, temporary, sidecar)):
        raise FileExistsError('Use a new .pt destination without prior partial/sidecar')
    if args.batch_size < 1:
        raise ValueError('Positive batch size required')
    s = load_settings(args.config)
    if s.image_size != 224 or sha256_file(args.checkpoint) != s.distillation_teacher_sha256:
        raise ValueError('Teacher SHA or image size mismatch')
    dataset = AuditedUnlabeledImages(args.image_manifest, args.image_manifest_sha256, s.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=s.num_workers,
                        collate_fn=collate_unlabeled)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload['architecture'] != 'frozen_qwen24_adapter192_identical_progressive_decoder_v1':
        raise ValueError('Wrong teacher architecture')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded = load_vision_backbone(s, device)
    s.resolve_architecture(loaded.model)
    loaded.model.requires_grad_(False).eval()
    head = AlignedReadout(s.vision_hidden_size).to(device)
    head.load_state_dict(payload['head'], strict=True)
    model = FrozenQwenVisionTeacher(loaded, head).eval()
    maps = torch.empty(len(dataset),1,224,224,dtype=torch.float16)
    ids, offset = [], 0
    try:
        for batch in loader:
            inputs = preprocess_vision(loaded.processor, batch['images'], device)
            logits = model(inputs['pixel_values'],inputs['image_grid_thw'])[0]
            count = len(batch['sample_ids'])
            maps[offset:offset+count].copy_(checked_cpu_logits(logits,count))
            ids.extend(batch['sample_ids'])
            offset += count
            if offset % 512 < count or offset == len(dataset):
                print(f'[unlabeled teacher] {offset}/{len(dataset)}',flush=True)
    finally:
        model.close()
    if ids != dataset.sample_ids or offset != len(dataset):
        raise ValueError('Teacher export identity/completeness mismatch')
    manifest = {
        'schema_version': 1, 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha256_file(args.checkpoint),
        'image_manifest': str(dataset.manifest_path), 'image_manifest_sha256': dataset.manifest_sha256,
        'split': 'extra_unlabeled_excluding_salicon_train_and_test', 'samples': len(ids),
        'preprocessing': PREPROCESSING, 'image_size': 224, 'augmentation': False,
        'ground_truth_available': False, 'teacher_selected_on_public_test': True,
        'student_inference_requires_teacher': False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('xb') as handle:
        torch.save({'manifest':manifest,'sample_ids':ids,'logits':maps},handle)
    # Atomic no-overwrite publication; retain partial if publication fails.
    os.link(temporary,args.output)
    temporary.unlink()
    manifest['cache_sha256'] = sha256_file(args.output)
    with sidecar.open('x',encoding='utf-8') as handle:
        json.dump(manifest,handle,indent=2)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
