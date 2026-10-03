"""Export actual teacher predictions on ONE precisely specified training view.

This is a training target cache, never part of optical student inference.
Original images, annotations, checkpoints and existing caches are read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import PIL
from PIL import Image

SIZE = 224
BOX = (6, 6, 218, 218)
VIEW = {
    'contract': 'salicon_rgb224_center212_bicubic224_v1',
    'input_mode': 'RGB', 'input_size_wh': [224, 224], 'crop_box_ltrb': list(BOX),
    'output_size_wh': [224, 224], 'rgb_resize': 'PIL.Resampling.BICUBIC',
    'teacher_target': 'teacher(cropped_resized_RGB), NOT warp(teacher(original_RGB))',
}


def rgb_sha256(image):
    if image.mode != 'RGB' or image.size != (SIZE, SIZE):
        raise ValueError('Fixed crop expects the exact preprocessed RGB224 image')
    return hashlib.sha256(b'RGB224x224\0' + image.tobytes()).hexdigest()


def crop_image(image):
    rgb_sha256(image)  # Reject silent extra resizing/conversion.
    return image.crop(BOX).resize((SIZE, SIZE), Image.Resampling.BICUBIC)


def validate_payload(payload, records, teacher_sha, annotations_sha):
    ids = [r.sample_id for r in records]
    manifest = payload['manifest']
    if (not ids or payload['sample_ids'] != ids or len(set(ids)) != len(ids)
            or any(not sid.startswith('train/') for sid in ids)
            or manifest.get('samples') != len(ids) or manifest.get('split') != 'train_only'
            or manifest.get('view') != VIEW or manifest.get('checkpoint_sha256') != teacher_sha
            or manifest.get('train_annotations_sha256') != annotations_sha
            or manifest.get('image_manifest_sha256') != hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()):
        raise ValueError('Fixed crop teacher identity/view contract mismatch')
    values = payload['logits']
    if (not isinstance(values, torch.Tensor) or values.shape != (len(ids), 1, SIZE, SIZE)
            or values.dtype != torch.float16 or values.device.type != 'cpu'
            or any(not torch.isfinite(v).all() for v in values.split(32))):
        raise ValueError('Invalid fixed crop teacher logits')
    for key in ('original_rgb_sha256', 'crop_rgb_sha256'):
        digests = payload[key]
        if (len(digests) != len(ids) or any(not isinstance(h, str) or len(h) != 64
                or any(c not in '0123456789abcdef' for c in h) for h in digests)):
            raise ValueError('Invalid fixed crop pixel fingerprint list')


def check_pixels(payload, index, original, cropped):
    """Consumer-side guard: the future student must see the teacher's exact RGB."""
    if (rgb_sha256(original) != payload['original_rgb_sha256'][index]
            or rgb_sha256(cropped) != payload['crop_rgb_sha256'][index]):
        raise ValueError('Fixed crop teacher/student RGB pixels differ')


def save_new_payload(payload, output):
    """No overwrite, including a concurrently created target; retain failed partials."""
    output = Path(output)
    temporary = output.with_suffix('.partial')
    sidecar = output.with_suffix('.json')
    if output.suffix != '.pt' or any(p.exists() for p in (output, temporary, sidecar)):
        raise FileExistsError('Use a new .pt output; preserve any existing/partial cache')
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as handle:
        torch.save(payload, handle)
    # Link is atomic and refuses an existing destination. Both are on the same FS.
    os.link(temporary, output)
    temporary.unlink()  # Only our temporary name; the complete bytes remain at output.
    from .modeling import sha256_file
    report = dict(payload['manifest'], cache_sha256=sha256_file(output), cache_bytes=output.stat().st_size)
    with sidecar.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    return report


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Positive batch size required')
    output = args.output.resolve()
    task_runs = (Path(__file__).resolve().parent/'runs/simulation').resolve()
    if not output.is_relative_to(task_runs):
        raise ValueError('Cache must remain inside T03 runs/simulation')
    if output.suffix != '.pt' or any(p.exists() for p in (output, output.with_suffix('.partial'), output.with_suffix('.json'))):
        raise FileExistsError('Refusing existing/partial cache output')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; no silent CPU fallback')
    from torch.utils.data import DataLoader
    from .settings import load_settings
    from .run import _seed
    from .modeling import load_vision_backbone, sha256_file
    from .aligned_baseline import AlignedReadout
    from .recheck_aligned import load_hashed_checkpoint
    from .training_support import use_spawn_workers
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import (
        prepare_salicon, SALICONSaliencyDataset, collate_salicon, _annotation_path)
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import preprocess_vision
    from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher
    _seed(42)
    s = load_settings(args.config)
    s.num_workers = 2
    checkpoint, digest = load_hashed_checkpoint(args.checkpoint)
    if (digest != s.distillation_teacher_sha256 or checkpoint.get('architecture')
            != 'frozen_qwen24_adapter192_identical_progressive_decoder_v1'):
        raise ValueError('Wrong aligned teacher checkpoint')
    bundle = prepare_salicon(s, persist=False)
    records = bundle.train_records
    if len(records) != 10000 or s.image_size != SIZE:
        raise ValueError('Require full original 10000-image training split and 224 input')
    annotations_sha = sha256_file(_annotation_path(s.data_root, 'train'))
    device = torch.device(args.device)
    loaded = load_vision_backbone(s, device)
    s.resolve_architecture(loaded.model)
    loaded.model.requires_grad_(False).eval()
    head = AlignedReadout(s.vision_hidden_size).to(device)
    head.load_state_dict(checkpoint['head'], strict=True)
    model = FrozenQwenVisionTeacher(loaded, head).eval()
    loader = DataLoader(SALICONSaliencyDataset(records, s, training=False), batch_size=args.batch_size,
                        shuffle=False, num_workers=2, collate_fn=collate_salicon)
    use_spawn_workers(loader)
    values = torch.empty(len(records), 1, SIZE, SIZE, dtype=torch.float16)
    ids, original_hashes, crop_hashes = [], [], []
    offset = 0
    try:
        for batch in loader:
            images = batch['images']
            crops = [crop_image(im) for im in images]
            inputs = preprocess_vision(loaded.processor, crops, device)
            logits = model(inputs['pixel_values'], inputs['image_grid_thw'])[0]
            n = len(images)
            if logits.shape != (n, 1, SIZE, SIZE):
                raise ValueError('Unexpected teacher output shape')
            values[offset:offset+n].copy_(logits.cpu().half())
            offset += n
            ids.extend(batch['sample_ids'])
            original_hashes.extend(map(rgb_sha256, images))
            crop_hashes.extend(map(rgb_sha256, crops))
            if offset % 512 < n or offset == len(records):
                print(f'[actual fixed-crop teacher] {offset}/{len(records)}', flush=True)
    finally:
        model.close()
    manifest = dict(schema_version=1, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=[sys.executable, '-m', __spec__.name, *sys.argv[1:]], checkpoint_sha256=digest,
        config_sha256=sha256_file(args.config),
        train_annotations_sha256=annotations_sha, split='train_only', samples=len(ids), view=VIEW,
        image_manifest_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
        dtype='float16', inference_precision='native frozen Qwen dtype, fp32 decoder, no extra autocast',
        torch_version=torch.__version__, pillow_version=PIL.__version__,
        teacher_selected_on_public_test=True, student_inference_requires_teacher=False,
        rgb_fingerprint='SHA256(b"RGB224x224\\0" + RGB.tobytes()), ordered one per sample')
    payload = dict(manifest=manifest, sample_ids=ids, logits=values,
                   original_rgb_sha256=original_hashes, crop_rgb_sha256=crop_hashes)
    validate_payload(payload, records, digest, annotations_sha)
    print(json.dumps(save_new_payload(payload, output), indent=2), flush=True)


if __name__ == '__main__':
    main()
