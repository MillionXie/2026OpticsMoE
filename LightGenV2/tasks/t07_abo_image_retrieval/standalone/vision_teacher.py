"""TRAIN-only merged visual-token teacher cache; never part of student inference.

Only build() imports a Qwen vision Transformer. Loading/loss/shape checks use
tensors, and add no student modules. Fixed clean contain_white224 images match
the student coordinate system; do not pair these tokens with augmented crops.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from .io import sha256, source_commit, write_json, picture, inputs


PREPROCESSING = 'RGB without EXIF transpose; contain_white224; clean view; normalized merged7x7 tokens'


def processor_identity(directory):
    directory=Path(directory)
    rows=[(p.relative_to(directory).as_posix(),sha256(p)) for p in sorted(directory.rglob('*')) if p.is_file()]
    if not rows:raise ValueError('Empty processor directory')
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def validate_cache(cache, samples, target_sha, pool_sha, processor_sha, image_hashes):
    """Require the full ordered training pool, not a smoke subset or test cache."""
    if (cache.get('schema')!=1 or cache.get('kind')!='vision_patch_teacher_cache' or
        cache.get('frozen_teacher') is not True or cache.get('teacher_trainable_parameters')!=0 or
        cache.get('student_inference_teacher_required') is not False or
        cache.get('complete_training_pool') is not True or
        cache.get('excluded_splits')!=['val','test'] or cache.get('preprocessing')!=PREPROCESSING):
        raise ValueError('Invalid training-only visual teacher contract')
    if (not samples or any(s.split!='train' for s in samples) or
        len({s.sample_id for s in samples})!=len(samples) or
        cache.get('ids')!=[s.sample_id for s in samples] or
        cache.get('target_manifest_sha256')!=target_sha or cache.get('pool_manifest_sha256')!=pool_sha or
        cache.get('processor_sha256')!=processor_sha or cache.get('image_sha256')!=image_hashes):
        raise ValueError('Visual teacher training identity changed')
    x=cache.get('features')
    if not isinstance(x,torch.Tensor) or x.shape!=(len(samples),49,2048) or x.dtype!=torch.float16:
        raise ValueError('Expected FP16 [training_images,49,2048] visual targets')
    # Chunked validation avoids expanding the whole ~1GiB cache to FP32.
    for part in x.split(64):
        if not torch.isfinite(part).all() or not torch.allclose(part.float().norm(dim=-1),torch.ones(part.shape[:2]),atol=.002,rtol=0):
            raise ValueError('Visual targets must be finite unit vectors')
    return x


def load_cache(path, samples, target, pool, assets, device):
    before=sha256(path);cache=torch.load(path,map_location='cpu',weights_only=True)
    x=validate_cache(cache,samples,sha256(target/'data/abo_similarity10_manifest.csv'),
                     sha256(pool/'manifest.csv'),processor_identity(assets/'processor'),
                     [sha256(s.image_path) for s in samples])
    if sha256(path)!=before:raise ValueError('Visual cache changed while loading')
    audit={k:v for k,v in cache.items() if k not in ('features','ids','image_sha256')}
    audit.update(cache_sha256=before,training_images=len(samples),storage_bytes=x.numel()*x.element_size())
    return x.to(device),audit


def merged_vision(model, batch):
    """Extra TRAIN forward through existing V path only, no new model parameters."""
    count=len(batch['input_ids']);grid=batch['image_grid_thw']
    if model.metadata.get('input_preprocessing')!='contain_white' or not torch.equal(grid,grid.new_tensor([[1,14,14]]).expand(count,-1)):
        raise ValueError('Visual distillation requires the fixed clean224 contract')
    patches=model.frontend.patches(batch['pixel_values'],count)
    return model.frontend.merge(model.vision(patches)).reshape(count,49,2048)


def patch_cosine_loss(student, teacher):
    if student.ndim!=3 or student.shape[1:]!=(49,2048) or student.shape!=teacher.shape:
        raise ValueError('Visual teacher/student token shapes differ')
    return (1-F.cosine_similarity(student.float(),teacher.detach().float(),dim=-1)).mean()


def build(args):
    """Standalone teacher-vision extraction, no language decoder weights loaded."""
    from safetensors import safe_open
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from .data import _load_contract
    from .domain_data import combine_training
    from .broad_transfer import load_pool
    if args.output.exists():raise FileExistsError(f'Refusing to overwrite {args.output}')
    if args.max_images is not None and args.max_images<1:raise ValueError('Positive smoke image limit required')
    if args.batch_size<1:raise ValueError('Positive batch size required')
    torch.set_num_threads(4);device=torch.device(args.device)
    original,_=_load_contract(args.target);external,_=load_pool(args.pool,args.abo,args.target)
    samples,_=combine_training(original,external);full_count=len(samples)
    if args.max_images is not None:samples=samples[:args.max_images]
    hashes=[sha256(s.image_path) for s in samples];processor_sha=processor_identity(args.assets/'processor')
    weights=args.model/'model.safetensors';config_file=args.model/'config.json'
    weight_sha=sha256(weights);config_sha=sha256(config_file)
    config=json.loads(config_file.read_text())['vision_config']
    required=dict(depth=24,hidden_size=1024,out_hidden_size=2048,patch_size=16,spatial_merge_size=2,temporal_patch_size=2)
    if any(config.get(k)!=v for k,v in required.items()):raise ValueError('Pinned Qwen vision geometry changed')
    cfg=Qwen3VLVisionConfig(**config);cfg._attn_implementation='sdpa'
    teacher=Qwen3VLVisionModel(cfg).to(torch.bfloat16).eval().requires_grad_(False)
    with safe_open(weights,framework='pt',device='cpu') as source:
        state={k[len('model.visual.'):]:source.get_tensor(k) for k in source.keys() if k.startswith('model.visual.')}
    teacher.load_state_dict(state,strict=True);del state;teacher.to(device)
    processor=AutoProcessor.from_pretrained(str(args.assets/'processor'),local_files_only=True)
    args.output.mkdir(parents=True,exist_ok=False)
    features=[]
    with torch.inference_mode():
        for start in range(0,len(samples),args.batch_size):
            chunk=samples[start:start+args.batch_size]
            batch=inputs(processor,[picture(s.image_path,'contain_white') for s in chunk],device)
            expected=batch['image_grid_thw'].new_tensor([[1,14,14]]).expand(len(chunk),-1)
            if not torch.equal(batch['image_grid_thw'],expected):raise ValueError('Teacher processor grid changed')
            merged,_=teacher(batch['pixel_values'].to(torch.bfloat16),batch['image_grid_thw'])
            if merged.shape!=(len(chunk)*49,2048) or not torch.isfinite(merged).all():raise ValueError('Invalid teacher visual output')
            features.append(F.normalize(merged.float(),dim=-1).reshape(len(chunk),49,2048).cpu().half())
            if (start//args.batch_size+1)%30==0:print(f'train-only visual teacher {min(start+args.batch_size,len(samples))}/{len(samples)}',flush=True)
    if hashes!=[sha256(s.image_path) for s in samples] or processor_identity(args.assets/'processor')!=processor_sha or sha256(weights)!=weight_sha or sha256(config_file)!=config_sha:
        raise ValueError('Teacher inputs changed during cache extraction')
    trainable=sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    if trainable:raise RuntimeError('Visual teacher must remain frozen')
    cache=dict(schema=1,kind='vision_patch_teacher_cache',frozen_teacher=True,teacher_trainable_parameters=trainable,
        teacher_visual_parameters=sum(p.numel() for p in teacher.parameters()),student_inference_teacher_required=False,
        complete_training_pool=len(samples)==full_count,full_training_pool_images=full_count,excluded_splits=['val','test'],
        preprocessing=PREPROCESSING,source_commit=source_commit(),model=str(args.model.resolve()),model_sha256=weight_sha,
        config_sha256=config_sha,processor_sha256=processor_sha,target_manifest_sha256=sha256(args.target/'data/abo_similarity10_manifest.csv'),
        pool_manifest_sha256=sha256(args.pool/'manifest.csv'),ids=[s.sample_id for s in samples],image_sha256=hashes,features=torch.cat(features))
    torch.save(cache,args.output/'cache.pt')
    report={k:v for k,v in cache.items() if k not in ('features','ids','image_sha256')}
    report.update(status='complete',command=sys.argv,pid=os.getpid(),device=str(device),torch=torch.__version__,
        cache_sha256=sha256(args.output/'cache.pt'),training_images=len(samples),storage_bytes=cache['features'].numel()*2,
        teacher_in_student_inference=False)
    write_json(args.output/'final_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    del teacher
    if device.type=='cuda':torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('target','pool','abo','model','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--max-images',type=int,help='Smoke only: incomplete caches are rejected by training loader')
    build(p.parse_args())


if __name__=='__main__':main()
