"""Training-only frozen-Qwen relations. Student inference never imports a Qwen model.

The CLI builds a train-only native-aspect teacher cache in a separate GPU process.
Loader/loss use tensors only; no teacher model or new projection in student training.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from .data import _load_contract
from .domain_data import combine_training
from .io import sha256, write_json, source_commit


def load_teacher_cache(path, samples, target, pool, device):
    cache=torch.load(path,map_location='cpu',weights_only=True)
    if cache.get('schema')!=1 or cache.get('frozen_teacher') is not True:
        raise ValueError('Not a frozen training-only teacher cache')
    if cache['target_manifest_sha256']!=sha256(target/'data/abo_similarity10_manifest.csv'):
        raise ValueError('Teacher target dataset changed')
    if cache['pool_manifest_sha256']!=sha256(pool/'manifest.csv'):
        raise ValueError('Teacher external pool changed')
    if cache['ids']!=[s.sample_id for s in samples] or any(s.split!='train' for s in samples):
        raise ValueError('Teacher IDs must match ordered training-only samples')
    if cache.get('image_sha256')!=[sha256(s.image_path) for s in samples]:
        raise ValueError('Teacher input image bytes changed')
    vectors=cache['vectors'].float()
    if vectors.shape!=(len(samples),2048) or not torch.isfinite(vectors).all() or not (vectors.norm(dim=-1)>0).all():
        raise ValueError('Invalid teacher vectors')
    audit={k:v for k,v in cache.items() if k not in ('vectors','ids','image_sha256')}
    audit.update(cache_sha256=sha256(path),training_images=len(samples))
    return F.normalize(vectors,dim=-1).to(device),audit


def gallery_relation_loss(query, own, labels, bank, bank_labels, teacher_query, teacher_bank, temperature=.10, teacher_temperature=None):
    """Match distributions over other TRAIN products, only when teacher top1 is correct.

    Bases/dimensions may differ (student64, teacher2048); compare similarities,
    not coordinate vectors. Never train on test predictions or restrict test gallery.
    """
    teacher_temperature=temperature if teacher_temperature is None else teacher_temperature
    if not all(math.isfinite(t) and t>0 for t in (temperature,teacher_temperature)):
        raise ValueError('Positive finite distillation temperatures required')
    if len(bank)!=len(teacher_bank) or len(query)!=len(teacher_query):raise ValueError('Unaligned relation banks')
    student=F.normalize(query.float(),dim=-1)@F.normalize(bank.detach().float(),dim=-1).T/temperature
    with torch.no_grad():
        teacher=F.normalize(teacher_query.detach().float(),dim=-1)@F.normalize(teacher_bank.detach().float(),dim=-1).T/teacher_temperature
        valid=torch.arange(len(bank),device=query.device)[None]!=own[:,None]
        target=teacher.masked_fill(~valid,-1e4).softmax(-1)
        correct=bank_labels[target.argmax(-1)].eq(labels)
        positive=bank_labels[None].eq(labels[:,None])&valid
        confidence=(target.mul(positive).sum(-1)-positive.float().sum(-1)/valid.sum(-1)).clamp_min(0)*correct
    divergence=F.kl_div(student.masked_fill(~valid,-1e4).log_softmax(-1),target,reduction='none').sum(-1)
    loss=(divergence*confidence).sum()/confidence.sum().clamp_min(1.)
    return loss,dict(teacher_correct_fraction=correct.float().mean().detach(),teacher_confidence=confidence.mean().detach())


def build(args):
    # Full model dependencies are intentionally imported ONLY in the builder CLI.
    from ..legacy_run import load_settings, load_backbone, _inputs, move_inputs, teacher_embeddings, INSTRUCTION
    from .broad_transfer import load_pool
    from PIL import Image,ImageOps
    if args.output.exists():raise FileExistsError(args.output)
    target,_=_load_contract(args.target)
    external,pool_report=load_pool(args.pool,args.abo,args.target)
    if not pool_report.get('target_types_only'):raise ValueError('Teacher requires target-mapped training pool')
    samples,_=combine_training(target,external)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    settings=load_settings(args.config);settings.model_id=str(args.model.resolve());settings.instruction=INSTRUCTION
    if settings.processor_min_pixels!=50176 or settings.processor_max_pixels!=50176:
        raise ValueError('Teacher processor budget must remain 50176 pixels')
    loaded=load_backbone(settings,torch.device('cuda'))
    loaded.model.eval().requires_grad_(False)
    vectors=[]
    with torch.inference_mode():
        for i,sample in enumerate(samples):
            with Image.open(sample.image_path) as source:picture=ImageOps.exif_transpose(source).convert('RGB')
            batch=move_inputs(_inputs(loaded.processor,picture),loaded.device)
            vectors.append(teacher_embeddings(loaded.model,batch,2048)[0].cpu().to(torch.float16))
            if (i+1)%120==0:print(f'train-only teacher {i+1}/{len(samples)}',flush=True)
    cache=dict(schema=1,frozen_teacher=True,teacher_trainable_parameters=sum(p.numel() for p in loaded.model.parameters() if p.requires_grad),
               source_commit=source_commit(),model=str(args.model.resolve()),prompt=INSTRUCTION,
               target_manifest_sha256=sha256(args.target/'data/abo_similarity10_manifest.csv'),
               pool_manifest_sha256=sha256(args.pool/'manifest.csv'),ids=[s.sample_id for s in samples],vectors=torch.stack(vectors),
               image_sha256=[sha256(s.image_path) for s in samples],
               preprocessing='EXIF RGB; native aspect; processor min=max pixels 50176',
               excluded_splits=['val','test'],student_inference_teacher_required=False)
    torch.save(cache,args.output/'cache.pt')
    write_json(args.output/'final_report.json',dict(status='complete',kind='train_only_teacher_cache',source_commit=source_commit(),
        command=sys.argv,training_images=len(samples),training_products=len({s.product_id for s in samples}),
        teacher_trainable_parameters=cache['teacher_trainable_parameters'],cache_sha256=sha256(args.output/'cache.pt'),
        target_manifest_sha256=cache['target_manifest_sha256'],pool_manifest_sha256=cache['pool_manifest_sha256'],
        excluded_splits=cache['excluded_splits'],gpu=torch.cuda.get_device_name(),model=cache['model'],python=sys.version,torch=torch.__version__))
    del loaded
    torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--abo',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/optical_top2_dc20.yaml')
    p.add_argument('--output',type=Path,required=True)
    build(p.parse_args())


if __name__=='__main__':main()
