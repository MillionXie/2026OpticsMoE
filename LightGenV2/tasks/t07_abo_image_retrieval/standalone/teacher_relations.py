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


def load_feature_alignment(path, expected_sha256, sample_ids, teacher_cache_sha256,
                           source_checkpoint_sha256, dimension=64):
    """Reuse a pinned TRAIN teacher basis, not rotate an already trained student."""
    if not expected_sha256 or sha256(path) != expected_sha256:
        raise ValueError('Teacher alignment file SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('teacher_center_fraction',0) != 0 or 'teacher_center' in payload:
        raise ValueError('Centered teacher targets cannot be reused as a raw teacher basis')
    if (payload.get('teacher_only') is not True or
        payload.get('fit_sample_ids') != sample_ids or
        payload.get('teacher_cache_sha256') != teacher_cache_sha256 or
        payload.get('source_checkpoint_sha256') != source_checkpoint_sha256 or
        payload.get('teacher_prefix_dimensions') != dimension):
        raise ValueError('Teacher alignment training identity changed')
    rotation = payload.get('rotation')
    if (not isinstance(rotation, torch.Tensor) or rotation.shape != (dimension, dimension)
        or not torch.isfinite(rotation).all()):
        raise ValueError('Invalid teacher alignment matrix')
    rotation = rotation.detach().float()
    if not torch.allclose(rotation.T@rotation, torch.eye(dimension), atol=1e-5, rtol=1e-5):
        raise ValueError('Teacher alignment must remain orthogonal')
    if sha256(path) != expected_sha256:
        raise ValueError('Teacher alignment changed while loading')
    return dict(payload, rotation=rotation)


@torch.no_grad()
def center_teacher_prefix(prefix, original_train_count, fraction):
    """Remove a fixed fraction of the ORIGINAL TRAIN mean from unit teacher rows.

    All rows must be from the separately validated train-only cache. The first
    original_train_count rows are the original training split, not external,
    validation or test rows. Nothing is fitted to or stored in the student.
    """
    if (prefix.ndim!=2 or type(original_train_count) is not int or
        not 1<=original_train_count<=len(prefix) or isinstance(fraction,bool) or
        not math.isfinite(fraction) or not 0<=fraction<=1):
        raise ValueError('Invalid training teacher centering configuration')
    values=prefix.detach().float()
    if not torch.isfinite(values).all() or not torch.allclose(values.norm(dim=-1),torch.ones(len(values),device=values.device),atol=1e-5,rtol=1e-5):
        raise ValueError('Teacher centering requires finite unit prefix rows')
    center=values[:original_train_count].mean(0)
    if fraction==0:return values,center
    shifted=values-float(fraction)*center
    if bool((shifted.norm(dim=-1)<1e-6).any()):
        raise ValueError('Teacher centering produced a degenerate target')
    return F.normalize(shifted,dim=-1),center


@torch.no_grad()
def fit_feature_alignment(teacher, student):
    """Rotate TRAIN teacher coordinates into the existing student basis.

    Does not alter student weights or inference. Caller supplies only aligned
    original training rows, never test rows. CPU float64 SVD avoids autocast.
    """
    if teacher.shape != student.shape or teacher.ndim != 2 or len(teacher) < teacher.shape[1]:
        raise ValueError('Alignment requires matched training matrices with enough rows')
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise ValueError('Nonfinite alignment feature')
    a = F.normalize(teacher.detach().cpu().double(), dim=-1)
    b = F.normalize(student.detach().cpu().double(), dim=-1)
    u, _, vh = torch.linalg.svd(a.T @ b, full_matrices=False)
    return (u @ vh).float().to(teacher.device)


def aligned_feature_loss(query, targets, own, labels, teacher_query, teacher_bank, bank_labels):
    """Cosine supervision gated by full teacher's leave-own-product correctness."""
    if query.shape != targets.shape:
        raise ValueError('Aligned teacher/student feature dimensions differ')
    with torch.no_grad():
        scores = F.normalize(teacher_query.detach().float(), dim=-1) @ F.normalize(teacher_bank.detach().float(), dim=-1).T
        scores[torch.arange(len(query), device=query.device), own] = -torch.inf
        correct = bank_labels[scores.argmax(1)].eq(labels)
    per_image = 1-F.cosine_similarity(query.float(), targets.detach().float(), dim=-1)
    loss = (per_image*correct).sum()/correct.sum().clamp_min(1)
    return loss, correct.float().mean().detach()


@torch.no_grad()
def select_agreeing_external(samples, vectors, target_count):
    """Training-only curriculum, NOT label correction or deletion of source data.

    Keep every original training image. For external images, require the frozen
    teacher's nearest OTHER training product to have the supplied category.
    Fit this selection once against the original full train bank, before filtering.
    """
    from .retrieval_training import product_bank
    if not 0 < target_count < len(samples) or len(vectors) != len(samples):
        raise ValueError('Expected aligned original + external training samples')
    if any(s.split != 'train' for s in samples):
        raise ValueError('Teacher selection accepts training samples only')
    # Fix the selector to CPU float32, independent of training GPU/autocast.
    teacher_cpu = vectors.detach().float().cpu()
    bank, bank_labels, own = product_bank(teacher_cpu, samples)
    query = F.normalize(teacher_cpu, dim=-1)
    predictions = []
    for start in range(0, len(samples), 256):
        scores = query[start:start+256] @ bank.T
        scores[torch.arange(len(scores), device=scores.device), own[start:start+256]] = -torch.inf
        predictions.extend(bank_labels[scores.argmax(1)].cpu().tolist())
    kept, rows = [], []
    for i, (s, prediction) in enumerate(zip(samples, predictions)):
        keep = i < target_count or prediction == s.category_id
        if keep: kept.append(i)
        rows.append(dict(sample_id=s.sample_id, product_id=s.product_id,
                         category_id=s.category_id, teacher_category_id=prediction,
                         original_train=i < target_count, kept=keep))
    selected = [samples[i] for i in kept]
    # Do not accidentally turn this into removing difficult task categories.
    external_categories = {s.category_id for s in samples[target_count:]}
    if {s.category_id for s in selected[target_count:]} != external_categories:
        raise ValueError('Teacher selection removed an entire external category')
    return selected, vectors[torch.tensor(kept, device=vectors.device)], rows


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


def reusable_training_vectors(path, expected_sha256, samples, target_sha256, model, prompt):
    """Reuse only image-verified intersections of a pinned train-only cache.

    A larger pool may choose different views of the same products. Missing old
    rows are reported, not appended into the new training manifest. No GPU use.
    """
    if not expected_sha256 or sha256(path)!=expected_sha256:
        raise ValueError('Reuse teacher cache SHA mismatch')
    cache=torch.load(path,map_location='cpu',weights_only=True)
    if (cache.get('schema')!=1 or cache.get('frozen_teacher') is not True or
        cache.get('teacher_trainable_parameters')!=0 or
        cache.get('target_manifest_sha256')!=target_sha256 or
        cache.get('model')!=model or cache.get('prompt')!=prompt or
        cache.get('preprocessing')!='EXIF RGB; native aspect; processor min=max pixels 50176' or
        cache.get('excluded_splits')!=['val','test'] or
        cache.get('student_inference_teacher_required') is not False):
        raise ValueError('Reuse teacher model/preprocessing/training identity changed')
    ids=cache.get('ids',[]);hashes=cache.get('image_sha256',[]);vectors=cache.get('vectors')
    if (not ids or len(set(ids))!=len(ids) or len(hashes)!=len(ids) or
        not isinstance(vectors,torch.Tensor) or vectors.shape!=(len(ids),2048) or
        vectors.dtype!=torch.float16 or not torch.isfinite(vectors).all() or
        not (vectors.float().norm(dim=-1)>0).all()):
        raise ValueError('Invalid reusable teacher vectors')
    if len({s.sample_id for s in samples})!=len(samples) or any(s.split!='train' for s in samples):
        raise ValueError('New teacher cache must use unique training-only samples')
    positions={sid:i for i,sid in enumerate(ids)};reused={}
    for sample in samples:
        i=positions.get(sample.sample_id)
        if i is None:continue
        if sha256(sample.image_path)!=hashes[i]:
            raise ValueError('Reusable teacher image content changed')
        reused[sample.sample_id]=vectors[i].detach().clone()
    if sha256(path)!=expected_sha256:raise ValueError('Reuse cache changed while reading')
    return reused,dict(source_cache_sha256=expected_sha256,source_pool_manifest_sha256=cache['pool_manifest_sha256'],
                       reused_images=len(reused),source_images_not_in_new_pool=len(ids)-len(reused),
                       new_teacher_forwards=len(samples)-len(reused))


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
    reused={};reuse_audit=None
    if getattr(args,'reuse_cache',None) is not None:
        reused,reuse_audit=reusable_training_vectors(args.reuse_cache,args.reuse_cache_sha256,samples,
            sha256(args.target/'data/abo_similarity10_manifest.csv'),str(args.model.resolve()),INSTRUCTION)
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
            if sample.sample_id in reused:vectors.append(reused[sample.sample_id])
            else:
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
               excluded_splits=['val','test'],student_inference_teacher_required=False,reuse= reuse_audit)
    torch.save(cache,args.output/'cache.pt')
    write_json(args.output/'final_report.json',dict(status='complete',kind='train_only_teacher_cache',source_commit=source_commit(),
        command=sys.argv,training_images=len(samples),training_products=len({s.product_id for s in samples}),
        teacher_trainable_parameters=cache['teacher_trainable_parameters'],cache_sha256=sha256(args.output/'cache.pt'),
        target_manifest_sha256=cache['target_manifest_sha256'],pool_manifest_sha256=cache['pool_manifest_sha256'],
        excluded_splits=cache['excluded_splits'],reuse=reuse_audit,gpu=torch.cuda.get_device_name(),model=cache['model'],python=sys.version,torch=torch.__version__))
    del loaded
    torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True);p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--abo',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/optical_top2_dc20.yaml')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path)
    p.add_argument('--reuse-cache-sha256')
    args=p.parse_args()
    if (args.reuse_cache is None)!=(args.reuse_cache_sha256 is None):p.error('Supply reuse cache and SHA together')
    build(args)


if __name__=='__main__':main()
