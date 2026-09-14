"""Reproduce two DIFFERENT ABO retrieval protocols without training Qwen.

Can run as `python baseline.py ...` outside the parent repository.
No project imports, optical weights, learned head, optimizer or backward pass.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.nn import functional as F

PROMPT = 'Represent this catalog product image for category-aware visual similarity retrieval.'
PARENT_SHA = '2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97'
ENROLLED_SHA = 'f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde'
MODEL_REVISION = '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'
MODEL_FILES_SHA = {
    'model.safetensors': 'c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1',
    'config.json': '9172f55b0b9cce70b7f67b10c58a408ccf3ec15c587e6efd4d5f41631237fded',
    'preprocessor_config.json': 'd75c1f9f474add87cb0188d43c4fbf5c232d2af64f156fcefc5311089049100d',
    'chat_template.jinja': 'a47e6afb389f86f45be7810f17d2686fd42b2bec7ba6e6958abf85845af258c5',
    'tokenizer_config.json': '355f2b4e5bad7b01f11ef6cb68ebc176f61b95c3276092ea225b1bea0e01e95c',
    'tokenizer.json': 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'added_tokens.json': 'c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680',
    'special_tokens_map.json': '76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_parent(rows):
    if len(rows) != 2400 or len({r['sample_id'] for r in rows}) != 2400:
        raise ValueError('Require all 2400 unique images')
    if len({r['image_path'] for r in rows}) != 2400:
        raise ValueError('Repeated image path')
    if Counter(r['split'] for r in rows) != dict(train=1440, val=480, test=480):
        raise ValueError('Original 1440/480/480 split changed')
    products = defaultdict(list)
    for r in rows:
        products[r['product_id']].append(r)
    if len(products) != 200:
        raise ValueError('Require 200 products')
    counts = Counter()
    for records in products.values():
        if len(records) != 12 or len({(r['split'], r['category_id']) for r in records}) != 1:
            raise ValueError('Product identity crosses split/category, or not 12 views')
        counts[(records[0]['split'], records[0]['category_id'])] += 1
    if counts != {(s, c): n for s, n in [('train', 12), ('val', 4), ('test', 4)] for c in range(10)}:
        raise ValueError('Original per-category split changed')


def load_rows(manifest, protocol, enrolled_manifest=None, data=None):
    if sha256(manifest) != PARENT_SHA:
        raise ValueError('Original manifest SHA mismatch; do not silently change split')
    with Path(manifest).open(encoding='utf-8', newline='') as f:
        parent = list(csv.DictReader(f))
    for r in parent:
        r['category_id'] = int(r['category_id'])
    validate_parent(parent)
    if protocol == 'legacy_category':
        rows = [dict(r, split='gallery') for r in parent if r['split'] == 'train']
        rows += [dict(r, split='query') for r in parent if r['split'] == 'test']
    else:
        if enrolled_manifest is None or sha256(enrolled_manifest) != ENROLLED_SHA:
            raise ValueError('Require the pinned enrolled protocol JSON (not a regenerated JSON with another source commit)')
        info = json.loads(Path(enrolled_manifest).read_text(encoding='utf-8'))
        if info['protocol'] != 'abo200_enrolled_sku_hash8train4query_v1' or info['parent_manifest_sha256'] != PARENT_SHA:
            raise ValueError('Wrong enrolled protocol')
        by_id = {r['sample_id']: r for r in parent}
        enrolled = info['rows']
        if len(enrolled) != 2400 or {r['sample_id'] for r in enrolled} != set(by_id):
            raise ValueError('Enrolled protocol omitted/repeated images')
        products = defaultdict(list)
        for r in enrolled:
            p = by_id[r['sample_id']]
            if any(r[k] != p[k] for k in ['product_id', 'category_id', 'image_path']) or r['original_split'] != p['split']:
                raise ValueError('Enrolled row identity changed')
            products[r['product_id']].append(r)
        for records in products.values():
            order = sorted(records, key=lambda r: hashlib.sha256(('abo-enrolled42:' + r['sample_id']).encode()).hexdigest())
            if [r['split'] for r in order] != ['train'] * 8 + ['query'] * 4:
                raise ValueError('Hash-based 8/4 rule changed')
        rows = [dict(r, split='gallery') for r in enrolled if r['split'] == 'train']
        rows += [dict(r) for r in enrolled if r['split'] == 'query']
        ghash = {r['image_sha256'] for r in rows if r['split'] == 'gallery'}
        qhash = {r['image_sha256'] for r in rows if r['split'] == 'query'}
        if ghash & qhash:
            raise ValueError('Identical image bytes cross gallery/query')
    if data is not None:
        root = Path(data).resolve()
        for r in rows:
            path = (root / r['image_path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing image or path escapes data root: ' + str(path))
            digest = sha256(path)
            if 'image_sha256' in r and digest != r['image_sha256']:
                raise ValueError('Image bytes changed')
            r['image_sha256'] = digest
    return rows


def normalize(z):
    if z.ndim != 2 or not torch.isfinite(z).all() or (z.float().norm(dim=-1) <= 1e-12).any():
        raise ValueError('Invalid descriptors')
    return F.normalize(z.float(), dim=-1)


def pool_last(hidden, mask):
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
        raise ValueError('Expected hidden[B,S,D] and mask[B,S]')
    pos = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask).masked_fill(mask == 0, -1).amax(1)
    if (pos < 0).any():
        raise ValueError('All-padding sequence')
    return hidden[torch.arange(len(mask), device=mask.device), pos].float()


def metrics(relevant):
    relevant = np.asarray(relevant, dtype=np.float64)
    positives = relevant.sum(1)
    if not len(relevant) or (positives == 0).any():
        raise ValueError('No positive candidate for a query')
    out = dict(query_count=len(relevant), candidate_count=relevant.shape[1], relevant_candidates_per_query=float(positives.mean()))
    for k in [1, 5, 10]:
        used = min(k, relevant.shape[1]); hits = relevant[:, :used].sum(1)
        out[f'hit_at_{k}'] = float((hits > 0).mean())
        out[f'precision_at_{k}'] = float((hits / used).mean())
        out[f'positive_recall_at_{k}'] = float((hits / positives).mean())
    n = min(10, relevant.shape[1]); top = relevant[:, :n]
    precision = np.cumsum(top, axis=1) / np.arange(1, n + 1)
    discount = 1 / np.log2(np.arange(2, n + 2))
    out['map_at_10'] = float(((precision * top).sum(1) / np.minimum(n, positives)).mean())
    ideal = np.array([discount[:int(min(n, p))].sum() for p in positives])
    out['ndcg_at_10'] = float(((top * discount).sum(1) / ideal).mean())
    return out


def evaluate_vectors(vectors, rows, protocol):
    z = normalize(vectors)
    if len(z) != len(rows) or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Descriptors/unique row identities disagree')
    gi = [i for i, r in enumerate(rows) if r['split'] == 'gallery']
    qi = [i for i, r in enumerate(rows) if r['split'] == 'query']
    if not gi or not qi:
        raise ValueError('Need gallery and query')
    if protocol == 'legacy_category':
        groups = defaultdict(list)
        for i in gi: groups[rows[i]['product_id']].append(i)
        keys = sorted(groups)
        gallery = normalize(torch.stack([z[groups[k]].mean(0) for k in keys]))
        gr = [rows[groups[k][0]] for k in keys]
        if {r['product_id'] for r in gr} & {rows[i]['product_id'] for i in qi}:
            raise ValueError('Old protocol requires unseen product queries')
        scores = (normalize(z[qi]) @ gallery.T).numpy()
        order = np.argsort(-scores, axis=1)  # Historical NumPy ranking convention.
    else:
        gallery = z[gi]; gr = [rows[i] for i in gi]
        scores = z[qi] @ gallery.T
        order = scores.argsort(dim=1, descending=True, stable=True).numpy()
    sku = np.asarray([[rows[i]['product_id'] == gr[j]['product_id'] for j in rank] for i, rank in zip(qi, order)])
    category = np.asarray([[rows[i]['category_id'] == gr[j]['category_id'] for j in rank] for i, rank in zip(qi, order)])
    relevant = category if protocol == 'legacy_category' else sku
    out = metrics(relevant)
    out['relevance'] = 'same category' if protocol == 'legacy_category' else 'exact SKU'
    out['same_ranking_category_hit_at_1_diagnostic'] = float(category[:, 0].mean())
    out['top1_same_category_wrong_sku_count'] = int((category[:, 0] & ~sku[:, 0]).sum())
    out['hit_count'] = int(relevant[:, 0].sum())
    predictions = [dict(sample_id=rows[i]['sample_id'], product_id=rows[i]['product_id'], category_id=rows[i]['category_id'],
        top1_product_id=gr[rank[0]]['product_id'], top1_category_id=gr[rank[0]]['category_id'],
        hit=int(ok[0]), same_category=int(cat[0]), same_sku=int(same[0]))
        for i, rank, ok, cat, same in zip(qi, order, relevant, category, sku)]
    return out, predictions


def load_cache(path, rows, protocol, preprocessing, expected_sha):
    if sha256(path) != expected_sha:
        raise ValueError('Feature cache SHA mismatch')
    c = torch.load(path, map_location='cpu', weights_only=True)
    if 'vectors_by_dimension' in c:
        if c['protocol'] != protocol or c['preprocessing'] != preprocessing:
            raise ValueError('Cache protocol/preprocessing differs')
        if c['parent_manifest_sha256'] != PARENT_SHA or (protocol == 'enrolled_sku' and c['manifest_sha256'] != ENROLLED_SHA):
            raise ValueError('Cache dataset differs')
        ids, values = c['ids'], c['vectors_by_dimension']
    elif protocol == 'legacy_category':
        ident = c['identity']
        if ident['manifest_sha256'] != PARENT_SHA or ident['prompt'] != PROMPT or ident['processor_min_pixels'] != 50176 or ident['processor_max_pixels'] != 50176:
            raise ValueError('Historical cache identity differs')
        ids = ident['ids']; values = {str(d): c[preprocessing][:, :d].float() for d in [64, 2048]}
    else:
        if c['manifest_sha256'] != ENROLLED_SHA or preprocessing != 'native':
            raise ValueError('Enrolled cache identity differs')
        ids = c['ids']; values = {'64': c['vectors']}
    if len(ids) != len(set(ids)) or set(ids) != {r['sample_id'] for r in rows}:
        raise ValueError('Cache omitted/repeated/different sample IDs')
    by_id = {name: i for i, name in enumerate(ids)}
    order = [by_id[r['sample_id']] for r in rows]
    for d, value in values.items():
        if value.shape != (len(ids), int(d)):
            raise ValueError('Cache dimension mismatch')
    return {d: v[order].float() for d, v in values.items()}


@torch.inference_mode()
def infer(args, rows):
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = None
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, min_pixels=50176, max_pixels=50176)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model), local_files_only=True,
            dtype=dtype, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Baseline must have ZERO trainable parameters')
        values = {str(d): [] for d in [64, 2048]}
        for start in range(0, len(rows), args.batch_size):
            images, texts = [], []
            for row in rows[start:start + args.batch_size]:
                with Image.open(args.data / row['image_path']) as src:
                    im = (ImageOps.exif_transpose(src).convert('RGB') if args.preprocessing == 'native' else
                          ImageOps.fit(src.convert('RGB'), (224, 224), Image.Resampling.BICUBIC, centering=(.5, .5)))
                messages = [{'role': 'system', 'content': [{'type': 'text', 'text': PROMPT}]},
                            {'role': 'user', 'content': [{'type': 'image', 'image': im}]}]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)); images.append(im)
            batch = processor(text=texts, images=images, padding=True, return_tensors='pt')
            batch = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']}
            ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and args.protocol == 'enrolled_sku' else contextlib.nullcontext()
            with ctx:
                hidden = model.model(**batch, use_cache=False, return_dict=True).last_hidden_state
                pooled = pool_last(hidden, batch['attention_mask'])
                # Historical pipeline stored normalized2048 in FP16 before prefix slicing.
                if args.protocol == 'legacy_category': pooled = normalize(pooled).half().float()
                for d in values: values[d].append(normalize(pooled[:, :int(d)]).cpu())
            if start % (args.batch_size * 30) == 0: print(f'encoded {min(start+args.batch_size,len(rows))}/{len(rows)}', flush=True)
        return {d: torch.cat(v) for d, v in values.items()}
    finally:
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['audit', 'infer'])
    p.add_argument('--protocol', required=True, choices=['legacy_category', 'enrolled_sku'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--enrolled-manifest', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--features', type=Path)
    p.add_argument('--expected-features-sha256')
    p.add_argument('--preprocessing', choices=['native', 'square'], default='native')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'infer' and (args.model is None or args.data is None): p.error('infer requires --model and --data')
    if args.mode == 'audit' and (args.features is None or not args.expected_features_sha256): p.error('audit requires pinned --features and --expected-features-sha256')
    args.batch_size = args.batch_size if args.batch_size is not None else (1 if args.protocol == 'legacy_category' else 4)
    if args.batch_size < 1: p.error('batch-size must be positive')
    if args.output.exists(): raise FileExistsError('Choose a NEW output; never overwrite evidence')
    torch.set_num_threads(4); torch.manual_seed(42)
    rows = load_rows(args.manifest, args.protocol, args.enrolled_manifest, args.data if args.mode == 'infer' else None)
    try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = 'standalone archive; see PACKAGE_MANIFEST.json'
    identity = dict(protocol=args.protocol, mode=args.mode, preprocessing=args.preprocessing, prompt=PROMPT,
        parent_manifest_sha256=PARENT_SHA, manifest_sha256=ENROLLED_SHA if args.protocol == 'enrolled_sku' else PARENT_SHA,
        frozen=True, trainable_parameters=0, optimizer=None, epochs=0, test_selected=False,
        source_commit=commit, evaluator_sha256=sha256(__file__), pid=os.getpid(), python=platform.python_version(),
        torch=str(torch.__version__), numpy=np.__version__, batch_size=args.batch_size,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_features_sha256=sha256(args.features) if args.mode == 'audit' else None,
        timing_and_power='NOT benchmarked; wall time includes model loading and disk I/O')
    if args.mode == 'infer':
        identity['model_path'] = str(args.model.resolve())
        identity['model_files_sha256'] = {x.name: sha256(x) for x in sorted(args.model.iterdir()) if x.is_file() and x.name not in ['README.md', '.gitattributes']}
        for name, expected in MODEL_FILES_SHA.items():
            if identity['model_files_sha256'].get(name) != expected:
                raise ValueError('Pinned Qwen model/processor bytes differ: ' + name)
        identity['model_revision'] = MODEL_REVISION
        identity['image_files_sha256'] = {r['sample_id']: r['image_sha256'] for r in rows}
    args.output.mkdir(parents=True)
    write_json(args.output/'execution.json', identity); write_json(args.output/'status.json', dict(status='running', pid=os.getpid()))
    started = time.time()
    try:
        vectors = (infer(args, rows) if args.mode == 'infer' else
                   load_cache(args.features, rows, args.protocol, args.preprocessing, args.expected_features_sha256))
        reports = {}
        for d, z in vectors.items():
            reports[d], predictions = evaluate_vectors(z, rows, args.protocol)
            with (args.output/f'predictions_{d}d.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
        if args.mode == 'infer':
            torch.save(dict(protocol=args.protocol, preprocessing=args.preprocessing, parent_manifest_sha256=PARENT_SHA,
                manifest_sha256=identity['manifest_sha256'], ids=[r['sample_id'] for r in rows], vectors_by_dimension=vectors), args.output/'features.pt')
        report = dict(identity, status='complete', metrics_by_dimension=reports, elapsed_seconds=time.time()-started)
        write_json(args.output/'report.json', report); write_json(args.output/'status.json', dict(status='complete', pid=os.getpid()))
        print(json.dumps(dict(status='complete', metrics_by_dimension=reports), indent=2))
    except BaseException as e:
        write_json(args.output/'status.json', dict(status='failed_or_interrupted', pid=os.getpid(), error=repr(e))); raise


if __name__ == '__main__':
    main()
