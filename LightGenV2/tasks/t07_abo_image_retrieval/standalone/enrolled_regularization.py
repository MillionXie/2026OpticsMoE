"""TRAIN-only external-SKU curriculum and whole-object augmentation.

No inference module or optical geometry changes. Never loads old target-trained
weights from a different split. External listing photos may include packaging or
detail views: same item_id is the positive label, not a claim of pure rotation.
"""
import csv
import json
import hashlib
from collections import Counter

import torch
from torch.nn import functional as F
from PIL import Image, ImageEnhance, ImageFilter

from .io import sha256
from .prepare_broad_abo import safe_image


def spin_target_identity(protocol):
    return hashlib.sha256(json.dumps(protocol['rows'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_external_pool(pool, root, protocol, groups, expected_sha):
    manifest = pool / 'manifest.csv'
    report = json.loads((pool / 'report.json').read_text())
    if not expected_sha or sha256(manifest) != expected_sha or report['manifest_sha256'] != expected_sha:
        raise ValueError('External manifest SHA mismatch')
    if report['target_manifest_sha256'] != protocol['parent_manifest_sha256']:
        raise ValueError('External pool target-exclusion contract mismatch')
    protected = groups['train'] + groups['query']
    spin_pool = report.get('pool_kind') == 'abo_spin_pretrain_v1'
    if spin_pool and (report.get('status') != 'ready'
                      or protocol.get('protocol') != 'abo200_enrolled_sku_hash8train4query_v1'
                      or report.get('enrolled_rows_sha256') != spin_target_identity(protocol)):
        raise ValueError('External spin pool current target partition mismatch')
    blocked_spins = {r.get('spin_id') for r in protected} - {None, ''}
    spin_owners, spin_angles, product_spins = {}, {}, {}
    if spin_pool and (type(report.get('views_per_product')) is not int or report['views_per_product'] < 2):
        raise ValueError('Invalid external spin view requirement')
    blocked_products = {r['product_id'] for r in protected}
    blocked_hashes = {r['image_sha256'] for r in protected}
    rows, ids, hashes = [], set(), set()
    with manifest.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if spin_pool:
                sid = row.get('spin_id')
                if (not sid or sid in blocked_spins or spin_owners.get(sid, row['product_id']) != row['product_id']
                        or product_spins.get(row['product_id'], sid) != sid):
                    raise ValueError('Target/shared spin in external pool')
                spin_owners[sid] = row['product_id']
                product_spins[row['product_id']] = sid
                spin_angles.setdefault(row['product_id'], set()).add(int(row['azimuth']))
            path = safe_image(root, row['image_path'])
            digest = sha256(path)
            if digest != row['image_sha256']:
                raise ValueError('External image changed')
            if row['product_id'] in blocked_products or digest in blocked_hashes:
                raise ValueError('Target product/image in external pool')
            if row['sample_id'] in ids or digest in hashes:
                raise ValueError('Duplicate external image')
            ids.add(row['sample_id']); hashes.add(digest)
            rows.append(dict(row, image_path=str(path), split='train', category_id=int(row['category_id'])))
    counts = Counter(r['product_id'] for r in rows)
    if spin_pool and any(len(spin_angles[k]) != report['views_per_product'] or count != report['views_per_product'] for k, count in counts.items()):
        raise ValueError('External spin view count mismatch')
    if len(rows) != report['selected_images'] or len(counts) != report['selected_products'] or not counts or min(counts.values()) < 2:
        raise ValueError('Invalid external counts or single-view SKU')
    fit = dict(train=rows, gallery=[dict(r, split='gallery') for r in rows], exclude_self=True,
               note='External SKU instance positives, ALL pool photos, self excluded; no target images')
    audit = dict(manifest_sha256=expected_sha, products=len(counts), images=len(rows),
                 pool_kind=report.get('pool_kind', 'abo_listing_pool'), target_spin_overlap=0 if spin_pool else None,
                 protected_products=len(blocked_products), target_product_overlap=0, target_sha_overlap=0,
                 original_duplicate_screen=report.get('duplicate_screen'),
                 original_hamming_threshold=report.get('hamming_threshold'),
                 limitation='Rechecked all file SHA and product IDs; original near-duplicate heuristic retained, not proof of semantic variant independence')
    return fit, audit


def load_external_relations(path, expected_sha, fit, protocol, pool_sha):
    """Reuse frozen Qwen cache, retaining ONLY revalidated external images.

    Historical cache also contains old target TRAIN images (some are now QUERY).
    None of those rows survive. No trained projection/old student is imported.
    """
    if not expected_sha or sha256(path) != expected_sha:
        raise ValueError('External teacher cache SHA mismatch')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    if (cache.get('schema') != 1 or cache.get('frozen_teacher') is not True
        or cache.get('teacher_trainable_parameters') != 0
        or cache.get('target_manifest_sha256') != protocol['parent_manifest_sha256']
        or cache.get('pool_manifest_sha256') != pool_sha
        or cache.get('prompt') != 'Represent this catalog product image for category-aware visual similarity retrieval.'
        or cache.get('preprocessing') != 'EXIF RGB; native aspect; processor min=max pixels 50176'
        or not str(cache.get('model', '')).endswith('/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda')):
        raise ValueError('External frozen teacher identity mismatch')
    ids, hashes, vectors = cache.get('ids', []), cache.get('image_sha256', []), cache.get('vectors')
    if (not ids or len(set(ids)) != len(ids) or len(hashes) != len(ids)
        or not isinstance(vectors, torch.Tensor) or vectors.shape != (len(ids), 2048)
        or vectors.dtype != torch.float16 or not torch.isfinite(vectors).all()):
        raise ValueError('Invalid external teacher vectors')
    index = {sid: i for i, sid in enumerate(ids)}
    protected = protocol.get('rows', [])
    blocked_ids = {r['sample_id'] for r in protected}
    blocked_products = {r['product_id'] for r in protected}
    blocked_hashes = {r['image_sha256'] for r in protected}
    kept = {}
    for row in fit['train']:
        sid = row['sample_id']
        if sid in blocked_ids or row['product_id'] in blocked_products or row['image_sha256'] in blocked_hashes:
            raise ValueError('Target image/product in external teacher fitting rows')
        if row['split'] != 'train' or sid not in index or hashes[index[sid]] != row['image_sha256']:
            raise ValueError('Missing/changed external teacher TRAIN image')
        vector = vectors[index[sid], :64].float().clone()
        if vector.norm() <= 0:
            raise ValueError('Zero external 64D teacher vector')
        kept[sid] = F.normalize(vector, dim=0)
    if set(kept) != {r['sample_id'] for r in fit['gallery']} or not kept:
        raise ValueError('External teacher gallery mismatch')
    return kept, dict(cache_sha256=expected_sha, retained_external_images=len(kept),
        discarded_other_cache_rows=len(ids)-len(kept), teacher_trainable_parameters=0,
        representation='Frozen2048 first64 then L2; external only, no target rows retained',
        prompt=cache['prompt'], preprocessing=cache['preprocessing'], model=cache['model'])


def curriculum_loss_weights(profile, external, taper):
    """External soft relations may replace SKU NLL; target ALWAYS retains NLL."""
    return ((profile.get('external_instance_weight', 1.), profile.get('external_teacher_weight', 0.))
            if external else (1., profile['teacher_weight']*taper))


def augment_whole_object(image, rng, mild=False):
    """Scale down into white canvas, never crop/flip/remove object parts."""
    image = image.convert('RGB')
    if mild:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(.95, 1.05))
        return ImageEnhance.Contrast(image).enhance(rng.uniform(.95, 1.05))
    side = image.width
    size = max(1, round(side * rng.uniform(.85, 1.)))
    canvas = Image.new('RGB', image.size, 'white')
    xy = (rng.randint(0, side-size), rng.randint(0, side-size))
    canvas.paste(image.resize((size, size), Image.Resampling.BILINEAR), xy)
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(.85, 1.15))
    canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(.85, 1.15))
    if rng.random() < .15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .5)))
    return canvas


def curriculum_epoch(epoch, pretrain_epochs, target_epochs):
    if not 1 <= epoch <= pretrain_epochs + target_epochs:
        raise ValueError('Epoch outside curriculum')
    external = epoch <= pretrain_epochs
    return external, epoch if external else epoch-pretrain_epochs, pretrain_epochs if external else target_epochs


@torch.no_grad()
def fitting_bank_diagnostics(vectors, labels, sample_ids, chunk_size=256):
    """Read-only TRAIN bank retrieval, excluding the exact same photograph.

    This reuses epoch-start clean encodings, not post-update TEST encodings.
    Chunked on the bank's existing device; no extra model forward or RNG calls.
    """
    n = len(sample_ids)
    if (vectors.ndim != 2 or len(vectors) != n or n < 4 or labels.shape != (n,)
            or len(set(sample_ids)) != n or chunk_size < 1
            or not torch.isfinite(vectors).all() or (vectors.float().norm(dim=1) == 0).any()):
        raise ValueError('Invalid clean fitting bank or duplicate image identity')
    labels = labels.to(vectors.device)
    _, counts = labels.unique(return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError('Each fitting SKU needs a distinct positive and a negative SKU')
    z = F.normalize(vectors.detach().float(), dim=1)
    hits, margins = [], []
    columns = torch.arange(n, device=z.device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        similarity = z[start:stop] @ z.T
        same = labels[start:stop, None] == labels[None]
        self_mask = torch.arange(start, stop, device=z.device)[:, None] == columns[None]
        similarity = similarity.masked_fill(self_mask, -torch.inf)
        hits.append(labels[similarity.argmax(1)] == labels[start:stop])
        best_positive = similarity.masked_fill(~same, -torch.inf).max(1).values
        best_negative = similarity.masked_fill(same, -torch.inf).max(1).values
        margins.append(best_positive - best_negative)
    margin = torch.cat(margins)
    return dict(hit_at_1=float(torch.cat(hits).float().mean()),
        median_best_positive_minus_negative_cosine=float(margin.quantile(.5)),
        mean_best_positive_minus_negative_cosine=float(margin.mean()),
        query_count=n, candidates_per_query=n-1, product_count=len(counts),
        timing='Epoch start, clean current-live TRAIN bank; NOT post-epoch or TEST accuracy',
        same_image_excluded=True, used_for_checkpoint_selection=False)
