"""Bounded, product-balanced ABO pretraining pool; exclude all target products/images."""
import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .io import picture,sha256,write_json,write_csv,source_commit
from .data import _load_contract

POPCOUNT=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)


def image_signature(path):
    im=picture(path).convert('L')
    h=np.asarray(im.resize((9,8),Image.Resampling.BILINEAR),dtype=np.int16)
    v=np.asarray(im.resize((8,9),Image.Resampling.BILINEAR),dtype=np.int16)
    return np.packbits(np.concatenate([(h[:,1:]>h[:,:-1]).ravel(),(v[1:]>v[:-1]).ravel()]))


def near_duplicate(signature,protected,threshold=4):
    return bool((POPCOUNT[np.bitwise_xor(protected,signature)].sum(1)<=threshold).any())


def safe_image(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Image metadata escapes ABO root')
    return path


def prepare(args):
    if args.output.exists():raise FileExistsError(args.output)
    target_samples,_=_load_contract(args.target)
    target_rows=list(csv.DictReader((args.target/'data/abo_similarity10_manifest.csv').open(encoding='utf-8')))
    blocked_products={r['product_id'] for r in target_rows};blocked_ids={r['image_id'] for r in target_rows}
    protected=np.stack([image_signature(s.image_path) for s in target_samples])
    protected_sha={sha256(s.image_path) for s in target_samples}
    image_meta=args.abo/'images/metadata/images.csv.gz'
    with gzip.open(image_meta,'rt') as stream:
        images={r['image_id']:r for r in csv.DictReader(stream)}
    products={};metadata_files=sorted((args.abo/'listings/metadata').glob('*.json.gz'))
    for path in metadata_files:
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);pid=row['item_id']
                if pid in products:continue
                types=row.get('product_type') or []
                if not types:continue
                image_ids=list(dict.fromkeys([row.get('main_image_id'),*row.get('other_image_id',[])]))
                products[pid]=(types[0]['value'],[i for i in image_ids if i])
    groups=defaultdict(list);excluded=Counter()
    for pid,(category,ids) in products.items():
        if pid in blocked_products:excluded['target_product']+=1;continue
        if set(ids)&blocked_ids:excluded['shared_target_image_id_product']+=1;continue
        usable=[i for i in ids if i in images and min(int(images[i]['width']),int(images[i]['height']))>=96]
        if len(usable)>=2:groups[category].append((pid,usable))
    # Cap common classes so phone cases cannot dominate; generic buckets are ambiguous labels.
    generic={'UNKNOWN','HOME','GROCERY','KITCHEN','OFFICE_PRODUCTS','SPORTING_GOODS','HEALTH_PERSONAL_CARE','HOME_FURNITURE_AND_DECOR'}
    categories=sorted((c for c,g in groups.items() if len(g)>=args.minimum_products and c not in generic),key=lambda c:(-len(groups[c]),c))
    rng=random.Random(42);rows=[];seen_hashes=set();seen_ids=set();counts={}
    for category in categories:
        if len(counts)>=args.categories:break
        candidates=groups[category][:];rng.shuffle(candidates);selected=[];product_count=0
        for pid,ids in candidates:
            chosen=[];reject_product=False
            for iid in ids:
                path=safe_image(args.abo/'images/small',images[iid]['path'])
                if not path.is_file():excluded['missing_image']+=1;continue
                try:
                    digest=sha256(path);signature=image_signature(path)
                except (OSError,ValueError):excluded['invalid_image']+=1;continue
                if digest in protected_sha or near_duplicate(signature,protected,args.hamming_threshold):
                    excluded['target_exact_or_near_duplicate_product']+=1;reject_product=True;break
                if iid in seen_ids or digest in seen_hashes or digest in {x['image_sha256'] for x in chosen}:
                    excluded['pool_duplicate_image']+=1;continue
                chosen.append(dict(product_id=pid,image_id=iid,category=category,
                    image_path=path.relative_to(args.abo.resolve()).as_posix(),image_sha256=digest,
                    signature128_hex=signature.tobytes().hex()))
                if len(chosen)==args.views:break
            if reject_product or len(chosen)<args.views:continue
            selected.extend(chosen);seen_ids.update(r['image_id'] for r in chosen);seen_hashes.update(r['image_sha256'] for r in chosen)
            product_count+=1
            if product_count>=args.products_per_category:break
        if product_count>=args.minimum_products:
            cid=len(counts);counts[category]=product_count
            for row in selected:row.update(category_id=cid,sample_id=row['product_id']+'__'+row['image_id'])
            rows.extend(selected)
            print(json.dumps({'category':category,'products':product_count,'classes':len(counts),'images':len(rows)}),flush=True)
    if len(counts)<20:raise RuntimeError(f'Only {len(counts)} usable types; inspect coverage before training')
    args.output.mkdir(parents=True)
    write_csv(args.output/'manifest.csv',rows)
    write_json(args.output/'report.json',dict(source_commit=source_commit(),seed=42,
        purpose='External-to-target ABO subset pretraining, not the target 10-class train images',
        abo_root=str(args.abo.resolve()),target_root=str(args.target.resolve()),
        target_manifest_sha256=sha256(args.target/'data/abo_similarity10_manifest.csv'),
        manifest_sha256=sha256(args.output/'manifest.csv'),image_metadata_sha256=sha256(image_meta),
        listing_metadata_sha256={p.name:sha256(p) for p in metadata_files},
        raw_products=len(products),raw_types=len({v[0] for v in products.values()}),
        selected_types=len(counts),selected_products=sum(counts.values()),selected_images=len(rows),
        products_per_type=counts,excluded=dict(excluded),
        blocked_target_products=len(blocked_products),blocked_target_images=len(target_rows),
        target_product_overlap=len(blocked_products&{r['product_id'] for r in rows}),
        target_image_id_overlap=len(blocked_ids&{r['image_id'] for r in rows}),
        duplicate_screen='file SHA256 + 128-bit horizontal/vertical dHash on fixed 224 crop',
        hamming_threshold=args.hamming_threshold,
        limitation='Conservative near-duplicate heuristic, not proof that all semantic product variants are disjoint',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    print((args.output/'report.json').read_text(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--abo',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--categories',type=int,default=128);p.add_argument('--products-per-category',type=int,default=48)
    p.add_argument('--minimum-products',type=int,default=20);p.add_argument('--views',type=int,default=2)
    p.add_argument('--hamming-threshold',type=int,default=4)
    a=p.parse_args()
    if a.views<2 or a.minimum_products<4 or a.products_per_category<a.minimum_products:raise ValueError('Invalid sampling limits')
    prepare(a)


if __name__=='__main__':main()
