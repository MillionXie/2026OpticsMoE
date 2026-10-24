"""Describe existing ABO difficulty using audited frozen caches, never tune TEST.

No new model, optimizer, feature extraction, split, candidate filtering or
checkpoint selection. Margins describe all original queries unconditionally.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from .easy100_data import Easy100
from .other_data import abo_data
from .evaluate_retrieval_budget import load_feature_cache
from .frozen_resnet18 import file_sha256
from .train_easy100 import TASK,ROOT,git,require_published_source,write_json

ENCODER_SHA='9287cf5db296812d2ef098170cf599fc8a4c3e2b854af87e10db9a542e947c9c'


def catalog_identity(train_skus,test_skus,title_skus=None):
    train=set(train_skus);test=set(test_skus)
    result=dict(TRAIN_SKUs=len(train),TEST_SKUs=len(test),shared_SKUs=len(train&test),
        TEST_unseen_SKUs=len(test-train),TRAIN_views_per_SKU=dict(Counter(Counter(train_skus).values())),
        TEST_views_per_SKU=dict(Counter(Counter(test_skus).values())),
        fraction_TEST_queries_of_seen_SKUs=sum(s in train for s in test_skus)/len(test_skus))
    if title_skus is not None:
        result.update(title_candidates=len(title_skus),title_candidates_shared_with_TRAIN=len(set(title_skus)&train),
            image_to_text_random_H1=1/len(title_skus),
            text_to_image_random_H1=sum(Counter(test_skus)[s]/len(test_skus) for s in title_skus)/len(title_skus))
    return result


def descriptor_margins(reference,queries,reference_skus,query_skus,*,exclude_self=False):
    if exclude_self and (reference_skus!=query_skus or len(reference)!=len(queries)):
        raise ValueError('Self-exclusion requires identical TRAIN order')
    reference=F.normalize(reference.float(),dim=-1);queries=F.normalize(queries.float(),dim=-1)
    positive=[];negative=[]
    for begin in range(0,len(queries),256):
        scores=queries[begin:begin+256]@reference.T
        same=torch.tensor([[q==c for c in reference_skus] for q in query_skus[begin:begin+256]],dtype=torch.bool)
        if exclude_self:scores[torch.arange(len(scores)),torch.arange(begin,begin+len(scores))]=-torch.inf
        p=scores.masked_fill(~same,-torch.inf).max(1).values;n=scores.masked_fill(same,-torch.inf).max(1).values
        if not torch.isfinite(p).all() or not torch.isfinite(n).all():raise ValueError('Missing positive/negative reference')
        positive.extend(p.tolist());negative.extend(n.tolist())
    delta=torch.tensor(positive)-torch.tensor(negative)
    def stats(values):
        values=torch.as_tensor(values)
        return dict(mean=float(values.mean()),minimum=float(values.min()),maximum=float(values.max()),
                    quantiles_05_25_50_75_95=torch.quantile(values,torch.tensor([.05,.25,.5,.75,.95])).tolist())
    return dict(queries=len(queries),same_SKU_best_cosine=stats(positive),other_SKU_best_cosine=stats(negative),
                same_minus_other_cosine_margin=stats(delta),fraction_positive_margin=float(delta.gt(0).float().mean()),
                same_SKU_cosine_at_least_099=sum(x>=.99 for x in positive)/len(positive),
                all_queries_retained=True,exclude_identical_index=exclude_self)


def main(args):
    require_published_source()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only diagnosis')
    torch.set_num_threads(2);run=args.run_dir.resolve()
    if not run.is_relative_to((TASK/'runs/simulation').resolve()):raise ValueError('Run outside task')
    run.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='preflight',pid=os.getpid(),command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],
        git_commit=git('rev-parse','HEAD'),source_SHA=file_sha256(Path(__file__)),started_unix=time.time(),
        new_neural_forwards=0,new_optimizer_steps=0,query_filtering=False,split_changed=False,historical_TEST_observed=True)
    write_json(run/'run_manifest.json',manifest)
    try:
        easy=Easy100(ROOT/'data/abo_easy100_dataset_20260906');a,b,identity=abo_data()
        bundles={'easy100':(easy.train,easy.test,easy.identity,easy.title_skus),
                 'abo200':(a.rows,b.rows,identity,None)}
        reports={};bindings={}
        for name,(train_rows,test_rows,data_identity,titles) in bundles.items():
            base=TASK/'runs/simulation/retrieval_fixed_budget_cpu_20261024/cache'/name
            train,ma=load_feature_cache(base/'train',data_identity=data_identity,encoder_hash=ENCODER_SHA)
            test,mb=load_feature_cache(base/'test',data_identity=data_identity,encoder_hash=ENCODER_SHA)
            if train['sample_ids']!=[r['sample_id'] for r in train_rows] or test['sample_ids']!=[r['sample_id'] for r in test_rows]:
                raise ValueError('Original cache/sample identities differ')
            reports[name]=dict(protocol=catalog_identity(train['skus'],test['skus'],titles),
                TRAIN_leave_one_view_out=descriptor_margins(train['features'],train['features'],train['skus'],train['skus'],exclude_self=True),
                original_TEST_all_query_margin_diagnosis=descriptor_margins(train['features'],test['features'],train['skus'],test['skus']))
            bindings[name]={s:meta['cache_binding_sha256'] for s,meta in (('train',ma),('test',mb))}
        reports['easy100']['protocol']['distinct_title_texts']=len({x['title'].strip() for x in easy.titles})
        write_json(run/'result.json',dict(status='complete',reports=reports,cache_bindings=bindings,
            new_neural_forwards=0,new_optimizer_steps=0,
            interpretations=['Closed-catalog held-out-view retrieval is not unseen-SKU generalization.',
              'Frozen appearance separability can support fast alignment to100 repeated TRAIN titles; this diagnosis does not quantify a unique causal contribution.',
              'Cosine similarity is not proof of byte-identical or perceptual duplicates.',
              'A harder split would require the same protocol for all compared models; no original score or split was changed.']))
        manifest['status']='complete'
    except BaseException as exc:manifest.update(status='failed',error=repr(exc));raise
    finally:manifest['finished_unix']=time.time();write_json(run/'run_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args())
