"""CPU paired product-cluster uncertainty and actual raw optical-route audit.

Scores remain automatic proxies, never automatic instruction pass rates.
No generation, model loading, TEST opening, or parameter selection occurs here.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

POSITIVE=('CLIP_T','DINOv2_I','CLIP_I')
METRICS=(*POSITIVE,'ImageReward_compact')


def product_key(row):
    # Current reference IDs are category-ABO_ASIN-view. Views of one product
    # share a bootstrap unit; repeated seeds are never independent products.
    category,asin,view=row['reference_id'].split('-')
    if category!=row['category'] or not asin or not view.isdigit():raise ValueError('Unknown reference-ID clustering contract')
    return category+'-'+asin


def paired_intervals(reference,candidate,repeats=2000):
    left={r['sample_id']:r for r in reference};right={r['sample_id']:r for r in candidate}
    if not left or len(left)!=len(reference) or set(left)!=set(right):raise ValueError('Exactly matching complete paired requests required')
    clusters=defaultdict(list)
    for key,a in left.items():
        b=right[key]
        for field in ['reference_id','category','condition_id','seed','prompt','input_sha256','initial_latent_sha256']:
            if a[field]!=b[field]:raise ValueError('Unmatched request/noise: '+field)
        clusters[product_key(a)].append(key)
    strata=defaultdict(list)
    for key in clusters:strata[key.split('-')[0]].append(key)
    def score(keys,metric):
        aa=statistics.mean(left[k][metric] for k in keys);bb=statistics.mean(right[k][metric] for k in keys)
        if metric in POSITIVE:
            if aa<=0:raise ValueError('Positive reference mean required for relative score')
            return bb/aa-1
        return bb-aa
    values={m:[] for m in METRICS};rng=random.Random(1742)
    for _ in range(repeats):
        # Preserve the fixed category composition when auditing the full set.
        keys=[i for products in strata.values() for chosen in rng.choices(sorted(products),k=len(products)) for i in clusters[chosen]]
        for m in METRICS:values[m].append(score(keys,m))
    result={}
    for m,samples in values.items():
        samples.sort();lo=samples[int(.025*repeats)];hi=samples[min(repeats-1,int(.975*repeats))]
        result[m]={'paired_relative_change' if m in POSITIVE else 'paired_absolute_change':score(list(left),m),
            'product_cluster_bootstrap_percentile_95_interval':[lo,hi]}
    return dict(requests=len(left),products=len(clusters),bootstrap_repeats=repeats,metrics=result,
        automatic_attribute_accuracy=False,independent_product_sample_size=len(clusters),
        warning='Few product clusters give uncertain intervals; repeated views/seeds do not increase product count.')


def route_summary(rows):
    result={}
    for modality in ['vision','language']:
        masks=[];alphas=[]
        for row in rows:
            d=(row.get('actual_raw_optical_routing') or {}).get(modality)
            if d is None:continue
            if d['router_calls_per_forward']!=1:raise ValueError('Single-router contract violated')
            if not all(.6<=v<=.900001 for v in d['alpha']):raise ValueError('Fusion bound violated')
            if (d['physical_active_roi'],d['propagation_canvas'],d['expert_tile'],d['top_k'])!=(478,518,224,2):raise ValueError('Physical contract changed')
            selected=d['selected_mask']
            if len(selected)!=1 or len(selected[0])!=4 or sum(v>.5 for v in selected[0])!=2:raise ValueError('B1 four-expert top2 required')
            masks.append(tuple(i for i,v in enumerate(selected[0]) if v>.5));alphas.extend(d['alpha'])
        if masks:
            counts=Counter(masks);loads=Counter(i for mask in masks for i in mask)
            result[modality]=dict(requests=len(masks),selection_counts={','.join(map(str,k)):v for k,v in sorted(counts.items())},
                observed_routing_varies_across_requests=len(counts)>1,expert_selection_fractions=[loads[i]/len(masks) for i in range(4)],
                alpha_range=[min(alphas),max(alphas)],source='actual raw RGB+text inference, not teacher-prefix KD inputs')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--raw-candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();assert json.loads((a.metrics/'report.json').read_text())['complete']
    rows=json.loads((a.metrics/'per_image.json').read_text());left=[x for x in rows if x['run_id']==a.reference];right=[x for x in rows if x['run_id']==a.candidate]
    raw=json.loads((a.raw_candidate/'images.json').read_text());assert json.loads((a.raw_candidate/'report.json').read_text())['complete']
    assert {r['sample_id']:r['generated_sha256'] for r in raw}=={r['sample_id']:r['generated_sha256'] for r in right}
    groups={group:paired_intervals([r for r in left if group in ['all',r['category']]],[r for r in right if group in ['all',r['category']]]) for group in ['all','lamp','pillow']}
    a.output.mkdir(parents=True)
    report=dict(complete=True,reference=a.reference,candidate=a.candidate,groups=groups,actual_raw_routing=route_summary(raw),
        argv=sys.argv,source_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric_input_sha256=hashlib.sha256((a.metrics/'per_image.json').read_bytes()).hexdigest(),
        generated_manifest_sha256=hashlib.sha256((a.raw_candidate/'images.json').read_bytes()).hexdigest(),
        generation_or_training_performed=False,quality_or_speed_accepted=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
