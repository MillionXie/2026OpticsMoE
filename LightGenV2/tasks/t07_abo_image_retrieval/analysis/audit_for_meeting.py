"""Audit fixed artifacts and data without training, test-time fitting, or GPU use.

Image/dHash statistics are screening proxies, not semantic leakage certificates.
Bootstrap resamples products within categories, not correlated individual views.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from ..standalone.data import INSTRUCTION, _load_contract, _gallery_centroids, _category_prototypes, _evaluate
from ..standalone.io import sha256, write_json, write_csv, source_commit
from ..standalone.retrieval_training import product_bank, gallery_loss


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def score_features(train, test, vectors, ids):
    if len(set(ids))!=len(ids) or len(vectors)!=len(ids):raise ValueError('Feature identity is ambiguous')
    lookup={key:i for i,key in enumerate(ids)}
    tr=F.normalize(vectors[[lookup[s.sample_id] for s in train]].float(),dim=-1)
    te=F.normalize(vectors[[lookup[s.sample_id] for s in test]].float(),dim=-1)
    bank,meta=_gallery_centroids(train,tr)
    metrics,rows,categories=_evaluate(te,test,bank,meta,_category_prototypes(bank,meta))
    return metrics,rows,categories,tr,te


def bootstrap(test, rows_by_model, repeats=10000):
    products=sorted({s.product_id for s in test})
    product_categories={s.product_id:s.category_id for s in test}
    matrix={}
    for name,rows in rows_by_model.items():
        success={r['sample_id']:float(r['top1_relevant']) for r in rows}
        values={p:np.mean([success[s.sample_id] for s in test if s.product_id==p]) for p in products}
        matrix[name]=np.array([[values[p] for p in products if product_categories[p]==c] for c in range(10)])
    rng=np.random.default_rng(42);indices=rng.integers(0,4,size=(repeats,10,4))
    distributions={n:a[np.arange(10)[None,:,None],indices].mean((1,2)) for n,a in matrix.items()}
    result={n:dict(hit1=float(a.mean()),conditional_product_bootstrap_95=np.quantile(distributions[n],[.025,.975]).tolist()) for n,a in matrix.items()}
    for name in distributions:
        if name!='optical':result[name]['paired_gap_minus_optical_95']=np.quantile(distributions[name]-distributions['optical'],[.025,.975]).tolist()
    return result


def image_audit(samples, manifest, output):
    image_rows=[];hashes=[];exact=defaultdict(list)
    for s in samples:
        with Image.open(s.image_path) as image:
            rgb=np.array(image.convert('RGB'));height,width=rgb.shape[:2]
            gray=np.array(image.convert('L').resize((9,9),Image.Resampling.BILINEAR))
        bits=np.concatenate(((gray[:8,1:]>gray[:8,:-1]).ravel(),(gray[1:,:8]>gray[:-1,:8]).ravel()))
        hashes.append(np.packbits(bits))
        digest=sha256(s.image_path);exact[digest].append(s)
        foreground=np.min(rgb,axis=-1)<245
        side=min(width,height);left=(width-side)//2;top=(height-side)//2
        loss=1-float(foreground[top:top+side,left:left+side].sum())/max(1,int(foreground.sum()))
        image_rows.append(dict(sample_id=s.sample_id,product_id=s.product_id,split=s.split,category=s.category_name,
            width=width,height=height,nonwhite_fraction=float(foreground.mean()),nonwhite_outside_center_square=loss,
            sha256=digest,azimuth_manifest=manifest[s.sample_id].get('azimuth','')))
    write_csv(output/'image_audit.csv',image_rows)
    cross=[dict(sha256=k,samples=[s.sample_id for s in v]) for k,v in exact.items() if len({s.split for s in v})>1]
    hashes=np.stack(hashes);train_index=[i for i,s in enumerate(samples) if s.split=='train'];test_index=[i for i,s in enumerate(samples) if s.split=='test']
    lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8);near=[]
    for i in test_index:
        distance=lut[np.bitwise_xor(hashes[i],hashes[train_index])].sum(1)
        best=int(distance.argmin());j=train_index[best]
        near.append(dict(test_id=samples[i].sample_id,train_id=samples[j].sample_id,hamming128=int(distance[best]),
            test_category=samples[i].category_name,train_category=samples[j].category_name))
    write_csv(output/'nearest_train_dhash.csv',near)
    return dict(images=len(samples),unique_file_hashes=len(exact),exact_cross_split_groups=cross,
        test_images_with_nearest_train_dhash_le4=sum(x['hamming128']<=4 for x in near),
        dhash_caveat='Screen only: similar silhouettes/backgrounds can trigger; not proof of duplicate product.',
        dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in image_rows)),
        nonwhite_crop_loss_mean=float(np.mean([r['nonwhite_outside_center_square'] for r in image_rows])),
        crop_proxy_caveat='Nonwhite<245 is not a segmentation mask; shadows/background count too.'),image_rows


def figures(output,names,categories,products,confusion,examples,sample_lookup):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained');x=np.arange(10)
    for offset,model,color in [(-.2,'optical','#0072B2'),(.2,'square_64d','#D55E00')]:
        ax.bar(x+offset,[100*r['hit_at_1'] for r in categories[model]],.38,label=model,color=color)
    ax.set(xticks=x,xticklabels=names,ylabel='Test Hit@1 (%)',ylim=(0,105));ax.tick_params(axis='x',rotation=35);ax.legend()
    fig.savefig(output/'01_per_category.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].imshow(confusion,cmap='Blues',vmin=0,vmax=48)
    axes[0].set(xticks=range(10),yticks=range(10),xticklabels=names,yticklabels=names,xlabel='Optical retrieved category',ylabel='Query category')
    axes[0].tick_params(axis='x',rotation=90)
    for i in range(10):
        for j in range(10):
            if confusion[i,j]:axes[0].text(j,i,str(confusion[i,j]),ha='center',va='center',fontsize=8,color='white' if confusion[i,j]>25 else 'black')
    axes[1].hist([p['correct_views'] for p in products],bins=np.arange(-.5,13.5,1),rwidth=.85,color='#0072B2')
    axes[1].set(xlabel='Correct views per test product (out of 12)',ylabel='Number of products',xticks=[0,3,6,9,12])
    fig.savefig(output/'02_errors_by_product.png');plt.close(fig)
    fig,axes=plt.subplots(len(examples),3,figsize=(9,3*len(examples)),squeeze=False,layout='constrained')
    for row,entry in enumerate(examples):
        for col,key in enumerate(('query_id','optical_gallery_view_id','qwen_gallery_view_id')):
            s=sample_lookup[entry[key]]
            with Image.open(s.image_path) as im:axes[row,col].imshow(im.convert('RGB'))
            axes[row,col].set_title(f"{['Query','Optical top-1 representative','Qwen top-1 representative'][col]}\n{s.category_name} | {s.product_id}",fontsize=9)
            axes[row,col].axis('off')
    fig.savefig(output/'03_error_examples.png');plt.close(fig)


def run(args):
    if args.output.exists():raise FileExistsError('Use a new analysis directory')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    samples,names=_load_contract(args.data);lookup={s.sample_id:s for s in samples}
    manifest={r['sample_id']:r for r in read_csv(args.data/'data/abo_similarity10_manifest.csv')}
    train=[s for s in samples if s.split=='train'];test=[s for s in samples if s.split=='test']
    optical=torch.load(args.optical/'retrieval_features.pt',map_location='cpu',weights_only=True)
    teacher=torch.load(args.baseline/'features.pt',map_location='cpu',weights_only=True)
    if teacher['identity']['manifest_sha256']!=sha256(args.data/'data/abo_similarity10_manifest.csv'):raise ValueError('Baseline manifest differs')
    if teacher['identity']['prompt']!=INSTRUCTION:raise ValueError('Baseline prompt differs')
    cache_ids=teacher['identity']['ids']
    variants={'optical':(torch.cat((optical['train'],optical['test'])),optical['train_ids']+optical['test_ids'])}
    for geometry in ('native','square'):
        for dim in (2048,64):variants[f'{geometry}_{dim}d']=(teacher[geometry][:,:dim],cache_ids)
    metrics={};rows={};categories={}
    for name,(v,ids) in variants.items():
        result,pred,cat,tr,te=score_features(train,test,v,ids);metrics[name]=result;rows[name]=pred;categories[name]=cat
        write_csv(args.output/f'{name}_predictions.csv',pred)
        if name=='optical':
            b,y,own=product_bank(tr,train)
            _,_,hit=gallery_loss(tr,torch.tensor([s.category_id for s in train]),own,b,y)
            metrics[name]['train_leave_own_product_out_hit1']=float(hit)
    confusion=np.zeros((10,10),int);byproduct=defaultdict(list)
    for s,r in zip(test,rows['optical']):
        confusion[s.category_id,json.loads(r['top10_category_ids'])[0]]+=1;byproduct[s.product_id].append(r)
    products=[dict(product_id=p,category=v[0]['query_category_name'],correct_views=sum(r['top1_relevant'] for r in v),views=len(v)) for p,v in byproduct.items()]
    write_csv(args.output/'test_products.csv',products)
    worst=sorted(products,key=lambda p:(p['correct_views'],p['product_id']));picked=[];seen=set()
    teacher_rows={r['sample_id']:r for r in rows['square_64d']}
    representatives={}
    for s in train:representatives.setdefault(s.product_id,s.sample_id)
    for product in worst:
        if product['category'] in seen:continue
        errors=[r for r in byproduct[product['product_id']] if not r['top1_relevant']]
        if not errors:continue
        r=sorted(errors,key=lambda r:r['sample_id'])[0];q=teacher_rows[r['sample_id']]
        picked.append(dict(query_id=r['sample_id'],optical_gallery_view_id=representatives[json.loads(r['top10_product_ids'])[0]],qwen_gallery_view_id=representatives[json.loads(q['top10_product_ids'])[0]]));seen.add(product['category'])
        if len(picked)==6:break
    write_json(args.output/'example_selection.json',dict(rule='Worst product in each category, then lexicographically first erroneous sample; first six categories. Gallery images are first manifest views representing 12-view centroids, not necessarily nearest individual views.',examples=picked))
    images,image_rows=image_audit(samples,manifest,args.output)
    categories_rows=[dict(model=m,**r) for m,rs in categories.items() for r in rs];write_csv(args.output/'per_category.csv',categories_rows)
    view_groups=defaultdict(list)
    for s,r in zip(test,rows['optical']):view_groups[manifest[s.sample_id]['azimuth']].append(bool(r['top1_relevant']))
    write_csv(args.output/'view_groups.csv',[dict(manifest_azimuth=k,query_count=len(v),hit1=float(np.mean(v))) for k,v in sorted(view_groups.items())])
    brands={split:{manifest[s.sample_id]['brand'] for s in samples if s.split==split} for split in ('train','val','test')}
    report=dict(status='complete',source_commit=source_commit(),cpu_only=True,
        identity=dict(manifest_sha256=sha256(args.data/'data/abo_similarity10_manifest.csv'),optical_features_sha256=sha256(args.optical/'retrieval_features.pt'),baseline_features_sha256=sha256(args.baseline/'features.pt'),optical_run=str(args.optical),baseline_run=str(args.baseline)),
        metrics=metrics,bootstrap=bootstrap(test,rows),image_audit=images,
        product_errors=dict(all_views_wrong=sum(p['correct_views']==0 for p in products),all_views_correct=sum(p['correct_views']==12 for p in products),test_products=len(products)),
        brands=dict(train_count=len(brands['train']),test_count=len(brands['test']),test_only=sorted(brands['test']-brands['train'])),
        notes=['40 test products, 12 correlated views each; not 480 independent products.',
               'Optical checkpoint was test-selected; bootstrap is descriptive conditional uncertainty, not unbiased confirmation.',
               'Baseline uses first 64 dimensions of its frozen embeddings, not a trained student head.',
               'No train, val or test inputs modified; no image enhancement or test fitting.'])
    write_json(args.output/'audit.json',report)
    figures(args.output,[names[i] for i in range(10)],categories,products,confusion,picked,lookup)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','baseline','optical','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
