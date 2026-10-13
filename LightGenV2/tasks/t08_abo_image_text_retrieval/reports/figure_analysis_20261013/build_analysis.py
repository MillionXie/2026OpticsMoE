"""Audit saved ABO I2T predictions and render reproducible scientific figures.

No model execution or training. Input predictions are preserved byte-for-byte.
Run from any directory: python /absolute/path/to/build_analysis.py
"""
from __future__ import annotations
import csv
import hashlib
import html
import json
import shutil
import textwrap
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = next((p for p in HERE.parents if (p/'LightGenV2').is_dir()), None)
RAW, DATA, FIG, CASES = [HERE / p for p in ('raw', 'processed', 'figures', 'cases')]
OUR_COLOR, REF_COLOR = '#16835F', '#3976B5'
CHECKPOINT = 'a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'
SELECTED = {
    92: ('bench', 'Light gray · Button-tufted · Storage bench'),
    29: ('desk', 'Black · Clear glass · Corner gaming desk'),
    25: ('monitor_arm', 'Black · Aluminum · Monitor arm stand'),
    86: ('vase', 'Off-white · Oval pattern · Stoneware vase'),
    6: ('baskets', 'Silver · Geometric wire · Basket'),
}
# Manual display excerpts from official titles. Three descriptors omit dimensions.
# They are not new retrieval candidates or independent attribute predictions.
DISPLAY = {
 0:('White','MFi certified','Lightning-to-USB cable'),
 1:('Stereo','Male-to-male','Audio cable'),
 2:('Portable','Remote-controlled','Air conditioner'),
 4:('Grey and gold','Hanging stoneware','Planter'),
 6:('Silver','Geometric wire','Basket'),
 7:('Ni-MH','USB port','Battery charger'),
 10:('Foldable','Metal','Platform bed frame'),
 11:('Modern','Velvet','Storage bench'),
 13:('Black and natural wood','Iron rungs','Blanket ladder'),
 14:('Multicolor','Cotton color-block','Throw pillow'),
 15:('Black','Curved metal','Wine rack'),
 16:('Silver','Aluminum','Step stool'),
 23:('White','Square wood','Wall clock'),
 24:('Black','Wood','Shoe rack'),
 25:('Black','Aluminum','Monitor arm stand'),
 29:('Black','Clear glass','Corner gaming desk'),
 30:('Beige','Modern stoneware','Decorative bowl'),
 32:('Teal and blue','Medallion stoneware','Coaster'),
 39:('Black','Flat workout','Exercise bench'),
 46:('Verdi green','Iron latticework','Hanging mirror'),
 49:('Silver','Metal and glass','Lantern'),
 53:('White','Stoneware with wood lid','Canister'),
 55:('Teal pattern','Anti-fatigue','Comfort mat'),
 56:('Grey and black','Folding steel','Step stool'),
 57:('Black','Glass column','Floor lamp'),
 59:('Folding','Memory foam','Guest bed'),
 65:('Black and white','Reading-by-water photo','Wall art'),
 66:('Ivory and turquoise','Stripe pattern','Throw pillow'),
 71:('Adjustable','Tablet-compatible','Tablet holder stand'),
 74:('Blue, gold and white','Rectangular concrete','Planter'),
 76:('Grey weave','Exposed wood','Sofa'),
 80:('Beige','Collapsible fabric','Storage cube'),
 81:('Black','Farmhouse wall-mounted','Coat rack'),
 84:('Silver','Dual-side aluminum','Step stool'),
 85:('Black','Plastic','Utility cart'),
 86:('Off-white','Oval-patterned stoneware','Vase'),
 92:('Light gray','Button-tufted','Storage bench'),
 93:('Glossy white and wood','Modern lacquer','Cabinet'),
 98:('Shell','Tapered legs','Ottoman'),
}

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_rows(path, records, fields=None):
    records = list(records)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(records[0]))
        w.writeheader()
        w.writerows(records)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def clean_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.6, length=3)

def panel(ax, char):
    ax.text(-.15, 1.06, char, transform=ax.transAxes, weight='bold', fontsize=11)

def main():
    for p in (RAW, DATA, FIG, CASES):
        p.mkdir(exist_ok=True)
    # A standalone extracted package already includes all raw inputs.
    # Repository-local sources are only used to populate absent inputs once.
    if not (RAW/'qwen_dynamic2048_predictions.csv').exists():
        if REPO is None:
            raise FileNotFoundError('Missing raw/qwen_dynamic2048_predictions.csv')
        baseline_source = REPO / 'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/qwen_frozen_5090d_easy100_controlled/retrieval_predictions.csv'
        shutil.copy2(baseline_source, RAW / 'qwen_dynamic2048_predictions.csv')
    for name in ('final_report.json', 'FINAL_VERIFICATION.json'):
        if not (RAW/('historical_'+name)).exists():
            if REPO is None:
                raise FileNotFoundError('Missing historical raw reference: '+name)
            shutil.copy2(REPO/'.codex_tmp/abo_import/review/inference/reference'/name, RAW / ('historical_' + name))
    titles = {int(r['label']): r for r in rows(RAW / 'titles.csv')}
    test = rows(RAW / 'test.csv')
    train = rows(RAW / 'train.csv')
    ours = rows(RAW / 'ours_same_weight_replay_predictions.csv')
    qwen = rows(RAW / 'qwen_dynamic2048_predictions.csv')
    report = json.loads((RAW / 'ours_same_weight_replay_metrics.json').read_text())
    assert report['checkpoint_sha256'] == CHECKPOINT
    assert len(test) == len(ours) == len(qwen) == 2400 and len(titles) == 100
    assert len(train) == 4800
    assert digest(RAW/'test.csv') == '4ca5834abf4bb4d614b41f05b5b2481b3d41720dd78c1c37b05923c97657dfb8'
    assert digest(RAW/'titles.csv') == 'c97e8da3c6c608a706f2cc0d284aeafb873fc6cc17d04a7c9f9f66b2135846c1'
    assert Counter(int(r['label']) for r in test) == Counter({i:24 for i in range(100)})
    assert Counter(int(r['label']) for r in train) == Counter({i:48 for i in range(100)})
    assert [r['sample_id'] for r in ours] == [r['sample_id'] for r in test]
    qmap = {r['sample_id']:r for r in qwen}
    assert len(qmap) == 2400 and set(qmap) == {r['sample_id'] for r in test}
    qwen = [qmap[r['sample_id']] for r in test]
    y = np.array([int(r['label']) for r in test])
    for source in (ours, qwen):
        assert np.array_equal(y, [int(r['true_label']) for r in source])
        for r in source:
            top = json.loads(r['top10_labels'])
            scores = json.loads(r['top10_scores'])
            assert len(top) == len(scores) == 10 and len(set(top)) == 10
            assert int(r['predicted_label']) == top[0]
            assert all(scores[i] >= scores[i+1]-1e-7 for i in range(9))
            rank = int(r['true_rank'])
            assert (top.index(int(r['true_label']))+1 if int(r['true_label']) in top else 11) == min(rank,11)
    ranks = np.array([[int(r['true_rank']) for r in src] for src in (ours,qwen)])
    pred = np.array([[int(r['predicted_label']) for r in src] for src in (ours,qwen)])
    assert int((ranks[0]==1).sum()) == 1931
    assert int((ranks[1]==1).sum()) == 1769
    assert abs((ranks[0]==1).mean()-report['metrics']['recall_at_1']) < 1e-12
    per_product = []
    for label in range(100):
        take = y == label
        a, b = ranks[:,take]
        per_product.append(dict(label=label, product_id=titles[label]['product_id'], title=titles[label]['title'],
            train_n=48, test_n=24, ours_correct_at_1=int((a==1).sum()), qwen_correct_at_1=int((b==1).sum()),
            ours_hit_at_1=float((a==1).mean()), qwen_hit_at_1=float((b==1).mean()),
            difference_pp=float(100*((a==1).mean()-(b==1).mean())),
            ours_hit_at_5=float((a<=5).mean()), qwen_hit_at_5=float((b<=5).mean()),
            ours_hit_at_10=float((a<=10).mean()), qwen_hit_at_10=float((b<=10).mean()),
            ours_mrr=float((1/a).mean()), qwen_mrr=float((1/b).mean())))
    write_rows(DATA/'per_product_metrics.csv', per_product)
    joined=[]
    all_top10=[]
    for i, sample in enumerate(test):
        joined.append(dict(sample_id=sample['sample_id'], product_id=sample['product_id'], true_label=int(y[i]),
            title=titles[int(y[i])]['title'], image_path=sample['image_path'], spin_id=sample.get('spin_id',''), azimuth=sample.get('azimuth',''),
            ours_rank=int(ranks[0,i]), qwen_rank=int(ranks[1,i]), ours_predicted_label=int(pred[0,i]),
            qwen_predicted_label=int(pred[1,i]), ours_correct=int(ranks[0,i]==1), qwen_correct=int(ranks[1,i]==1)))
        for model,source in (('ours_same_weight_replay',ours),('qwen_dynamic2048',qwen)):
            for position,(lab,score) in enumerate(zip(json.loads(source[i]['top10_labels']),json.loads(source[i]['top10_scores'])),1):
                all_top10.append(dict(model=model,sample_id=sample['sample_id'],true_label=int(y[i]),rank=position,
                    candidate_label=lab,candidate_product_id=titles[lab]['product_id'],candidate_title=titles[lab]['title'],
                    cosine_similarity=score,same_sku=int(lab==y[i])))
    write_rows(DATA/'paired_query_metrics.csv',joined)
    write_rows(DATA/'all_queries_top10_with_titles.csv',all_top10)
    # Resample whole products (24 correlated views kept together), not image rows.
    rng = np.random.default_rng(20261013)
    indices = rng.integers(0,100,size=(10000,100))
    per_k = np.array([[[p[f'{model}_hit_at_{k}'] for p in per_product] for k in (1,5,10)] for model in ('ours','qwen')])
    boots = per_k[:,:,indices].mean(-1)
    ci = np.quantile(boots,[.025,.975],axis=-1)
    diff_boot = boots[0,0] - boots[1,0]
    delta=np.array([p['difference_pp'] for p in per_product])
    sign_rng=np.random.default_rng(20261013)
    perm=(sign_rng.choice([-1,1],size=(20000,100))*delta).mean(1)
    permutation_p=(1+np.sum(np.abs(perm)>=abs(delta.mean())-1e-12))/(len(perm)+1)
    summary = dict(checkpoint_sha256=CHECKPOINT, n_products=100, views_per_product=24, n_queries=2400,
        historical_ours_r1=1934/2400, replay_ours_r1=float((ranks[0]==1).mean()),
        replay_ours_correct=int((ranks[0]==1).sum()), qwen_r1=float((ranks[1]==1).mean()),
        qwen_correct=int((ranks[1]==1).sum()), difference_pp=float(delta.mean()),
        difference_product_bootstrap_ci95_pp=(100*np.quantile(diff_boot,[.025,.975])).tolist(),
        improved_products=int((delta>0).sum()),tied_products=int((delta==0).sum()),worse_products=int((delta<0).sum()),
        exploratory_product_signflip_p=float(permutation_p), perfect_ours_products=int(sum(p['ours_correct_at_1']==24 for p in per_product)),
        qwen_wins_ours_losses=int(((ranks[0]!=1)&(ranks[1]==1)).sum()),
        ours_wins_qwen_losses=int(((ranks[0]==1)&(ranks[1]!=1)).sum()),
        selection='Historical TEST-selected checkpoint. This analysis reuses the same query set.',
        uncertainty='Descriptive catalogue/product bootstrap only; no training-seed variation or correction for model selection.',
        comparison='Ours fixed224 centered preprocessing versus frozen Qwen dynamic2048 system reference; not matched preprocessing ablation.')
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    bootrows=[]
    for m,model in enumerate(('ours_same_weight_replay','qwen_dynamic2048')):
        for j,k in enumerate((1,5,10)):
            bootrows.append(dict(model=model,k=k,hit_rate=float(per_k[m,j].mean()),ci_low=float(ci[0,m,j]),ci_high=float(ci[1,m,j]),n_products=100,n_queries=2400))
    write_rows(DATA/'recall_product_bootstrap.csv',bootrows)
    write_rows(DATA/'bootstrap_top1_difference.csv',(dict(replicate=i,difference_pp=float(d*100)) for i,d in enumerate(diff_boot)))
    conf=np.zeros((2,100,100),dtype=int)
    for m in range(2):
        np.add.at(conf[m],(y,pred[m]),1)
        write_rows(DATA/f'confusion_{("ours","qwen")[m]}_long.csv',
            (dict(true_label=a,predicted_label=b,count=int(conf[m,a,b]),row_fraction=float(conf[m,a,b]/24)) for a in range(100) for b in range(100)))
    assert conf.sum()==4800 and np.trace(conf[0])==1931
    errorpairs=[]
    for m,model in enumerate(('ours','qwen')):
        for a in range(100):
            for b in range(100):
                if a!=b and conf[m,a,b]:
                    errorpairs.append(dict(model=model,true_label=a,true_product_id=titles[a]['product_id'],true_title=titles[a]['title'],
                        predicted_label=b,predicted_product_id=titles[b]['product_id'],predicted_title=titles[b]['title'],count=int(conf[m,a,b])))
    write_rows(DATA/'error_pairs.csv',sorted(errorpairs,key=lambda r:(r['model'],-r['count'],r['true_label'],r['predicted_label'])))
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
        'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
    ax=axes[0,0]; panel(ax,'a')
    for m,(model,color) in enumerate(zip(('LightGen-II (replay)','Frozen Qwen 2048D'),(OUR_COLOR,REF_COLOR))):
        x=np.arange(3)+(m-.5)*.16
        mean=100*per_k[m].mean(1)
        ax.errorbar(x,mean,yerr=np.vstack((mean-ci[0,m]*100,ci[1,m]*100-mean)),fmt='o-',lw=1,ms=4,color=color,label=model,capsize=3)
    ax.set(xticks=range(3),xticklabels=['Hit@1','Hit@5','Hit@10'],ylabel='Query hit rate (%)',ylim=(55,103))
    ax.legend(fontsize=7,loc='lower right');clean_axes(ax)
    ax=axes[0,1];panel(ax,'b')
    a=np.array([p['ours_hit_at_1'] for p in per_product])*100
    b=np.array([p['qwen_hit_at_1'] for p in per_product])*100
    counts=Counter(zip(b,a)); ax.scatter([v[0] for v in counts],[v[1] for v in counts],s=[12+9*n for n in counts.values()],color=OUR_COLOR,alpha=.65,linewidth=.3,edgecolors='white')
    ax.plot([0,100],[0,100],':',color='#777777',lw=.7)
    ax.set(xlabel='Frozen Qwen per-product Hit@1 (%)',ylabel='LightGen-II per-product Hit@1 (%)',xlim=(-5,110),ylim=(-5,110))
    ax.text(.04,.96,f'{summary["improved_products"]} better / {summary["tied_products"]} equal / {summary["worse_products"]} lower',transform=ax.transAxes,va='top',fontsize=7)
    clean_axes(ax)
    ax=axes[1,0];panel(ax,'c')
    ax.hist(delta,bins=np.arange(-102,106,8.333333),color=OUR_COLOR,edgecolor='white',linewidth=.5)
    ax.axvline(0,color='#333333',lw=.8);ax.set(xlabel='Per-product Hit@1 difference (percentage points)',ylabel='Number of products')
    ax.text(.03,.96,f'Mean difference: {delta.mean():.2f} pp\nProduct-bootstrap 95% interval:\n{summary["difference_product_bootstrap_ci95_pp"][0]:.2f} to {summary["difference_product_bootstrap_ci95_pp"][1]:.2f} pp',transform=ax.transAxes,va='top',fontsize=7)
    clean_axes(ax)
    ax=axes[1,1];panel(ax,'d')
    for m,color in enumerate((OUR_COLOR,REF_COLOR)):
        vals=[int(((ranks[m]>=lo)&(ranks[m]<=hi)).sum()) for lo,hi in ((1,1),(2,5),(6,10),(11,20),(21,100))]
        ax.bar(np.arange(5)+(m-.5)*.34,np.array(vals)/24,width=.32,color=color)
    ax.set(xticks=range(5),xticklabels=['1','2–5','6–10','11–20','21–100'],xlabel='Rank of the correct product title',ylabel='Test queries (%)')
    clean_axes(ax);save(fig,'01_paired_performance')
    order=sorted(range(100),key=lambda i:(per_product[i]['ours_hit_at_1'],per_product[i]['qwen_hit_at_1'],i))
    for page in range(2):
        fig,ax=plt.subplots(figsize=(7.2,10.4),layout='constrained')
        ids=order[page*50:(page+1)*50];yy=np.arange(50)
        aa=a[ids];bb=b[ids]
        ax.hlines(yy,np.minimum(aa,bb),np.maximum(aa,bb),color='#bdbdbd',lw=.8)
        ax.scatter(bb,yy,s=12,color=REF_COLOR,label='Frozen Qwen 2048D',zorder=3)
        ax.scatter(aa,yy,s=14,color=OUR_COLOR,label='LightGen-II (replay)',zorder=4)
        names=[f'{i:02d} {titles[i]["product_id"]}' for i in ids]
        ax.set(yticks=yy,yticklabels=names,xlim=(-2,104),xlabel='Per-product Hit@1 (%)',ylim=(-1,50))
        ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7,length=0)
        ax.xaxis.grid(True,color='#eeeeee',lw=.5)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.065),ncol=2,fontsize=8)
        ax.set_title(f'All 100 product classes: products {page*50+1}–{(page+1)*50}\n24 test views per product; sorted by LightGen-II accuracy',loc='left')
        clean_axes(ax);save(fig,f'02_all_products_page{page+1}')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),layout='constrained')
    for m,ax in enumerate(axes):
        im=ax.imshow(conf[m]/24*100,vmin=0,vmax=100,cmap='viridis',interpolation='nearest')
        ax.set(title=('LightGen-II (same-weight replay)','Frozen Qwen 2048D')[m],xlabel='Retrieved product label',ylabel='True product label',xticks=[0,20,40,60,80,99],yticks=[0,20,40,60,80,99])
        panel(ax,'ab'[m])
    fig.colorbar(im,ax=list(axes),label='Views within true product (%)',shrink=.8)
    save(fig,'03_product_confusion')
    fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
    pairs=sorted([r for r in errorpairs if r['model']=='ours'],key=lambda r:(-r['count'],r['true_label'],r['predicted_label']))[:15]
    ax.barh(np.arange(len(pairs)),[r['count'] for r in pairs],color='#B45D53')
    ax.set(yticks=range(len(pairs)),yticklabels=[f'{r["true_label"]:02d} → {r["predicted_label"]:02d}' for r in pairs],xlabel='Number of wrong test views',title='Most frequent product-title confusions (LightGen-II replay)')
    ax.invert_yaxis();clean_axes(ax);save(fig,'04_common_error_pairs')
    case_records=[]
    display_records=[]
    for label,(name,display) in SELECTED.items():
        indices_for=[i for i in range(2400) if y[i]==label]
        first_correct=next((i for i in indices_for if ranks[0,i]==1),indices_for[0])
        sample=test[first_correct];case_dir=CASES/f'{label:02d}_{name}';case_dir.mkdir(exist_ok=True)
        src=RAW/sample['image_path'];original=case_dir/('input_original'+src.suffix.lower());shutil.copy2(src,original)
        top=[r for r in all_top10 if r['sample_id']==sample['sample_id']]
        write_rows(case_dir/'top10_exact_titles.csv',top)
        all_views=[r for r in all_top10 if r['true_label']==label]
        write_rows(case_dir/'all24_views_top10.csv',all_views)
        rec=dict(label=label,product_id=sample['product_id'],sample_id=sample['sample_id'],display_attributes=display,
            original_title=titles[label]['title'],original_image=original.relative_to(HERE).as_posix(),image_sha256=digest(original),
            ours_true_rank=int(ranks[0,first_correct]),qwen_true_rank=int(ranks[1,first_correct]),
            selection='First correct LightGen-II query in the original TEST order; if none, first TEST query.',
            ours_product_correct=per_product[label]['ours_correct_at_1'],qwen_product_correct=per_product[label]['qwen_correct_at_1'])
        case_records.append(rec)
        gallery=[]
        contact,contact_axes=plt.subplots(4,6,figsize=(10.8,7.2),layout='constrained')
        for ia,i in zip(contact_axes.flat,indices_for):
            item=test[i]
            image_path=RAW/item['image_path']
            ia.imshow(ImageOps.exif_transpose(Image.open(image_path)))
            ia.axis('off')
            ia.set_title(f'View {item.get("azimuth","")} | Ours rank {ranks[0,i]} | Qwen {ranks[1,i]}',fontsize=7)
            rel='../../raw/'+item['image_path']
            gallery.append(f'<div><a href="{html.escape(rel)}"><img src="{html.escape(rel)}"></a><p>{item["sample_id"]}<br>view field {item.get("azimuth","")}; Ours rank {ranks[0,i]}; Qwen rank {ranks[1,i]}</p></div>')
        contact.suptitle(f'All 24 original TEST views: product {label:02d} / {sample["product_id"]}',fontsize=11)
        save(contact,f'08_all24_views_{label:02d}_{name}')
        (case_dir/'all24_views.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:14px Arial}.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}img{width:100%;height:210px;object-fit:contain}p{font-size:12px}</style><h1>'+html.escape(titles[label]['title'])+'</h1><p>All 24 TEST views, in original order. Click to view original pixels. Rankings are from the same-weight replay and frozen Qwen.</p><div class="gallery">'+''.join(gallery)+'</div>',encoding='utf-8')
        # One original query, full unmodified top-10 title ranking, short GT attributes.
        fig=plt.figure(figsize=(10.7,8.4),facecolor='white')
        ax=fig.add_axes([.04,.48,.30,.44]);ax.imshow(ImageOps.exif_transpose(Image.open(original)));ax.axis('off')
        ax.set_title('Input test image',fontsize=11)
        fig.text(.04,.40,'Ground-truth title (display summary)',fontsize=10,weight='bold')
        fig.text(.04,.345,textwrap.fill(display,33),fontsize=10,color=OUR_COLOR)
        fig.text(.04,.23,f'Product {label:02d} / {sample["product_id"]}\nSample {sample["sample_id"]}\nLightGen-II true-title rank: {ranks[0,first_correct]}\nFrozen Qwen true-title rank: {ranks[1,first_correct]}',fontsize=8,linespacing=1.7)
        fig.text(.04,.08,'Display text omits dimensions.\nThe ranked candidates retain the\ncomplete original titles.',fontsize=8,color='#555555')
        ra=fig.add_axes([.37,.04,.60,.90]);ra.axis('off')
        ra.text(0,1.02,'LightGen-II: original Top-10 product titles',fontsize=12,weight='bold',transform=ra.transAxes)
        otop=[r for r in top if r['model']=='ours_same_weight_replay']
        for j,r in enumerate(otop):
            yy=.965-j*.094
            correct=bool(r['same_sku']);color=OUR_COLOR if correct else '#444444'
            if correct:
                ra.add_patch(patches.FancyBboxPatch((-.015,yy-.064),1.02,.088,boxstyle='round,pad=.008',facecolor='#EAF5EE',edgecolor=OUR_COLOR,lw=.8,transform=ra.transAxes))
            ra.text(0,yy,f'#{j+1}   {r["candidate_product_id"]}    similarity {r["cosine_similarity"]:.4f}'+('   same SKU' if correct else ''),fontsize=8,weight='bold',color=color,va='top',transform=ra.transAxes)
            ra.text(0,yy-.025,textwrap.fill(r['candidate_title'],88),fontsize=8,color=color,va='top',linespacing=1.1,transform=ra.transAxes)
        save(fig,f'05_case_{label:02d}_{name}')
        show=[]
        seen=set()
        for r in otop:
            attrs=DISPLAY[r['candidate_label']]
            row={**r,'descriptor1':attrs[0],'descriptor2':attrs[1],'item_type':attrs[2],
                 'display_no_dimensions':' · '.join(attrs),'display_source':'Manually excerpted from original title; illustrative text only.'}
            display_records.append(row)
            if attrs[2] not in seen:
                seen.add(attrs[2]);show.append(row)
        write_rows(case_dir/'top10_display_no_dimensions.csv',[r for r in display_records if r['sample_id']==sample['sample_id']])
        write_rows(case_dir/'category_filtered_from_top10.csv',show)
        for mode,ranked in (('06_top4',otop[:4]),('07_category_filtered',show[:4])):
            fig=plt.figure(figsize=(8.7,4.5),facecolor='white')
            ia=fig.add_axes([.03,.25,.40,.64]);ia.imshow(ImageOps.exif_transpose(Image.open(original)));ia.axis('off')
            ia.set_title('Input test image',fontsize=12)
            fig.text(.035,.16,'Ground truth: '+str(sample['product_id']),fontsize=9)
            fig.text(.035,.075,textwrap.fill(display,47),fontsize=10,color=OUR_COLOR)
            ra=fig.add_axes([.47,.04,.50,.85]);ra.axis('off')
            ra.text(0,1.06,'Retrieved product descriptions',fontsize=12,transform=ra.transAxes)
            if mode=='07_category_filtered':
                ra.text(0,1.005,'One per item type; original ranks retained',fontsize=8,color='#555555',transform=ra.transAxes)
            for j,r in enumerate(ranked):
                yy=.94-j*.235
                correct=bool(r['same_sku']);color=OUR_COLOR if correct else '#555555'
                ra.add_patch(patches.FancyBboxPatch((0,yy-.16),1,.18,boxstyle='round,pad=.013',facecolor='#EAF5EE' if correct else '#F4F6F7',edgecolor=OUR_COLOR if correct else '#CBD1D6',lw=1,transform=ra.transAxes))
                ra.text(.035,yy,f'#{r["rank"]}  '+('Same SKU' if correct else 'Different SKU'),fontsize=10,va='top',color=color,transform=ra.transAxes)
                ra.text(.035,yy-.066,textwrap.fill(' · '.join(DISPLAY[r['candidate_label']]),48),fontsize=10,color=color,va='top',transform=ra.transAxes)
            save(fig,f'{mode}_{label:02d}_{name}')
    write_rows(DATA/'recommended_cases.csv',case_records)
    write_rows(DATA/'recommended_case_top10_display_no_dimensions.csv',display_records)
    (DATA/'recommended_cases.json').write_text(json.dumps(case_records,ensure_ascii=False,indent=2),encoding='utf-8')
    # Browser-friendly source-data index; figures and raw inputs have relative links.
    tr=''.join(f'<tr><td>{p["label"]}</td><td>{p["product_id"]}</td><td>{html.escape(p["title"])}</td><td>{p["ours_correct_at_1"]}/24 ({p["ours_hit_at_1"]:.1%})</td><td>{p["qwen_correct_at_1"]}/24 ({p["qwen_hit_at_1"]:.1%})</td></tr>' for p in per_product)
    cs=''.join(f'<li><a href="figures/05_case_{r["label"]:02d}_{SELECTED[r["label"]][0]}.png">{html.escape(r["display_attributes"])}</a> — <a href="{r["original_image"]}">original image</a> — <a href="cases/{r["label"]:02d}_{SELECTED[r["label"]][0]}/top10_exact_titles.csv">Top-10 titles</a> — <a href="cases/{r["label"]:02d}_{SELECTED[r["label"]][0]}/all24_views.html">all 24 original test views</a> — <a href="figures/06_top4_{r["label"]:02d}_{SELECTED[r["label"]][0]}.png">compact Top-4</a> — <a href="figures/07_category_filtered_{r["label"]:02d}_{SELECTED[r["label"]][0]}.png">item-type-filtered view</a></li>' for r in case_records)
    fi=''.join(f'<p><a href="figures/{name}.svg">editable SVG</a> · <a href="figures/{name}.pdf">PDF</a><br><img src="figures/{name}.png" style="width:100%;max-width:1050px"></p>' for name in ('01_paired_performance','02_all_products_page1','02_all_products_page2','03_product_confusion','04_common_error_pairs'))
    explanation=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ABO 图搜文分析与绘图素材</title><style>body{{font:16px/1.65 Arial,"Microsoft YaHei",sans-serif;max-width:1100px;margin:36px auto;padding:0 25px;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}a{{color:#16714f}}h1,h2{{line-height:1.3}}</style>
    <h1>ABO easy100 图搜文：原图、Top-10 排名与逐商品统计</h1>
    <p>权重 SHA：{CHECKPOINT}。数据为 100 个已见商品，每商品 48 TRAIN、24 TEST。同一 spin 序列的不同视角跨划分。无独立 validation；历史训练用周期 TEST 选模。</p>
    <p><b>历史报告：1934/2400 = 80.5833%。本包逐样本证据：同权重 A100 回放 1931/2400 = 80.4583%。</b>二者净差 3 张，未取得历史1934条命中的原始预测；不得把本包逐商品统计标为历史原始1934结果。冻结 Qwen 动态2048D 参照为 {summary['qwen_correct']}/2400 = {summary['qwen_r1']:.4%}。它与 LightGen 固定224输入的预处理不同。</p>
    <h2>推荐图片与实际检索</h2><ul>{cs}</ul>
    <p>每个商品选原TEST顺序中第一张LightGen命中的图片，适合展示正确检索；这属于明确筛选的案例，不能据此推断整体胜率。raw/images 内保存五个商品全部120张TEST原图，cases各目录另有所有24视角的两个模型Top-10。每条排名均含完整标题、商品ID、相似度和是否同SKU。三个非尺寸属性仅是人工摘录的图注，不是模型独立预测属性的结果。05图保留完整原始Top10，06图展示原始Top4的简写，07图按人工item_type在原Top10内去重，保留最先出现者和真实原名次。07为单独展示视图，不参与任何准确率计算，不能标作未经筛选的Top4。</p>
    <h2>统计解释与论文图注</h2>
    <p>Fig.1a：Hit@1/5/10及按商品整组重采样的95%百分位区间（100商品，10000次，seed20261013）。Fig.1b：100商品的配对Top-1比较，重叠点面积代表商品数。Fig.1c：逐商品差值分布。Fig.1d：正确标题的排名分布。该区间描述固定目录内商品异质性，不代表训练seed误差，也不能消除TEST选模偏差。</p>
    <p>Fig.2：全部100个商品的Top-1，按LightGen成绩排序，分两页。24视角中多命中一张改变4.1667个百分点，所以避免把小差值解释为稳定的商品差异。Fig.3：100×100商品混淆，行分母24。Fig.4：最频繁的误检商品对，完整标题在error_pairs.csv。此处类别指SKU，不是地毯、家具等粗物体种类；原coarse_type标签含噪声，未拿它直接分组。</p>
    <p>本次LightGen比Qwen净多命中{summary['replay_ours_correct']-summary['qwen_correct']}张，平均高{summary['difference_pp']:.2f}个百分点；{summary['improved_products']}商品更好、{summary['tied_products']}相同、{summary['worse_products']}更低。差值商品bootstrap区间为{summary['difference_product_bootstrap_ci95_pp'][0]:.2f}至{summary['difference_product_bootstrap_ci95_pp'][1]:.2f}个百分点。配对商品sign-flip p={summary['exploratory_product_signflip_p']:.5g}仅作为探索性描述，不作为独立泛化显著性主张。</p>
    <p>论文中建议表述为“Seen-product image-to-title retrieval on ABO easy100; fixed checkpoint, same-weight simulation replay”。完整原图/排名保留供选图；不要将闭目录同商品新视角识别解释为从图片测得真实物理尺寸。</p>
    <h2>绘图数据与复现</h2><p>raw 为字节原样保存的预测、标题/划分及两份历史报告。processed 为带标题Top-10长表、逐图配对表、100商品指标、混淆长表、错误对、bootstrap输入输出和案例索引。全部数值文件使用UTF-8 CSV/JSON。figures各图同时提供400dpi PNG、可编辑文字SVG和矢量PDF。运行 build_analysis.py 可从已保存输入重画；无网络、模型或GPU要求。依赖：numpy、matplotlib、Pillow。全部文件身份见manifest.json。</p>
    {fi}<h2>全部100商品及逐商品准确率</h2><table><tr><th>Label</th><th>商品ID</th><th>完整标题</th><th>LightGen回放</th><th>Frozen Qwen</th></tr>{tr}</table></html>'''
    (HERE/'index.html').write_text(explanation,encoding='utf-8')
    file_manifest={p.relative_to(HERE).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in HERE.rglob('*') if p.is_file() and p.name not in ('manifest.json','ABO_I2T_Figure_SourceData_20261013.zip') and '__pycache__' not in p.parts}
    (HERE/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=CHECKPOINT,analysis_summary=summary,files=file_manifest),indent=2),encoding='utf-8')
    with zipfile.ZipFile(HERE/'ABO_I2T_Figure_SourceData_20261013.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in HERE.rglob('*'):
            if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts:
                z.write(p,p.relative_to(HERE).as_posix())
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Saved analysis, source data, original images, editable figures and ZIP:', HERE)

if __name__=='__main__':
    main()
