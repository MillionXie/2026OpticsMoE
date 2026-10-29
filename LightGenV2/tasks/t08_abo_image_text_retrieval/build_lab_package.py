"""Build a full TEST/100-query drawing ZIP, never modify or synthesize photos.

New A100 DeepSeek rankings and historical comparison rankings stay separately
labelled. All100 queries included; the old hand-picked ten are not a test set.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from urllib.parse import quote
import zipfile
from . import deepseek_t2i_a100_20261029 as task


def normalized_rankings(titles,rows,reference,new_predictions):
    by_id={r['sample_id']:r for r in rows}
    ours=task.read_csv(reference/'ours_10cm_predictions.csv')
    qwen=json.loads((reference/'qwen_dynamic_2048_predictions.json').read_text(encoding='utf8'))
    clip=json.loads((reference/'clip_predictions.json').read_text(encoding='utf8'))['text_to_image_top10']
    if not all(len(v)==100 for v in (ours,qwen,clip,new_predictions)):raise ValueError('All100 query rankings required')
    result=[]
    for i,title in enumerate(titles):
        if ours[i]['product_id']!=title['product_id'] or qwen[i]['product_id']!=title['product_id'] or new_predictions[i]['product_id']!=title['product_id']:raise ValueError('Query SKU/order mismatch')
        models={
            'LightGen-II_historical_simulation':json.loads(ours[i]['top10_sample_ids']),
            'Qwen3-VL_historical_frozen_2048D':qwen[i]['top10_sample_ids'],
            'DeepSeek-VL2-Tiny_new_A100': [r['sample_id'] for r in new_predictions[i]['top10']],
            'CLIP_historical_frozen_ViT-B32':clip[i],
        }
        values={}
        for model,ids in models.items():
            if len(ids)!=10 or len(set(ids))!=10 or any(v not in by_id for v in ids):raise ValueError('Exact original Top10 IDs, no invented/missing image')
            values[model]=[dict(rank=k+1,sample_id=v,product_id=by_id[v]['product_id'],image_path=task.checked_relative(by_id[v]['image_path']),correct=by_id[v]['product_id']==title['product_id']) for k,v in enumerate(ids)]
        result.append(dict(query_index=i,product_id=title['product_id'],title=title['title'],models=values))
    return result


def build(a):
    if a.output.exists() or a.zip.exists():raise FileExistsError('New immutable drawing package only')
    manifest=json.loads(a.input_manifest.read_text(encoding='utf8'));titles,rows=task.verify_inputs(a.data,manifest)
    audit=task.audit_saved_run(a.run)
    results=normalized_rankings(titles,rows,a.reference,json.loads((a.run/'predictions.json').read_text(encoding='utf8')))
    a.output.mkdir(parents=True);dataset=a.output/'dataset';dataset.mkdir()
    for name in ('test.csv','titles.csv'):shutil.copyfile(a.data/name,dataset/name)
    for r in manifest['files']:
        dest=dataset/task.checked_relative(r['path']);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.data/r['path'],dest)
        if task.sha(dest)!=r['sha256']:raise ValueError('Original photo copy differs')
    sources=a.output/'source';sources.mkdir()
    for name in ('report.json','predictions.json','process_audit.json','status_events.jsonl'):
        shutil.copyfile(a.run/name,sources/('new_DS_'+name))
    for name in ('ours_10cm_predictions.csv','ours_10cm_final_report.json','ours_10cm_run_manifest.json','qwen_dynamic_2048_predictions.json','qwen_report.json','clip_predictions.json','clip_report.json'):
        shutil.copyfile(a.reference/name,sources/('historical_'+name))
    task.write(sources/'new_DS_independent_audit.json',audit)
    task.write(a.output/'queries_and_top10.json',results)
    links=[];flat=[]
    css='body{font:16px Arial;margin:24px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd;vertical-align:top}img{width:180px;height:180px;object-fit:contain}.ok{color:#087c36}.bad{color:#ad2929}'
    for query in results:
        key=f"{query['query_index']:03d}_{query['product_id']}";folder=a.output/'queries'/key;folder.mkdir(parents=True)
        (folder/'query.txt').write_text(query['title']+'\n',encoding='utf8')
        cards=[];query_rows=[]
        for model,ranking in query['models'].items():
            photos=[]
            for r in ranking:
                row=dict(query_index=query['query_index'],query_product_id=query['product_id'],query_text=query['title'],model=model,**r,package_image='dataset/'+r['image_path'])
                flat.append(row);query_rows.append(row)
                if r['rank']<=3:
                    url='../../dataset/'+quote(r['image_path'],safe='/')
                    photos.append('<div><img src="'+url+'"><p class="'+('ok' if r['correct'] else 'bad')+'">Top '+str(r['rank'])+' | '+html.escape(r['product_id'])+' | '+('same SKU' if r['correct'] else 'different SKU')+'</p></div>')
            cards.append('<td><h3>'+html.escape(model)+'</h3>'+''.join(photos)+'</td>')
        with (folder/'top10.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(query_rows[0]));w.writeheader();w.writerows(query_rows)
        (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style><h1>'+html.escape(query['title'])+'</h1><p>Full TEST: 2400 images, 24 exact-SKU positives. New DS A100; other columns historical, not remeasured.</p><table><tr>'+''.join(cards)+'</tr></table>',encoding='utf8')
        links.append('<li><a href="queries/'+key+'/index.html">'+str(query['query_index'])+' | '+html.escape(query['title'])+'</a></li>')
    with (a.output/'all_queries_top10.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style><h1>ABO text-to-image: all100 queries</h1><p>Illustrative categories in the supplied figure: 16 step stool, 20 speaker mount, 99 dining table. These are examples, not an unbiased estimate. Overall metrics use all100 queries.</p><ol>'+''.join(links)+'</ol>',encoding='utf8')
    text='''# Drawing handoff / 画图素材说明

打开 index.html，可按100条官方标题查四组Top3；各查询目录另有完整标题query.txt和Top10 CSV。
dataset/包含全部2400张原TEST JPG字节，以及原test.csv/titles.csv，不含TRAIN图片、模型权重或凭据。
all_queries_top10.csv/queries_and_top10.json提供4000项可追溯检索记录；正确性只按同商品SKU，不按视觉相似。
DeepSeek-VL2-Tiny是本次A100冻结零训练推理；LightGen-II仿真.86、Qwen动态2048D、CLIP为已保存历史比较。
不把其他三列冒称本次A100重测，也不把LightGen仿真.86混成实拍.85。
用户示意图的三个类别可看16/20/99；旧十例原有选择偏倚，不能代表整体胜率。此包保留全部100查询。
指标、版本和源文件身份见source/。manifest.json与ZIP旁SHA用于交接校验。原图未裁剪、去背景、重编码或生成。
本数据是同一100个已登记商品的不同视图，不是未见商品泛化；历史LightGen使用过TEST选模。
'''
    (a.output/'README.txt').write_text(text,encoding='utf8')
    files=[dict(path=p.relative_to(a.output).as_posix(),bytes=p.stat().st_size,sha256=task.sha(p)) for p in sorted(a.output.rglob('*')) if p.is_file()]
    task.write(a.output/'manifest.json',dict(status='full_TEST_drawing_handoff',query_count=100,gallery_count=2400,new_DS_report_sha256=audit['report_sha256'],files=files))
    with zipfile.ZipFile(a.zip,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(a.output.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(a.output).as_posix())
    with zipfile.ZipFile(a.zip) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        for r in files:
            if hashlib.sha256(z.read(r['path'])).hexdigest()!=r['sha256']:raise ValueError('ZIP original byte SHA differs')
    digest=task.sha(a.zip)
    with a.zip.with_suffix(a.zip.suffix+'.sha256').open('x') as f:f.write(digest+'  '+a.zip.name+'\n')
    print(json.dumps(dict(status='verified_full_TEST_drawing_ZIP',zip=str(a.zip),bytes=a.zip.stat().st_size,sha256=digest),ensure_ascii=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','input-manifest','run','reference','output','zip'):p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())


if __name__=='__main__':main()
