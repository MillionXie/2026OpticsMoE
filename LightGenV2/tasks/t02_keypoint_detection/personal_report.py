"""Show EVERY held-out personal image, before/after, without selecting attractive cases."""
import argparse,html,json,os
from pathlib import Path
from PIL import Image,ImageDraw
from .personal_split import crop_box,EDGES


def run(dataset,ours,baseline,output=None):
    dataset=Path(dataset);ours=Path(ours);baseline=Path(baseline)
    output=Path(output) if output else dataset
    output.mkdir(parents=True,exist_ok=True)
    m=json.loads((dataset/'annotations_provisional.json').read_text(encoding='utf-8'))
    manifests=[json.loads((p/'run_manifest.json').read_text()) for p in [ours,baseline]]
    splits=[v['data'].get('photo_ids') for v in manifests]
    if splits[0]!=splits[1]:raise ValueError('Different evaluation splits')
    test_ids=set(splits[0]['test']) if splits[0] else {r['id'] for r in m['images'] if r['split']=='test'}
    panels=output/'comparisons';panels.mkdir(exist_ok=False)
    reports={k:json.loads((p/'final_report.json').read_text()) for k,p in [('ours',ours),('baseline',baseline)]}
    baseline_frozen=reports['baseline'].get('training_performed') is False
    if baseline_frozen:
        r=reports['baseline'];r.update(before=r['metrics'],after=r['metrics'],best_epoch=None,fusion=None)
    rows=[]
    for row in m['images']:
        if row['id'] not in test_ids:continue
        for person in row['people']:
            if not person['include']:continue
            ident=row['id']+'_'+person['id']
            pts=[v if j<12 else [v[0],v[1],0] for j,v in enumerate(person['keypoints'])]
            box=crop_box(pts);im=Image.open(dataset/row['image']).convert('RGB').crop(box).resize((224,224),Image.Resampling.BILINEAR)
            label=im.copy();d=ImageDraw.Draw(label)
            q=[[(v[0]-box[0])*224/(box[2]-box[0]),(v[1]-box[1])*224/(box[3]-box[1]),v[2]] for v in pts]
            for a,b in EDGES:
                if q[a][2] and q[b][2]:d.line([tuple(q[a][:2]),tuple(q[b][:2])],fill='#1aff75',width=2)
            parts=[im,label]+[Image.open(p/f'{"before" if p==baseline and baseline_frozen else stage}_images/{ident}.png').convert('RGB') for p in [ours,baseline] for stage in ['before','after']]
            sheet=Image.new('RGB',(224*6,248),'white');ds=ImageDraw.Draw(sheet)
            for i,(im,title) in enumerate(zip(parts,['Input','RTMPose proposal (NOT GT)','Ours before','Ours after','Qwen before','Qwen after'])):
                sheet.paste(im,(i*224,24));ds.text((i*224+3,5),title,fill='black')
            sheet.save(panels/f'{ident}.jpg',quality=95)
            rows.append(f'<h3>{html.escape(ident)} · {html.escape(row["group_id"])}</h3><img src="comparisons/{ident}.jpg">')
    metrics={k:{'before_pseudo_pck12':r['before']['pck_at_0.2_torso'],'after_pseudo_pck12':r['after']['pck_at_0.2_torso'],
                'best_epoch':r['best_epoch'],'fusion':r['fusion'],'formal_gt_evaluation':r['formal_ground_truth_evaluation']} for k,r in reports.items()}
    (output/'pilot_comparison.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>自采姿态迁移试跑</title><style>body{font:16px system-ui;margin:24px;background:#f4f6fa}img{width:100%;max-width:1450px}p{max-width:1000px;line-height:1.6}pre{background:white;padding:16px}</style><h1>Ours / Qwen 迁移前后 · 全部20张测试图</h1><p>注意：绿色仅为独立自动预标注，不是人工真值。试跑只监督和评价12个四肢点，红色预测仍显示模型全部14点。所有图片按固定文件顺序展示，没有挑选高分图片。按5个拍摄组留出的20张照片存在组内相关性，不代表20个独立场景。正式论文结果须先完成标注核对。</p><p><a href="review.html">打开标注审核页</a></p><pre>'+html.escape(json.dumps(metrics,indent=2,ensure_ascii=False))+'</pre>'+''.join(rows)
    page=page.replace('全部20张测试图',f'全部{len(test_ids)}张测试图').replace('按5个拍摄组留出的20张照片存在组内相关性，不代表20个独立场景。','按拍摄组划分；同组照片相关，不等同于独立场景。').replace('href="review.html"',f'href="{html.escape(os.path.relpath(dataset / "review.html", output).replace(chr(92), "/"))}"')
    if baseline_frozen:page=page.replace('<h1>','<p>Qwen 未微调：最后两列是同一原始模型结果，不代表微调前后。Ours只适配姿态头。用户筛选后的探索性数据集，不能替代原全集性能。</p><h1>',1)
    (output/'COMPARE.html').write_text(page,encoding='utf-8');print(json.dumps(metrics))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--ours',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.dataset,a.ours,a.baseline,a.output)
