"""Matched five-model HTML delivery; verifies every image, request and score first."""
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
import statistics
from .white_reference_metrics import load_records
from .white_reference_scene_probe import digest,write

METRICS=('CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact')
ROLES=('light_lg','original_lg','qwen','native','peer')


def request_signature(row):
    return tuple(row[k] for k in ('input_sha256','prompt','seed','initial_latent_sha256','category','condition_id'))


def verified_models(manifest):
    models=manifest['models']
    if len(models)!=5 or {m['role'] for m in models}!=set(ROLES):
        raise ValueError('All five prespecified model roles required, exactly once')
    result=[];signature=None;metric_protocol=None
    for role in ROLES:
        spec=next(m for m in models if m['role']==role);root=Path(spec['run']);metric_root=Path(spec['metrics'])
        report=json.loads((root/'report.json').read_text());metric_report=json.loads((metric_root/'report.json').read_text())
        if (report.get('scope')!='DEV_expanded' or report['prompt_profile']!='two_sentence_natural_room_v4'
            or report['resolution']!=512 or not metric_report['complete']):
            raise ValueError('Only completed matched natural DEV128 at512; no TEST/old prompts')
        if metric_protocol is None:metric_protocol=metric_report['metric_protocol']
        elif metric_report['metric_protocol']!=metric_protocol:raise ValueError('Different metric definitions')
        if role=='light_lg' and not report.get('compression_training_performed'):
            raise ValueError('Lightweight result requires completed recovery, not the zero-shot cut')
        rows=list(load_records(root));images={r['sample_id']:r for r in rows}
        if len(images)!=128:raise ValueError('Incomplete image denominator')
        current={k:request_signature(v) for k,v in images.items()}
        if signature is None:signature=current
        elif current!=signature:raise ValueError('RGB, full prompt, seed or initial noise differ between models')
        scores=[r for r in json.loads((metric_root/'per_image.json').read_text()) if r['run_id']==root.name]
        if len(scores)!=128 or {r['sample_id'] for r in scores}!=set(images):raise ValueError('Incomplete score denominator')
        indexed={r['sample_id']:r for r in scores}
        for key,row in images.items():
            scored=indexed[key]
            if scored['generated_sha256']!=row['generated_sha256'] or request_signature(scored)!=request_signature(row):
                raise ValueError('Scores are not bound to the displayed generated pixels')
            if not all(math.isfinite(float(scored[k])) for k in METRICS):raise ValueError('Invalid score')
        result.append(dict(spec,root=root,report=report,images=images,scores=indexed,
            metrics_manifest_sha256=digest(metric_root/'per_image.json')))
    return result


def means(model,category='all'):
    rows=[r for r in model['scores'].values() if category=='all' or r['category']==category]
    return {key:statistics.mean(float(r[key]) for r in rows) for key in METRICS}


def degradation(actual,reference):
    return None if abs(reference)<1e-6 else (reference-actual)/abs(reference)*100


def build(manifest,output):
    models=verified_models(manifest)
    if output.exists():raise ValueError('New delivery directory required')
    output.mkdir(parents=True);asset=output/'images';asset.mkdir();e=html.escape
    source=models[0];keys=sorted(source['images']);summaries=[];sections=[]
    for category in ('all','lamp','pillow'):
        baseline=means(next(m for m in models if m['role']=='qwen'),category);table=[]
        for m in models:
            values=means(m,category);drops={k:degradation(v,baseline[k]) for k,v in values.items()}
            summaries.append(dict(model=m['label'],role=m['role'],category=category,n=128 if category=='all' else 64,
                **values,**{k+'_drop_percent':v for k,v in drops.items()}))
            cells=''.join('<td>'+f'{values[k]:.4f}'+'</td>' for k in METRICS)
            worst=max(v for v in drops.values() if v is not None)
            table.append('<tr><th>'+e(m['label'])+'</th>'+cells+f'<td>{worst:+.2f}%</td></tr>')
        sections.append('<h3>'+{'all':'全部128请求','lamp':'台灯64请求','pillow':'靠垫64请求'}[category]+'</h3><table><tr><th>模型</th>'+
            ''.join('<th>'+e(k)+'</th>' for k in METRICS)+'<th>相对Qwen最大下降</th></tr>'+''.join(table)+'</table>')
    identities=[]
    for m in models:
        r=m['report'];params=r.get('parameters')
        if params is None and all(k in r for k in ['generator_loaded_parameters','Qwen_parameters','head_parameters']):
            params=sum(r[k] for k in ['generator_loaded_parameters','Qwen_parameters','head_parameters'])
        identities.append('<tr><th>'+e(m['label'])+'</th><td>'+str(r['sampling_steps'])+'</td><td>'+e(str(params)if params is not None else'未记录')+
            '</td><td>'+e(m['training_note'])+'</td><td><code>'+e(r.get('checkpoint_sha256',r.get('official_weights_revision','')))+'</code></td></tr>')
    articles=[]
    for index,key in enumerate(keys):
        row=source['images'][key];name=f'{index:03d}_reference.png'
        shutil.copyfile(source['root']/(row['reference_id']+'.png'),asset/name)
        figures=['<figure><figcaption>白底输入</figcaption><img src="images/'+name+'" alt="reference" loading="lazy"></figure>']
        for m in models:
            entry=m['images'][key];name=f'{index:03d}_{m["role"]}.png';shutil.copyfile(m['root']/entry['generated_PNG'],asset/name)
            score=m['scores'][key];caption=' · '.join(k+' '+f'{score[k]:.3f}' for k in METRICS)
            figures.append('<figure><figcaption>'+e(m['label'])+'</figcaption><img src="images/'+name+'" alt="'+e(m['label'])+
                '" loading="lazy"><small>'+e(caption)+'</small></figure>')
        articles.append('<article data-category="'+e(row['category'])+'"><h3>'+e(key)+'</h3><p class="prompt">'+e(row['prompt'])+
            '</p><small>seed '+str(row['seed'])+' · '+e(row['reference_id'])+'</small><div class="grid">'+''.join(figures)+'</div></article>')
    timing=manifest.get('timing_note','本轮正式计时尚未完成；不把旧模型速度或参数降幅当成新候选实测。')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LG轻量化与五模型同题对照</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#202936;background:#f6f7fa;margin:0;padding:26px}h1{font-size:26px}h2{margin-top:30px}h3{font-size:17px}p{max-width:1200px}table{border-collapse:collapse;background:white;margin:12px 0;width:100%}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #dce0e8}code{font-size:11px;word-break:break-all}.notice{background:#fff4da;border-left:4px solid #c48d22;padding:14px}.grid{display:grid;grid-template-columns:repeat(6,minmax(220px,1fr));gap:10px;overflow:auto}figure{margin:0;min-width:220px}img{width:100%;height:auto;display:block}figcaption{font-weight:650;padding:6px 0}small{display:block;color:#5f6876;font-size:12px}.prompt{font-size:17px;font-weight:550;margin:8px 0}article{background:white;margin:22px 0;padding:16px;border-radius:8px}select{padding:8px}summary{cursor:pointer;font-weight:650}a{color:#245bb2}
</style><h1>LG轻量化与生成模型：同题完整对照</h1>
<p>同一白底参考图、两句自然提示词、同一源噪声和512输出。128个DEV请求全部展示，16个参考视图、8款商品、每视图8场景；这是开发集对照，未重跑或重选冻结TEST500。</p>
<div class="notice">CLIP-T、DINO-I、CLIP-I与ImageReward均为自动代理分数，越高越好，不能等同于逐属性成功率。ImageReward可为负数，相对下降采用(Qwen−模型)/|Qwen|；保留失败图，不挑最好结果。不同模型的任务微调与采样预算在下表分别注明。</div>
<h2>整体及分类均值</h2>'''+''.join(sections)+'''<h2>模型与预算</h2><table><tr><th>模型</th><th>采样步数</th><th>实际推理参数</th><th>训练范围</th><th>权重身份</th></tr>'''+''.join(identities)+'''</table>
<h2>耗时口径</h2><p>'''+e(timing)+'''</p><p>光电模型须去除FFT传播计算并换入每请求固定6周期SLM＋CCD，共5.165536ms；Qwen仅前向一次，不乘UNet采样步数。软件完整请求、光电模型估算与现场实测分列，未知通信成本不假定为零。</p>
<details><summary>逐样本指标与可追溯数据</summary><p><a href="summary.csv">均值CSV</a> · <a href="per_image.json">逐图分数JSON</a> · <a href="delivery_manifest.json">模型、输入与评分身份</a></p></details>
<h2>全部128组输出</h2><label>类别 <select id="category"><option value="all">全部</option><option value="lamp">台灯</option><option value="pillow">靠垫</option></select></label>'''+''.join(articles)+'''
<script>document.getElementById('category').onchange=function(){document.querySelectorAll('article').forEach(x=>x.hidden=this.value!=='all'&&x.dataset.category!==this.value)}</script></html>'''
    (output/'index.html').write_text(page,encoding='utf8')
    with(output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    write(output/'per_image.json',[dict(r,display_model=m['label']) for m in models for r in m['scores'].values()])
    write(output/'delivery_manifest.json',dict(complete=True,scope='matched_DEV128',contexts=128,models=[dict(role=m['role'],label=m['label'],run=str(m['root']),report_sha256=digest(m['root']/'report.json'),metrics_sha256=m['metrics_manifest_sha256'])for m in models],
        all640_outputs_shown=True,no_quality_selection=True,independent_TEST_opened=False,raw_data_manifest=manifest,
        HTML_builder_source_sha256=digest(Path(__file__)),caller_source_commit=os.environ.get('T12_SOURCE_COMMIT'),
        caller_commit_is_not_a_claim_all_components_equal_that_commit=True))
    write(output/'file_manifest.json',{p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))for p in sorted(output.rglob('*'))if p.is_file()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();build(json.loads(a.manifest.read_text()),a.output)


if __name__=='__main__':main()
