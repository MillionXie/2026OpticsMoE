"""Add the prior six-step candidate to the verified five-model middle delivery.

All128 requests, scores and pixels are bound before showing the sixth model.
Prior-candidate timing is not mixed into the new paired timing comparison.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import re
import shutil
from .scene_acceleration_delivery import build as build_five
from .scene_compression_delivery import verified_models,request_signature,means,degradation,METRICS
from .white_reference_scene_probe import digest,write


def threefold_table(rows):
    reference=next((r for r in rows if r['role']=='qwen'and r['mode']=='software'),None)
    if reference is None or reference['mean_ms']<=0:raise ValueError('Positive electronic reference required')
    baseline=reference['mean_ms'];parts=['<h2>同边界完整请求计时（本轮3倍目标）</h2><table><tr><th>模型</th><th>方式</th><th>均值 ms</th><th>相对电子30步</th><th>保持&gt;3倍的额外设备H上界</th></tr>']
    for row in rows:
        replay=row['mode']=='ideal_CCD_fixed_wait';value=row['mean_ms']
        if value<=0:raise ValueError('Positive request time required')
        row['extra_hardware_H_budget_for3x_ms']=baseline/3-value if replay else None
        row['above3x_before_unknown_H']=baseline/value>3 if replay else None
        budget=f"{row['extra_hardware_H_budget_for3x_ms']:.3f} ms"if replay else'—'
        parts.append('<tr><th>'+html.escape(row['label'])+'</th><td>'+('无FFT＋固定6光周期'if replay else'完整电子软件请求')+'</td><td>'+f'{value:.3f}'+'</td><td>'+f'{baseline/value:.4f}×'+'</td><td>'+budget+'</td></tr>')
    parts.append('</table><p>本表目标为超过3倍；H必须小于对应预算。完整请求理想CCD时序模拟与实机结果分列，未知SDK／SLM传输／额外DMA没有假定为零。</p>')
    return ''.join(parts)


def add_table_row(content,heading,row):
    pattern='('+re.escape(heading)+r'<table>.*?)(</table>)'
    result,n=re.subn(pattern,lambda m:m.group(1)+row+m.group(2),content,count=1,flags=re.S)
    if n!=1:raise ValueError('Expected one bound summary table')
    return result


def build(manifest,output):
    prior_spec=manifest['prior_light_candidate']
    if prior_spec['role']!='light_lg':raise ValueError('Prior candidate must pass the same lightweight verifier')
    alternate=dict(manifest,models=[prior_spec if m['role']=='light_lg'else m for m in manifest['models']])
    base_models=verified_models(manifest);prior=next(m for m in verified_models(alternate)if m['role']=='light_lg')
    current=base_models[0];electronic=next(m for m in base_models if m['role']=='qwen')
    if prior['report']['sampling_steps']!=6:raise ValueError('Explicit previous six-step comparison required')
    if prior['report']['checkpoint_sha256']==current['report']['checkpoint_sha256']:raise ValueError('Middle and prior checkpoints must differ')
    keys=sorted(current['images'])
    if {k:request_signature(v)for k,v in prior['images'].items()}!={k:request_signature(v)for k,v in current['images'].items()}:
        raise ValueError('Middle and prior inputs/prompts/noise differ')
    build_five(manifest,output)
    guard=json.loads((output/'delivery_manifest.json').read_text());guard.update(complete=False,six_model_augmentation_in_progress=True)
    write(output/'delivery_manifest.json',guard)
    page=output/'index.html';content=page.read_text(encoding='utf8');e=html.escape
    original=next(m for m in base_models if m['role']=='original_lg');anchor='<figure><figcaption>'+e(original['label'])+'</figcaption>'
    index=0
    def article(match):
        nonlocal index
        part=match.group(0);key=keys[index]
        if '<h3>'+e(key)+'</h3>'not in part or part.count(anchor)!=1:raise ValueError('Article order or figure identity changed')
        row=prior['images'][key];name=f'{index:03d}_prior_light_lg.png'
        source=prior['root']/row['generated_PNG']
        if digest(source)!=row['generated_sha256']:raise ValueError('Prior PNG changed')
        shutil.copyfile(source,output/'images'/name)
        score=prior['scores'][key];caption=' · '.join(k+' '+f'{score[k]:.3f}'for k in METRICS)
        figure='<figure><figcaption>'+e(prior['label'])+'</figcaption><img src="images/'+name+'" alt="'+e(prior['label'])+'" loading="lazy"><small>'+e(caption)+'</small></figure>'
        index+=1;return part.replace(anchor,figure+anchor,1)
    content,n=re.subn(r'<article\b.*?</article>',article,content,flags=re.S)
    if n!=128 or index!=128:raise ValueError('All128 articles required')
    additional=[]
    for category,title in [('all','全部128请求'),('lamp','台灯64请求'),('pillow','靠垫64请求')]:
        values=means(prior,category);drops={k:degradation(v,means(electronic,category)[k])for k,v in values.items()}
        row='<tr><th>'+e(prior['label'])+'</th>'+''.join('<td>'+f'{values[k]:.4f}'+'</td>'for k in METRICS)+f'<td>{max(v for v in drops.values()if v is not None):+.2f}%</td></tr>'
        content=add_table_row(content,'<h3>'+title+'</h3>',row)
        additional.append(dict(model=prior['label'],role='prior_light_lg',category=category,n=128 if category=='all'else 64,**values,**{k+'_drop_percent':v for k,v in drops.items()}))
    r=prior['report'];params=r.get('parameters')
    if params is None and all(k in r for k in ['generator_loaded_parameters','Qwen_parameters','head_parameters']):params=sum(r[k]for k in ['generator_loaded_parameters','Qwen_parameters','head_parameters'])
    row='<tr><th>'+e(prior['label'])+'</th><td>6</td><td>'+e(str(params)if params is not None else'未记录')+'</td><td>'+e(prior['training_note'])+'</td><td><code>'+e(r['checkpoint_sha256'])+'</code></td></tr>'
    content=add_table_row(content,'<h2>模型与预算</h2>',row)
    content=content.replace('repeat(6,minmax(220px,1fr))','repeat(7,minmax(220px,1fr))')
    content=content.replace('<h1>LG轻量化与生成模型：同题完整对照</h1>','<h1>LG中间版、此前六步版与生成模型：完整同题对照</h1>')
    content=content.replace('<h2>全部128组输出</h2>','<p>每组依次为白底输入、中间版、此前六步版、原F9、完整Qwen、SDXL＋IP、OmniGen；共768张生成图，全部保留。三倍判定使用中间版与E30的新配对计时，未挪用旧六步版本的其他计时窗口。</p><h2>全部128组输出</h2>')
    timing_path=output/'timing.json';timing=json.loads(timing_path.read_text());table=threefold_table(timing['rows'])
    content,n=re.subn(r'<h2>同边界完整请求计时</h2><table>.*?</table>',lambda _:table,content,count=1,flags=re.S)
    if n!=1:raise ValueError('One verified timing table required')
    content=content.replace('只有H不超过表中预算时，这一时序模型才仍达到四倍','只有H小于表中预算时，这一时序模型才仍超过三倍')
    timing.update(requested_speedup_above=3.,physical_measurement=False);write(timing_path,timing)
    page.write_text(content,encoding='utf8')
    csv_path=output/'summary.csv'
    with csv_path.open(encoding='utf-8-sig',newline='')as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    with csv_path.open('w',encoding='utf-8-sig',newline='')as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows+additional)
    score_path=output/'per_image.json';scores=json.loads(score_path.read_text());scores.extend(dict(v,display_model=prior['label'])for v in prior['scores'].values());assert len(scores)==768;write(score_path,scores)
    record=json.loads((output/'delivery_manifest.json').read_text());record.update(complete=True,six_model_augmentation_in_progress=False,requested_speedup_above=3.,all768_generated_outputs_shown=True,generated_model_count=6,middle_builder_source_sha256=digest(Path(__file__)),prior_candidate=dict(role='prior_light_lg',label=prior['label'],run=str(prior['root']),report_sha256=digest(prior['root']/'report.json'),metrics_sha256=prior['metrics_manifest_sha256']))
    write(output/'delivery_manifest.json',record)
    write(output/'file_manifest.json',{p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))for p in sorted(output.rglob('*'))if p.is_file()and p.name!='file_manifest.json'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(json.loads(a.manifest.read_text()),a.output)


if __name__=='__main__':main()
