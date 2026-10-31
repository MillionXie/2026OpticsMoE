"""Same-request HTML with per-image metrics, complete means and honest timing.

Rendering this page is not manual image acceptance or optical deployment.
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil
import statistics
from .scene_instruction_data import sha
from .scene_paper_prompt_protocol import write

METRICS=['CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact']


def timing(rounds):
    all_ms=[];identity=True
    for folder in rounds:
        r=json.loads((folder/'software_latency.json').read_text())
        if(r['n']!=8 or r['batch_size']!=1 or r['extra_warmup'] or not r['output_PNG_identity_vs_completed_quality_run']):
            raise ValueError('Three actual independent-process8-request timing rounds required')
        all_ms.extend(r['requests_ms']);identity=identity and r['output_PNG_identity_vs_completed_quality_run']
    if len(rounds)!=3 or len(all_ms)!=24:raise ValueError('Exactly3 processes by8 requests required')
    return dict(mean_ms=statistics.mean(all_ms),n=24,processes=3,PNG_identity_checked=identity)


def build(runs,labels,metrics,e_rounds,lg_rounds,workflow,output):
    if output.exists() or len(runs)!=len(labels):raise ValueError('New same-cohort delivery required')
    m=json.loads((metrics/'report.json').read_text());scores=json.loads((metrics/'per_image.json').read_text())
    summary=json.loads((metrics/'summary.json').read_text())
    if not m['complete'] or not m['TEST_used_for_model_selection']:raise ValueError('Full TEST-selection metrics required')
    reports=[json.loads((r/'report.json').read_text()) for r in runs]
    manifest={r['TEST_selection_protocol']['request_sha256'] for r in reports}
    if len(manifest)!=1 or any(r['generated_count']!=500 or r['scope']!='TEST_selection' for r in reports):
        raise ValueError('All models must have the same complete TEST500')
    rows={run.name:{r['sample_id']:r for r in scores if r['run_id']==run.name} for run in runs}
    if any(len(r)!=500 for r in rows.values()):raise ValueError('Missing scored outputs')
    common=set(rows[runs[0].name])
    if any(set(r)!=common for r in rows.values()):raise ValueError('Different scored requests')
    E=timing(e_rounds);LG=timing(lg_rounds);speed=E['mean_ms']/LG['mean_ms']
    occupancy=[json.loads(p.read_text()) for p in workflow.glob('*_GPU_ownership.json')]
    uncontended=bool(occupancy) and not any(v['foreign_observed'] for v in occupancy)
    e_name=runs[1].name;lg_name=runs[0].name;gates=[]
    for category in ['all','lamp','pillow']:
        for metric in METRICS:
            e=summary[e_name+'/'+category]['scores'][metric]['mean'];l=summary[lg_name+'/'+category]['scores'][metric]['mean']
            near_zero=abs(e)<.05;drop=None if near_zero else (e-l)/abs(e)
            gates.append(dict(category=category,metric=metric,baseline=e,LG=l,absolute_difference=l-e,
                              relative_drop=drop,passes_10_percent=None if near_zero else drop<=.1))
    gate=dict(quality_proxy_gates=gates,all12_proxy_gates_pass=all(r['passes_10_percent']is True for r in gates),
        E30=E,LG8=LG,speedup=speed,modeled_speed_pass=speed>=3 and uncontended,foreign_GPU_observed=not uncontended,
        fixed_six_optical_ms=5.1655361611877,additional_unmeasured_H_ms_budget_for_3x=E['mean_ms']/3-LG['mean_ms'],
        unknown_real_hardware_transport_DMA_not_assumed_zero=True,physical_end_to_end_measurement=False,
        manual_visual_review_pending=True,hardware_handoff_approved=False,TEST_used_for_selection=True)
    output.mkdir(parents=True);(output/'images').mkdir();write(output/'acceptance.json',gate)
    write(output/'plot_data.json',dict(summary=summary,acceptance=gate))
    esc=lambda x:html.escape(str(x));parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>商品场景：同题模型对比</title>',
        '<style>body{font:15px system-ui;margin:24px;color:#19232f}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #d6dce3;text-align:left}img{width:100%;max-width:360px;aspect-ratio:1;object-fit:contain}.grid{display:grid;gap:12px;grid-template-columns:repeat('+str(len(runs)+1)+',minmax(0,1fr))}article{margin:28px 0;padding-top:16px;border-top:2px solid #d6dce3}.note{background:#edf2f7;padding:16px}.metric{font-size:12px;line-height:1.6}h2{font-size:19px}</style>',
        '<h1>白底商品＋两句提示词 → 512场景图</h1>',
        '<p class="note">完整500个固定请求；灯／靠垫各250，来自18款26视图。每请求一个种子。无validation，TEST参与PT选择，结果不作独立泛化估计。CLIP／DINO／ImageReward是代理指标，不等于全部指令成功率。失败图保留。</p>',
        '<p>电子30步平均 '+f'{E["mean_ms"]:.2f}'+' ms；中间LG8步平均 '+f'{LG["mean_ms"]:.2f}'+' ms；比值 <b>'+f'{speed:.3f}×</b>。RGB＋文字就绪到CPU RGB，3进程×8请求，含首请求；FFT已移除并换成6次固定光周期共5.166ms。未知现场传输／DMA等额外H未测，不是物理整机实测。额外H不超过 '+f'{gate["additional_unmeasured_H_ms_budget_for_3x"]:.2f}'+' ms时才仍满足3倍。</p>',
        '<table><tr><th>模型／类别</th><th>n</th>'+''.join('<th>'+x+'</th>' for x in METRICS)+'</tr>']
    for run,label,report in zip(runs,labels,reports):
        for category in ['all','lamp','pillow']:
            s=summary[run.name+'/'+category]
            parts.append('<tr><td>'+esc(label)+' / '+category+'</td><td>'+str(s['n'])+'</td>'+''.join('<td>'+f'{s["scores"][x]["mean"]:.4f}'+'</td>' for x in METRICS)+'</tr>')
    parts.append('</table><h2>模型与预算</h2><table><tr><th>模型</th><th>采样步数</th><th>实际加载参数</th><th>训练记录</th><th>权重身份</th></tr>')
    for run,label,report in zip(runs,labels,reports):
        total=report.get('generator_loaded_parameters',0)+report.get('Qwen_parameters',0)+report.get('head_parameters',0)
        if not total:total=report.get('loaded_parameters')
        history=report.get('checkpoint_training_history')or{}
        budget=('未任务微调（冻结公开模型）' if report.get('peer_has_no_task_tuning') else
                '完成 '+str(history.get('completed_epochs'))+' 轮 / '+str(history.get('completed_updates'))+' 更新；所选第 '+str(history.get('selected_epoch'))+' 轮')
        parts.append('<tr><td>'+esc(label)+'</td><td>'+esc(report.get('sampling_steps'))+'</td><td>'+esc(total)+'</td><td>'+esc(budget)+'</td><td>'+esc(report['checkpoint_sha256'])+'</td></tr>')
    parts.append('</table><p>人工验图和正式放行仍待完成；页面生成不代表已经通知光路部署。</p>')
    for sid in sorted(common):
        first=rows[runs[0].name][sid]
        parts.append('<article><h2>'+esc(sid)+'</h2><p>'+esc(first['prompt'])+'</p><div class="grid">')
        ref=runs[0]/(first['reference_id']+'.png');dest=output/'images'/('input-'+first['reference_id']+'.png')
        if not dest.exists():shutil.copyfile(ref,dest)
        parts.append('<div>白底输入<img loading="lazy" src="images/'+esc(dest.name)+'"></div>')
        for index,(run,label) in enumerate(zip(runs,labels)):
            row=rows[run.name][sid];source=run/row['generated_PNG']
            if sha(source)!=row['generated_sha256']:raise ValueError('Changed delivered pixels')
            name=str(index)+'-'+row['generated_PNG'];shutil.copyfile(source,output/'images'/name)
            parts.append('<div>'+esc(label)+'<img loading="lazy" src="images/'+esc(name)+'"><div class="metric">'+
                         ' · '.join(x+' '+f'{row[x]:.4f}' for x in METRICS)+'</div></div>')
        parts.append('</div></article>')
    parts.append('</html>');(output/'index.html').write_text(''.join(parts),encoding='utf8')
    write(output/'report.json',dict(complete=True,model_runs=list(map(str,runs)),labels=labels,request_sha256=next(iter(manifest)),
        complete_TEST_requests=500,metrics_source_sha256=sha(metrics/'per_image.json'),HTML_sha256=sha(output/'index.html'),
        quality_and_speed_acceptance=gate,manual_review_not_replaced_by_HTML=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--electronic-rounds',type=Path,nargs=3,required=True);p.add_argument('--LG-rounds',type=Path,nargs=3,required=True)
    for name in ['metrics','workflow','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build(a.runs,a.labels,a.metrics,a.electronic_rounds,a.LG_rounds,a.workflow,a.output)


if __name__=='__main__':main()
