"""Render only the independently replayed cold, distinct-video MPS measurements."""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import shutil
import statistics
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO=Path(__file__).resolve().parents[4]
ROOT=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_dataset_once_a100_20261020/presentation'
OUT=REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
NAMES=dict(qwen='Qwen3-VL-2B',glm='GLM-4.6V-Flash',clip='CLIP ViT-B/32',yolo='YOLO11s',deepseek='DeepSeek-VL2',kimi='Kimi-VL-A3B')
COLORS=dict(qwen='#0072B2',glm='#92527F',clip='#007F63',yolo='#9B7000',deepseek='#79502C',kimi='#D14F64')
MARKERS=dict(qwen='o',glm='P',clip='D',yolo='^',deepseek='s',kimi='v')
OPTICAL=dict(time16_ms=10.245569591370225,energy16_j=1.1776895097085185,srcc=.7977138739)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def plain_model_names(names):
    """Remove display footnote glyphs without changing measurements or caveats."""
    return {key:value.replace('†','').strip() for key,value in names.items()}


def selected_batch_view(root,fixed_batch=None):
    """Replace only changed-batch costs; preserve approved B1 and raw history."""
    prior_root=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_batch_activity_a100_20261021/resident01'
    old_source=(prior_root/'selected_batch_paper.json' if fixed_batch else
        REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_dataset_once_rental_20261021_replacement02/presentation/dataset_once_paper.json')
    data=deepcopy(json.loads(old_source.read_text()))
    audit_path=root/'final_saved_audit.json'
    audit=json.loads(audit_path.read_text())
    historical_path=(prior_root if fixed_batch else root)/'historical_80gb_published_audit.json'
    historical=json.loads(historical_path.read_text())
    assert historical['status']=='passed'
    assert audit['status']=='passed' and audit['audited_points']==(24 if fixed_batch else 54)
    if fixed_batch:
        assert fixed_batch==6 and audit['suite_status']=='complete' and audit['mps_released']
        assert all(audit['selections'][m]['batch']==fixed_batch for m in ('clip','yolo'))
    data['prior_paper_source']=dict(path=old_source.relative_to(REPO).as_posix(),sha256=digest(old_source))
    data['batch_selection_audit']=dict(path=audit_path.relative_to(REPO).as_posix(),sha256=digest(audit_path))
    data['archived_previous_repeats']=deepcopy(data['repeats'])
    # Preserve the last approved exports before any replacement, not on later reruns.
    archive=root/'paper_exports_before_selected_batch.json'
    if not archive.exists():
        files={}
        for target in [OUT/'LGVQ_MPS_dataset_once_comparison.xlsx',OUT/'mps_dataset_once_plot_data.csv',*sorted(OUT.glob('mps_dataset_once_comparison*'))]:
            if not target.is_file():continue
            sha=digest(target)
            saved=root/f'prior_{sha[:12]}_{target.name}'
            if not saved.exists():shutil.copy2(target,saved)
            assert digest(saved)==sha
            files[target.name]=dict(sha256=sha,archive=saved.relative_to(REPO).as_posix())
        archive.write_text(json.dumps(files,indent=2)+'\n')
    for point in data['points']:
        model,cap=point['model'],point['resource_cap']
        if model not in ('clip','yolo'):
            point['measurement_reuse']='Approved unchanged B1 cost archive; no new inference'
            continue
        selected=audit['selections'][model]
        rows=[]
        for raw in [p for p in audit['points'] if p['mode']=='cost' and p['model']==model and p['cap']==cap]:
            source=root/raw['report'];folder=source.parent
            report=json.loads(source.read_text());cost=report['cost'];integration=cost['power_integration']
            assert report['batch']==selected['batch'] and report['test_videos']==558 and report['cost_videos']==200
            assert digest(source)==raw['report_sha256']
            samples=list(csv.DictReader((folder/'telemetry.csv').open(newline='')))
            rows.append(dict(model=model,resource_cap=cap,repeat=report['repeat'],batch=report['batch'],
                srcc=report['performance']['srcc'],plcc=report['performance']['plcc'],
                time16_ms=cost['time16_ms'],energy16_j=cost['energy16_j'],
                compute_only_time16_ms=cost['compute_s']*80,nonpreprocessing_gap16_ms=cost['nonpreprocessing_gap_s']*80,
                charged_s=integration['elapsed_s'],gpu_energy_j=integration['energy_j']-338.2*integration['elapsed_s'],
                video_evaluations=200,visible_sms=report['visible_multiprocessors'],
                actual_busy_percent=statistics.mean(float(r['utilization_percent']) for r in samples),
                checkpoint_sha256=report['checkpoint_sha256'],parameter_dtypes=report['floating_parameter_dtypes'],
                readout_dtypes=report['quality_readout_dtypes'],source=source.relative_to(REPO).as_posix(),
                completed_at_beijing=datetime.fromtimestamp(json.loads((folder/'status.json').read_text())['unix'],timezone(timedelta(hours=8))).isoformat(),
                source_commit=report['git_commit'],source_sha256=digest(source),identity_sha256=digest(folder/'identity.json')))
        assert len(rows)==3
        assert all(r['checkpoint_sha256']==point['repeats'][0]['checkpoint_sha256'] for r in rows)
        point.update(batch=selected['batch'],time16_ms=statistics.mean(r['time16_ms'] for r in rows),
            energy16_j=statistics.mean(r['energy16_j'] for r in rows),repeats=rows,
            srcc_repeats=[r['srcc'] for r in rows],measured_srcc=statistics.mean(r['srcc'] for r in rows),
            measurement_reuse='New changed-batch cold distinct-video costs, 21 October 2026')
    for point in data['points']:
        ref=next(p for p in data['points'] if p['model']==point['model'] and p['resource_cap']==100)
        point.update(srcc=statistics.mean(ref['srcc_repeats']),accuracy_reference_repeats=ref['srcc_repeats'],
            accuracy_reference_reports=[dict(source=r['source'],sha256=r['source_sha256']) for r in ref['repeats']])
        if point['model'] in audit['selections']:
            condition=audit['selections'][point['model']]['caps'][str(point['resource_cap'])]
            point['resident_utilization_reference_percent']=condition['resident_nvml_mean_percent']
        elif point['model'] in ('deepseek','kimi'):
            condition=next(v for v in historical['values'] if v['model']==point['model'] and v['cap']==point['resource_cap'])
            assert condition['batch']==point['batch'] and condition['checkpoint_sha256']==point['repeats'][0]['checkpoint_sha256']
            point['resident_utilization_reference_percent']=condition['resident_mean_busy_percent']
            point['resident_utilization_note']='Historical warmed80GB reference, different physical instance from cost; not same-window telemetry'
        point['cost_instrumentation_note']='All charged internal gaps retained; new CLIP/YOLO workers log each batch, older workers about every25 batches.'
    data['repeats']=[r for p in data['points'] for r in p['repeats']]
    data['selected_batch_view']=True
    data['selection_batches']={p['model']:p['batch'] for p in data['points']}
    assert data['selection_batches']==dict(qwen=1,glm=1,clip=fixed_batch or 4,yolo=fixed_batch or 2,deepseek=1,kimi=1)
    assert next(p for p in data['points'] if p['model']=='qwen' and p['resource_cap']==100)['time16_ms']==1188.3861464262009
    if fixed_batch:
        # Every unchanged model remains byte-for-byte identical in numerical data.
        before=json.loads(old_source.read_text())
        for point in data['points']:
            if point['model'] in ('clip','yolo'):continue
            prior=next(p for p in before['points'] if p['model']==point['model'] and p['resource_cap']==point['resource_cap'])
            for key in ('time16_ms','energy16_j','srcc','repeats'):
                assert point[key]==prior[key],(point['model'],key)
    source=root/('batch6_paper.json' if fixed_batch else 'selected_batch_paper.json')
    source.write_text(json.dumps(data,indent=2)+'\n')
    return source,data


def panel(ax,points,metric):
    xs=[OPTICAL[metric]]+[p[metric] for p in points]
    ys=[OPTICAL['srcc']]+[p['srcc'] for p in points]
    ax.scatter(OPTICAL[metric],OPTICAL['srcc'],s=52,marker='*',color='#7828A8',zorder=4)
    for model in NAMES:
        rr=sorted([p for p in points if p['model']==model],key=lambda r:-r['resource_cap'])
        assert len(rr)==2
        ax.plot([r[metric] for r in rr],[r['srcc'] for r in rr],color=COLORS[model],lw=.9,zorder=2,
                linestyle='--' if any('cost_evidence' in r for r in rr) else '-')
        # Draw the larger open mark first so coincident conditions remain visible.
        # Coordinates and model-specific marker geometry are never displaced.
        for r in reversed(rr):
            ax.scatter(r[metric],r['srcc'],s=18 if r['resource_cap']==100 else 52,marker=MARKERS[model],
                       facecolor=COLORS[model] if r['resource_cap']==100 else 'white',
                       edgecolor=COLORS[model],linewidth=.9,zorder=3)
    ax.set_xscale('log');ax.set_xlim(min(xs)*.7,max(xs)*1.55)
    ymin=min(.65,min(ys)-.025);ymax=max(.85,max(ys)+.025)
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('Reference temporal quality (SRCC)' if any('accuracy_reference_cap' in p for p in points) else 'Temporal quality (SRCC)')
    ax.set_xlabel('Equivalent processing time (ms)' if metric=='time16_ms' else 'Estimated equivalent energy (J)')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(which='major',color='#D8DDE2',lw=.45);ax.grid(which='minor',axis='x',color='#ECEFF1',lw=.3)
    ax.tick_params(direction='out',length=2.5)
    assert all(ax.get_xlim()[0]<x<ax.get_xlim()[1] for x in xs)
    assert all(ymin<y<ymax for y in ys)
    return dict(plotted_points=len(xs),models=len(NAMES),connected_pairs=len(NAMES),x_limits=list(ax.get_xlim()),y_limits=list(ax.get_ylim()))


def legends(fig):
    models=[Line2D([],[],marker='*',ls='none',color='#7828A8',label='LightGen-II',markersize=7)]
    models.extend(Line2D([],[],marker=MARKERS[m],ls='-',color=COLORS[m],label=NAMES[m],markersize=4,lw=.8)for m in NAMES)
    fig.legend(handles=models,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2 if len(NAMES)==4 else 3,frameon=False,columnspacing=.9,handlelength=1.2)
    caps=[Line2D([],[],marker='o',ls='none',color='#58646B',markerfacecolor=fc,label=label,markersize=size)
          for fc,label,size in [('#58646B','100% MPS allocation',4.2),('white','50% MPS allocation',5.6)]]
    fig.legend(handles=caps,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=1,frameon=False,handlelength=1.0,labelspacing=.2)


def fixed_batch_comparison(root):
    """Show all three cold repeats at B6 and B10 without reselecting a result."""
    rows=[]
    for batch in (6,10):
        source=root.parent/f'batch{batch}_01/independent_cost_check.json'
        data=json.loads(source.read_text())
        assert data['status']=='passed' and data['independent_time_and_overlap_energy_replay']
        for model in ('clip','yolo'):
            for cap in (100,50):
                group=data['models'][model]['caps'][str(cap)]
                assert group['batch']==batch and len(group['time_repeats_ms'])==len(group['energy_repeats_j'])==3
                rows.append(dict(model=model,batch=batch,resource_cap=cap,
                    time16_ms=group['time16_ms'],energy16_j=group['energy16_j'],
                    time_repeats_ms=group['time_repeats_ms'],energy_repeats_j=group['energy_repeats_j'],
                    source=source.relative_to(REPO).as_posix(),source_sha256=digest(source)))
    fig,axes=plt.subplots(1,2,figsize=(156/25.4,83/25.4))
    fig.subplots_adjust(left=.09,right=.985,bottom=.25,top=.85,wspace=.32)
    for ax,metric,raw,label in zip(axes,['time16_ms','energy16_j'],['time_repeats_ms','energy_repeats_j'],['a','b']):
        for index,(model,batch) in enumerate([('clip',6),('clip',10),('yolo',6),('yolo',10)]):
            for cap,offset in ((100,-.13),(50,.13)):
                row=next(r for r in rows if (r['model'],r['batch'],r['resource_cap'])==(model,batch,cap))
                x=index+offset;values=row[raw];color=COLORS[model]
                ax.scatter([x]*3,values,s=8,color=color,alpha=.35,zorder=2)
                ax.errorbar(x,row[metric],yerr=statistics.stdev(values),fmt='o',ms=4,
                    mfc=color if cap==100 else 'white',mec=color,color=color,capsize=2,lw=.8,zorder=3)
        ax.set_xticks(range(4),['CLIP\nBatch 6','CLIP\nBatch 10','YOLO\nBatch 6','YOLO\nBatch 10'])
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Equivalent processing time (ms)' if metric=='time16_ms' else 'Estimated equivalent energy (J)')
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(axis='y',color='#D8DDE2',lw=.4);ax.set_axisbelow(True)
        ax.text(-.17,1.05,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
    handles=[Line2D([],[],marker='o',ls='none',color='#58646B',mfc=fc,label=label,ms=4)
        for fc,label in [('#58646B','100% MPS allocation'),('white','50% MPS allocation')]]
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.04,'Three repeats; mean and sample standard deviation. Cold, distinct-video costs.',ha='center',fontsize=6.5)
    files={}
    for ext in ('png','svg','pdf'):
        target=OUT/f'mps_clip_yolo_batch6_batch10.{ext}'
        fig.savefig(target,dpi=600);files[target.name]=digest(target)
    plt.close(fig)
    target=OUT/'mps_clip_yolo_batch6_batch10.csv'
    with target.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    files[target.name]=digest(target)
    (root/'batch6_batch10_figure_receipt.json').write_text(json.dumps(dict(files=files,rows=rows),indent=2)+'\n')


def ready_input_batch_comparison(root):
    """All observed B6/B8/B10 repeats, consistently reaccounted, no best-run choice."""
    source=root/'ready_input_comparison.json'
    rows=json.loads(source.read_text())
    plan=[('clip',6),('clip',8),('clip',10),('yolo',6),('yolo',10)]
    assert len(rows)==10
    fig,axes=plt.subplots(1,2,figsize=(156/25.4,88/25.4))
    fig.subplots_adjust(left=.10,right=.985,bottom=.26,top=.84,wspace=.38)
    for ax,metric,raw,label in zip(axes,['time16_ms','energy16_j'],['time_repeats_ms','energy_repeats_j'],['a','b']):
        for index,(model,batch) in enumerate(plan):
            for cap,offset in ((100,-.15),(50,.15)):
                row=next(r for r in rows if (r['model'],r['batch'],r['resource_cap'])==(model,batch,cap))
                values=row[raw];x=index+offset;color=COLORS[model]
                assert len(values)==3
                ax.scatter([x]*3,values,s=8,color=color,alpha=.4,zorder=2)
                ax.errorbar(x,row[metric],yerr=statistics.stdev(values),fmt='o',ms=4,
                    mfc=color if cap==100 else 'white',mec=color,color=color,capsize=2,lw=.8,zorder=3)
        ax.set_xticks(range(5),[f'{m.upper()}\nBatch {b}' for m,b in plan],fontsize=6)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Equivalent processing time (ms)' if metric=='time16_ms' else 'Estimated equivalent energy (J)')
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(axis='y',color='#D8DDE2',lw=.4);ax.set_axisbelow(True)
        ax.text(-.19,1.05,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
    handles=[Line2D([],[],marker='o',ls='none',color='#58646B',mfc=fc,label=label,ms=4)
        for fc,label in [('#58646B','100% MPS allocation'),('white','50% MPS allocation')]]
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.07,'Three cold repeats; mean and sample standard deviation.',ha='center',fontsize=6.5)
    fig.text(.5,.035,'Prepared-input model calls and recorded internal gaps; no logging subtraction.',ha='center',fontsize=6.5)
    files={}
    for ext in ('png','svg','pdf'):
        target=OUT/f'mps_clip_yolo_batch6_batch10.{ext}'
        fig.savefig(target,dpi=600);files[target.name]=digest(target)
    plt.close(fig)
    target=OUT/'mps_clip_yolo_batch6_batch10.csv'
    with target.open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['model','batch','resource_cap','time16_ms','energy16_j','time_repeats_ms','energy_repeats_j','accounting','source','source_sha256']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({**{k:json.dumps(row[k]) if isinstance(row[k],list) else row[k] for k in fields[:7]},
                'accounting':'prepared_input_model_call_through_cpu_return_plus_internal_gaps_v2',
                'source':source.relative_to(REPO).as_posix(),'source_sha256':digest(source)})
    files[target.name]=digest(target)
    (root/'ready_input_comparison_figure_receipt.json').write_text(json.dumps(dict(files=files,source_sha256=digest(source)),indent=2)+'\n')


def main():
    global ROOT,NAMES
    p=argparse.ArgumentParser();p.add_argument('--combined',action='store_true');p.add_argument('--replacement02',action='store_true');p.add_argument('--finalized-qwen',action='store_true');p.add_argument('--selected-batches',action='store_true');p.add_argument('--fixed-batch6',action='store_true');p.add_argument('--ready-input',type=Path);p.add_argument('--plain-model-labels',action='store_true');args=p.parse_args()
    if args.finalized_qwen and not args.combined:raise ValueError('Finalized Qwen presentation requires combined data')
    if args.combined:ROOT=ROOT.parent.with_name('lgvq_mps_dataset_once_rental_20261021'+('_replacement02' if args.replacement02 else ''))/'presentation'
    else:NAMES={k:v for k,v in NAMES.items() if k not in ('deepseek','kimi')}
    if args.ready_input:
        assert args.combined and args.finalized_qwen and not (args.fixed_batch6 or args.selected_batches)
        source=args.ready_input.resolve();data=json.loads(source.read_text());ROOT=source.parent
        assert data['accounting']=='prepared_input_model_call_through_cpu_return_plus_internal_gaps_v2'
        assert data['all_logs_retained_no_estimated_subtraction']
        anchor=next(p for p in data['points'] if p['model']=='qwen' and p['resource_cap']==100)
        assert (anchor['time16_ms'],anchor['energy16_j'])==(1188.3861464262009,462.1134010767563)
    elif args.fixed_batch6:
        assert args.combined and args.finalized_qwen and not args.selected_batches
        ROOT=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_batch_activity_a100_20261021/batch6_01'
        source,data=selected_batch_view(ROOT,fixed_batch=6)
    elif args.selected_batches:
        assert args.combined and args.finalized_qwen
        ROOT=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_batch_activity_a100_20261021/resident01'
        source,data=selected_batch_view(ROOT)
    else:
        source=ROOT/('dataset_once_combined.json' if args.combined else 'dataset_once_audited.json');data=json.loads(source.read_text())
    assert data['status']=='complete' and len(data['points'])==len(NAMES)*2 and len(data['repeats'])==len(NAMES)*6
    if args.combined and not (args.selected_batches or args.fixed_batch6 or args.ready_input):
        previous_receipt=ROOT/'figure_receipt.json'
        if not previous_receipt.exists():previous_receipt=ROOT.parents[1]/'lgvq_mps_dataset_once_a100_20261020/presentation/figure_receipt.json'
        previous=json.loads(previous_receipt.read_text())
        if previous.get('finalized_qwen_cost') and not args.finalized_qwen:raise RuntimeError('Do not silently undo the finalized Qwen presentation')
        for name,expected in previous['files'].items():
            target=OUT/name
            if not target.exists():continue
            actual=digest(target)
            if actual!=expected:raise RuntimeError('Existing export changed externally; preserve '+name)
            archive=ROOT/(target.stem+'_prior_'+actual[:12]+target.suffix)
            if not archive.exists():shutil.copy2(target,archive)
    elif not (args.selected_batches or args.fixed_batch6 or args.ready_input) and (ROOT/'figure_receipt.json').exists():
        previous=json.loads((ROOT/'figure_receipt.json').read_text())
        for name,expected in previous['files'].items():
            target=OUT/name
            if target.exists() and digest(target)!=expected:
                raise RuntimeError('Refuse older four-model export over a newer or externally changed presentation: '+name)
    if args.finalized_qwen and not (args.selected_batches or args.fixed_batch6 or args.ready_input):
        from .mps_paper_view import finalized_qwen_view,full_resource_accuracy_view
        original_source=source
        data=full_resource_accuracy_view(finalized_qwen_view(data,REPO))
        data['measurement_source']={'path':original_source.relative_to(REPO).as_posix(),'sha256':digest(original_source)}
        source=ROOT/'dataset_once_paper.json'
        source.write_text(json.dumps(data,indent=2)+'\n')
        NAMES={**NAMES,'qwen':'Qwen3-VL-2B†'}
    if args.selected_batches or args.fixed_batch6 or args.ready_input:NAMES={**NAMES,'qwen':'Qwen3-VL-2B†'}
    if args.plain_model_labels:NAMES=plain_model_names(NAMES)
    optical_source=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_strict_tf_rental_dual_20261020/presentation/eight_tables.json'
    prior=json.loads(optical_source.read_text())['optical']
    assert (prior['time_ms'],prior['energy_j'],prior['srcc'])==(OPTICAL['time16_ms'],OPTICAL['energy16_j'],OPTICAL['srcc'])
    points=data['points'];plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7.5,'svg.fonttype':'none','pdf.fonttype':42})
    receipt=dict(source_sha256=digest(source),marker_rendering='Larger open50 mark behind smaller filled100 mark, exact coordinates and same model geometry',optical_reference_source=optical_source.relative_to(REPO).as_posix(),
                 renderer_source_sha256=digest(Path(__file__)),
                 optical_reference_sha256=digest(optical_source),protocol='dataset_once_legacy_block_plus_nonpreprocessing_gaps_v1',finalized_qwen_cost=args.finalized_qwen,selected_batch_view=args.selected_batches or args.fixed_batch6,fixed_clip_yolo_batch=6 if args.fixed_batch6 else None,files={},panels=[])
    if args.ready_input:
        receipt.update(protocol=data['accounting'],accounting_exception=data['accounting_exception'],
            selected_batches={p['model']:p['batch'] for p in data['points']})
    receipt['plain_model_labels']=args.plain_model_labels
    for metric,kind in [('time16_ms','latency'),('energy16_j','energy')]:
        fig,ax=plt.subplots(figsize=(78/25.4,88/25.4));fig.subplots_adjust(left=.20,right=.97,bottom=.32,top=.84)
        receipt['panels'].append(panel(ax,points,metric))
        if args.combined:
            # Six model names fit legibly in two columns at the existing width.
            handles=[Line2D([],[],marker='*',ls='none',color='#7828A8',label='LightGen-II',markersize=7)]
            handles.extend(Line2D([],[],marker=MARKERS[m],ls='-',color=COLORS[m],label=NAMES[m],markersize=4,lw=.8)for m in NAMES)
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,columnspacing=.9,handlelength=1.2)
            caps=[Line2D([],[],marker='o',ls='none',color='#58646B',markerfacecolor=fc,label=label,markersize=4.2)
                  for fc,label in [('#58646B','100% MPS allocation'),('white','50% MPS allocation')]]
            fig.legend(handles=caps,loc='upper center',bbox_to_anchor=(.5,1),frameon=False,labelspacing=.2)
        else:legends(fig)
        for ext in ('png','svg','pdf'):
            target=OUT/f'mps_dataset_once_comparison_{kind}.{ext}';fig.savefig(target,dpi=600);receipt['files'][target.name]=digest(target)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(156/25.4,88/25.4));fig.subplots_adjust(left=.09,right=.985,bottom=.28,top=.84,wspace=.30)
    for ax,metric,label in zip(axes,['time16_ms','energy16_j'],['a','b']):
        panel(ax,points,metric);ax.text(-.18,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
    legends(fig);target=OUT/'mps_dataset_once_comparison_pair.png';fig.savefig(target,dpi=600);receipt['files'][target.name]=digest(target);plt.close(fig)
    target=OUT/'mps_dataset_once_plot_data.csv'
    base_fields=['model','resource_cap','batch','time16_ms','energy16_j','srcc']
    fields=['model','model_name',*base_fields[1:],'measured_srcc_repeats','reference_srcc_repeats','measured_srcc','accuracy_basis','time_ratio_to_optical','energy_ratio_to_optical','unique_cost_videos_per_repeat','full_test_videos_per_repeat','repeats','evidence_type','source_json','source_sha256','gpu_platform','power_limit_w','parameter_dtypes','readout_dtypes','cost_source','cost_source_sha256','cost_source_selector','measured_time16_ms','measured_energy16_j','cost_boundary','measurement_reuse','resident_utilization_reference_percent','cost_instrumentation_note']
    with target.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerow(dict(model='ours',model_name='LightGen-II',**OPTICAL,time_ratio_to_optical=1,energy_ratio_to_optical=1,
                        evidence_type='Prior optical component-cost reference',source_json=optical_source.relative_to(REPO).as_posix(),source_sha256=digest(optical_source)))
        for p in points:
            w.writerow({**{k:p[k] for k in base_fields},'model_name':NAMES[p['model']],
                        'measured_srcc_repeats':json.dumps(p['srcc_repeats']),
                        'reference_srcc_repeats':json.dumps(p.get('accuracy_reference_repeats',p['srcc_repeats'])),
                        'measured_srcc':p.get('measured_srcc',p['srcc']),
                        'accuracy_basis':'Shared MPS100 full558 reference, not MPS50 measured accuracy' if args.finalized_qwen else 'Measured per allocation',
                        'time_ratio_to_optical':p['time16_ms']/OPTICAL['time16_ms'],
                        'energy_ratio_to_optical':p['energy16_j']/OPTICAL['energy16_j'],
                        'unique_cost_videos_per_repeat':200,'full_test_videos_per_repeat':558,'repeats':3,'evidence_type':p.get('cost_evidence',p.get('measurement_reuse','New cold distinct-video cost; shared MPS100 accuracy reference' if args.finalized_qwen else 'New cold distinct-video measurement')),
                        'source_json':source.relative_to(REPO).as_posix(),'source_sha256':digest(source),
                        'gpu_platform':'A100-SXM4-80GB' if p['model'] in ('deepseek','kimi') else 'A100-PCIE-40GB',
                        'power_limit_w':400 if p['model'] in ('deepseek','kimi') else 250,
                        'parameter_dtypes':', '.join(p['repeats'][0]['parameter_dtypes']),
                        'readout_dtypes':', '.join(p['repeats'][0]['readout_dtypes']),
                        **{k:p.get(k,'') for k in ['cost_source','cost_source_sha256','cost_source_selector','measured_time16_ms','measured_energy16_j','cost_boundary','measurement_reuse','resident_utilization_reference_percent','cost_instrumentation_note']}})
    receipt['files'][target.name]=digest(target)
    (ROOT/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if args.fixed_batch6:fixed_batch_comparison(ROOT)
    if args.ready_input:ready_input_batch_comparison(ROOT)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
