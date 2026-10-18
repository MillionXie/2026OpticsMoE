"""Replot verified sparse figure points, never reconstruct model measurements."""
from __future__ import annotations
import json
import hashlib
import shutil
import sys
import csv
import math
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as PlotPath
import numpy as np

REPO=Path(__file__).resolve().parents[4]
RUN=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_selected_execution_a100_20261016'
if '--yolo-clip-recheck' in sys.argv:RUN=RUN.with_name('lgvq_yolo_clip_recheck_a100_20261017')
if '--yolo-clip-sequential' in sys.argv:RUN=RUN.with_name('lgvq_yolo_clip_sequential_a100_20261017')
CLEAN_KIMI='--kimi-clean-extension' in sys.argv
RENTAL='--rental-extension' in sys.argv or CLEAN_KIMI
INCLUDE_CROSS_DEVICE_AUDIT='--include-cross-device-audit' in sys.argv
if RENTAL:RUN=RUN.with_name('lgvq_full_deepseek_kimi_rental_20261017')
if CLEAN_KIMI:RUN=RUN/'figure5_kimi_refresh_20261018'
OUT=REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
COLORS=dict(ours='#7828A8',qwen='#0072B2',glm='#92527F',small='#C25B18',clip='#007F63',yolo='#9B7000',deepseek='#5B6470')
MARKERS=dict(ours='*',qwen='o',glm='P',small='s',clip='D',yolo='^',deepseek='v')
if RENTAL:
    COLORS.update(full='#79502C',kimi='#B64058')
    MARKERS.update(full='s',kimi='X')
    if INCLUDE_CROSS_DEVICE_AUDIT:
        COLORS['qwen80']='#256F99';MARKERS['qwen80']='h'


def paper_rows(rows):
    """17 October user decision: rental Qwen is audit-only, not a paper point."""
    return list(rows) if INCLUDE_CROSS_DEVICE_AUDIT else [r for r in rows if r['baseline']!='qwen80']


def place_labels(fig,ax):
    """Place every visible label without altering data or hiding points."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=ax.get_window_extent(renderer);placed=[]
    points=[ax.transData.transform(point) for point in ax._audited_points]
    grid=[(x,y) for x in np.linspace(.07,.94,14) for y in np.linspace(.07,.94,14)]
    for ann in [t for t in ax.texts if t.get_bbox_patch() is not None]:
        preferred=ann.get_position();box=ann.get_bbox_patch().get_window_extent(renderer)
        original=ax.transAxes.transform(preferred)
        choices=[preferred]+sorted(grid,key=lambda p:np.linalg.norm(ax.transAxes.transform(p)-ax.transData.transform(ann.xy)))
        for position in choices:
            delta=ax.transAxes.transform(position)-original;candidate=box.translated(*delta)
            if not(bounds.contains(candidate.x0,candidate.y0) and bounds.contains(candidate.x1,candidate.y1)):continue
            if any(candidate.padded(2).overlaps(other) for other in placed):continue
            if any(candidate.padded(5).contains(*point) for point in points):continue
            ann.set_position(position);placed.append(candidate);break
        else:raise AssertionError('No nonoverlapping label slot')


def panel(ax,rows,kind):
    key='energy_16_videos_j' if kind=='energy' else 'time_16_videos_ms'
    # Fixed text slots for this audited sparse cohort, with leaders to the true data.
    common={('ours',None):(.09,.735),('small',1):(.93,.61),('small',3):(.90,.44),('small',4):(.78,.39),
            ('deepseek',1):(.86,.23),('deepseek',8):(.71,.27)}
    slots={
        'latency':{('qwen',1):(.77,.70),('qwen',3):(.61,.75),('qwen',4):(.53,.64),
                   ('glm',1):(.86,.85),('glm',2):(.70,.87),('clip',1):(.46,.61),('clip',10):(.26,.65),
                   ('yolo',1):(.59,.46),('yolo',3):(.40,.43),('yolo',4):(.50,.35),('clip',11):(.16,.43)},
        'energy':{('qwen',1):(.70,.71),('qwen',3):(.54,.78),('qwen',4):(.64,.59),
                  ('glm',1):(.80,.86),('glm',2):(.68,.85),('clip',1):(.48,.62),('clip',10):(.31,.65),
                  ('yolo',1):(.60,.46),('yolo',3):(.43,.43),('yolo',4):(.53,.35),('clip',11):(.20,.50)}}
    for model,color in COLORS.items():
        selected=[x for x in rows if x['baseline']==model]
        if not selected:continue
        connected=[x for x in selected if not(model=='qwen' and x['batch_size_videos']==1)]
        if len(connected)>1:ax.plot([x[key] for x in connected],[x['measured_full558_srcc'] for x in connected],color=color,lw=1,ls='--' if model=='full' or (model=='kimi' and not CLEAN_KIMI) else '-',alpha=.55,zorder=2)
        for x in selected:
            historic=model=='qwen' and x['batch_size_videos']==1
            ax.scatter(x[key],x['measured_full558_srcc'],s=125 if model=='ours' else 39,marker=MARKERS[model],facecolors='white' if historic or model in ('full','kimi','qwen80') else color,edgecolors=color,linewidths=1.1,zorder=4)
            if model=='ours':text='N=16'
            else:
                b=x['batch_size_videos'];text=f'B={b}'+('†' if historic else '')+f"\n{x['display_gpu_util_mean_percent']:.2f}%"
            extra_slot=((np.log10(x[key])-np.log10(.8 if kind=='energy' else 8))/(4-np.log10(.8 if kind=='energy' else 8))+.03,(x['measured_full558_srcc']-.5)/.4+.055) if RENTAL else None
            slot=(model,x['batch_size_videos']);xytext=slots[kind].get(slot,common.get(slot,extra_slot));assert xytext is not None,slot
            ax.annotate(text,(x[key],x['measured_full558_srcc']),xytext=xytext,textcoords='axes fraction',ha='center',va='center',fontsize=8.5,color=color,bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.88),arrowprops=dict(arrowstyle='-',color=color,lw=.6,alpha=.65),zorder=5)
    xs=[x[key] for x in rows];ys=[x['measured_full558_srcc'] for x in rows]
    ax._audited_points=list(zip(xs,ys))
    xmin=min(.8 if kind=='energy' else 8,min(xs)*.75);xmax=max(10000,10**np.ceil(np.log10(max(xs)*1.08)))
    ymin=min(.5,np.floor(min(ys)*20)/20-.05);ymax=max(.85,np.ceil(max(ys)*20)/20+.05)
    ax.set_xscale('log');ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax);ax.set_yticks(np.arange(ymin,ymax+.001,.05));ax.set_ylabel('Temporal quality (SRCC)')
    assert all(xmin<x<xmax and ymin<y<ymax for x,y in zip(xs,ys))
    ax.set_xlabel('Estimated energy per 16 videos (J)' if kind=='energy' else 'Latency per 16 videos (ms)')
    ax.grid(which='major',color='#D8DDE2',linewidth=.6);ax.grid(which='minor',axis='x',color='#EBEEF0',linewidth=.4)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.tick_params(labelsize=9,direction='out')


def verify_label_geometry(fig,axes,check_lines=False):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();checks=[]
    for ax in axes:
        boxes=[text.get_bbox_patch().get_window_extent(renderer) for text in ax.texts if text.get_bbox_patch() is not None]
        assert len(boxes)==len(ax._audited_points)
        bounds=ax.get_window_extent(renderer)
        for i,box in enumerate(boxes):
            assert bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1),'Label outside axes'
            assert all(not box.padded(1).overlaps(other) for other in boxes[i+1:]),'Overlapping data labels'
            assert all(not box.padded(2).contains(*ax.transData.transform(point)) for point in ax._audited_points),f'Label covers data marker: annotation {i}'
            if check_lines:
                for line in ax.lines:
                    vertices=line.get_transform().transform(line.get_xydata())
                    assert not PlotPath(vertices).intersects_bbox(box.padded(1),filled=False),'Label covers model connector'
                for j,text in enumerate(ax.texts):
                    if j!=i and getattr(text,'arrow_patch',None) is not None:
                        assert not text.arrow_patch.get_path().intersects_bbox(box.padded(.3),filled=False),'Label covers another annotation leader'
        checks.append(dict(labels=len(boxes),overlaps=0,covered_markers=0,covered_connectors=0 if check_lines else None,all_labels_within_axes=True,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())))
    return checks


def main():
    if '--compact-groups-data' in sys.argv:
        compact_group_data()
        return
    if '--compact-groups-render' in sys.argv:
        compact_group_figures()
        return
    state=json.loads((RUN/'presentation_selected.json').read_text());assert state['status']=='complete'
    rows=paper_rows(state['rows']);assert 13<=len(rows)<=(20 if RENTAL else 18)
    if RENTAL and not INCLUDE_CROSS_DEVICE_AUDIT and not CLEAN_KIMI:
        archive=RUN/'paper_qwen80_exclusion_archive'
        if not archive.exists():
            archive.mkdir();receipts={}
            for name in ['latency_comparison_reference','energy_comparison_reference','srcc_cost_comparison_reference','batch_cost_comparison','gpu_utilization_comparison']:
                for ext in ['png','svg']:
                    source=OUT/(name+'.'+ext)
                    if source.exists():
                        shutil.copy2(source,archive/source.name)
                        receipts[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
            source=RUN/'figure_label_audit.json'
            if source.exists():shutil.copy2(source,archive/source.name)
            (archive/'manifest.json').write_text(json.dumps(dict(reason='User excluded rental Qwen B1 from paper; original evidence retained',files=receipts),indent=2)+'\n')
    sequential='--yolo-clip-sequential' in sys.argv or RENTAL
    assert [(x['batch_size_videos'],x['display_gpu_util_mean_percent']) for x in rows if x['baseline']=='qwen' and x['batch_size_videos']==1]==[(1,63.133111480865225)]
    plt.rcParams.update({'font.family':'Arial','font.size':10,'svg.fonttype':'none','axes.linewidth':.8,'savefig.facecolor':'white'})
    legend=[Line2D([],[],marker=MARKERS[m],color=COLORS[m],lw=1,ls='--' if m=='full' or (m=='kimi' and not CLEAN_KIMI) else '-',markerfacecolor='white' if m in ('full','kimi','qwen80') else COLORS[m],markersize=8 if m=='ours' else 5,label=next(x['model'] for x in rows if x['baseline']==m)+(' (optimized B=3,4)' if m=='qwen' else (' (80GB)' if m in ('full','kimi') else ''))) for m in COLORS];label_checks=[]
    platform_note='80GB: SXM4 /400W; other GPU points: PCIe40GB /250W. Kimi B1/B6: same card; dashed DeepSeek full links: different cards.'
    if RENTAL:legend.insert(1,Line2D([],[],marker='o',color=COLORS['qwen'],markerfacecolor='white',lw=0,markersize=5,label='Qwen3-VL-2B B=1† (40GB reference)'))
    for kind,name in [('latency','latency_comparison_reference'),('energy','energy_comparison_reference')]:
        fig,ax=plt.subplots(figsize=(11.0,7.0) if RENTAL else (10.0,6.0));panel(ax,rows,kind)
        fig.subplots_adjust(left=.10,right=.95,top=.96,bottom=.30 if RENTAL else .24)
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=9)
        fig.text(.10,.035,'Independent resident utilization. YOLO B=3 / CLIP B=10: sequential model process. Qwen B=3,4: validated execution optimization.' if sequential else 'Labels: batch size and independent resident GPU utilization. Qwen B=3,4 use equivalence-validated execution optimization.',fontsize=8,color='#4B5563')
        fig.text(.10,.012,'† Qwen B=1 and LightGen-II are retained references. GPU costs normalize the first 200 TEST videos to 16 videos.',fontsize=8,color='#4B5563')
        if RENTAL:
            fig.text(.10,.045,platform_note if CLEAN_KIMI else '80GB series: A100-SXM4 /400W; other GPU points: PCIe40GB /250W. Dashed rental B1/B2: different card sessions.',fontsize=8,color='#4B5563')
            place_labels(fig,ax)
        label_checks+=verify_label_geometry(fig,[ax])
        for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=240)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(17.0,7.0) if RENTAL else (15.0,6.0),sharey=True)
    for ax,kind,label in zip(axes,['energy','latency'],['a','b']):
        panel(ax,rows,kind);ax.text(-.13,1.02,label,transform=ax.transAxes,fontweight='bold',fontsize=14)
    axes[1].set_ylabel('');fig.subplots_adjust(left=.065,right=.98,top=.95,bottom=.30 if RENTAL else .24,wspace=.16)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=9)
    fig.text(.065,.025,'GPU SRCC: 558 TEST. Inference costs normalized to 16. YOLO B=3 / CLIP B=10 utilization: sequential model process. LightGen-II and † Qwen B=1 retained.' if sequential else 'GPU SRCC: 558 TEST videos. Costs: first 200 normalized to 16. Independent resident utilization. LightGen-II and † Qwen B=1: retained references.',fontsize=8,color='#4B5563')
    if RENTAL:
        fig.text(.065,.050,platform_note if CLEAN_KIMI else '80GB series: A100-SXM4 /400W; other GPU points: PCIe40GB /250W. Rental B1/B2 use different cards; † denotes the retained40GB native Qwen B=1.',fontsize=8,color='#4B5563')
        for ax in axes:place_labels(fig,ax)
    label_checks+=verify_label_geometry(fig,axes)
    for name in ['srcc_cost_comparison_reference','batch_cost_comparison']:
        for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=240)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6.5) if RENTAL else (10,5.3));util_points=[]
    for m in COLORS:
        if m=='ours':continue
        points=[x for x in rows if x['baseline']==m]
        if m!='qwen':ax.plot([x['batch_size_videos'] for x in points],[x['display_gpu_util_mean_percent'] for x in points],lw=1,ls='--' if m=='full' or (m=='kimi' and not CLEAN_KIMI) else '-',alpha=.6,color=COLORS[m])
        for x in points:
            b=x['batch_size_videos'];u=x['display_gpu_util_mean_percent'];historic=m=='qwen' and b==1
            ax.scatter(b,u,facecolors='white' if m in ('full','kimi','qwen80') else COLORS[m],edgecolors=COLORS[m],marker=MARKERS[m],s=35);util_points.append((b,u))
            offsets_by_point={('qwen',1):(7,7),('qwen',3):(7,15),('qwen',4):(7,-17),
                ('glm',1):(-8,15),('glm',2):(-8,7),('small',1):(7,20),('small',3):(7,-20),('small',4):(7,5),
                ('clip',1):(7,-14),('clip',9):(-7,-16),('clip',10):(-7,-17),('clip',11):(-7,8),
                ('yolo',1):(7,8),('yolo',3):(-7,2),('yolo',4):(7,15),('deepseek',1):(7,-10),('deepseek',6):(7,-16),('deepseek',8):(7,8)}
            offset=offsets_by_point.get((m,b),(7,7))
            if RENTAL:ax.annotate(f'{u:.2f}%'+('†' if historic else ''),(b,u),xytext=((b-.7)/(max(x['batch_size_videos'] for x in rows if x['baseline']!='ours')-.1)+.04,(u-20)/85+.045),textcoords='axes fraction',ha='center',va='center',bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.88),arrowprops=dict(arrowstyle='-',color=COLORS[m],lw=.5,alpha=.6),color=COLORS[m],fontsize=8)
            else:ax.annotate(f'{u:.2f}%'+('†' if historic else ''),(b,u),xytext=offset,textcoords='offset points',ha='left' if offset[0]>0 else 'right',color=COLORS[m],fontsize=8)
    ax.axhline(98,color='#6B7280',ls='--',lw=.7);ax.text(.58,.93,'98% reference',transform=ax.transAxes,color='#6B7280',fontsize=8)
    maxbatch=max(x['batch_size_videos'] for x in rows if x['baseline']!='ours')
    ax.set_xlim(.7,maxbatch+.6);ax.set_ylim(20,105);ax.set_xticks(range(1,maxbatch+1))
    ax.set_xlabel('Video batch size');ax.set_ylabel('Resident GPU utilization (%)');ax.grid(color='#E5E7EB',lw=.5)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.subplots_adjust(left=.09,right=.97,bottom=.30 if RENTAL else .25,top=.95)
    fig.legend(handles=legend[1:],loc='lower center',bbox_to_anchor=(.5,.04),ncol=3,frameon=False,fontsize=9)
    if RENTAL:
        ax._audited_points=util_points;place_labels(fig,ax);label_checks+=verify_label_geometry(fig,[ax])
        fig.text(.09,.015,platform_note if CLEAN_KIMI else '80GB series: SXM4 /400W; other GPU points: PCIe40GB /250W. Dashed rental links span different cards; † retains native40GB Qwen B=1.',fontsize=8,color='#4B5563')
    for ext in ['png','svg']:fig.savefig(OUT/('gpu_utilization_comparison.'+ext),dpi=240)
    plt.close(fig)
    (RUN/'figure_label_audit.json').write_text(json.dumps(dict(status='passed',panels=label_checks,paper_points=len(rows),excluded_from_paper=[] if INCLUDE_CROSS_DEVICE_AUDIT else ['qwen80'],raw_presentation_and_cross_device_measurements_preserved=True),indent=2)+'\n')
    print(json.dumps(dict(figures_updated=5,selected_points=len(rows),historical_Qwen_B1_preserved=True)))


COMPACT_SOURCE = REPO/('LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_unified_inference_recheck_20261018/presentation/compact_groups_20261018.json' if '--unified-inference' in sys.argv else 'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_resident_busy50_a100_20261017/attempt03_remaining_20261018/service_presentation/compact_groups_20261018.json')
COMPACT_HIGH = dict(qwen=3, glm=2, clip=10, yolo=3, full=2, kimi=6)
COMPACT_COLORS = dict(ours='#7A3E9D', qwen='#0072B2', glm='#9B567C', clip='#008B70', yolo='#B17D00', full='#79502C', kimi='#D14F64')
COMPACT_MARKERS = dict(ours='*', qwen='o', glm='P', clip='D', yolo='^', full='s', kimi='X')


def compact_resident_metrics(repeats, batch):
    """Duration-weight power, sample-pool utilization, and retain every repeat."""
    assert isinstance(batch,int) and batch>0 and len(repeats)==3
    assert all(r['duration_s']>=30 and r['calls']>0 and r['samples']>0 for r in repeats)
    duration=sum(r['duration_s'] for r in repeats);calls=sum(r['calls'] for r in repeats)
    board=sum(r['duration_s']*r['board_w'] for r in repeats)/duration
    util=sum(r['samples']*r['utilization_percent'] for r in repeats)/sum(r['samples'] for r in repeats)
    time=duration*16000/(batch*calls)
    return dict(duration_s=duration,calls=calls,board_w=board,utilization_percent=util,
                time_ms=time,energy_j=(338.2+board)*time/1000)


def compact_group_data():
    """Two explicitly different cost protocols; no new GPU measurements."""
    if '--unified-inference' in sys.argv:
        return unified_inference_group_data()
    sources = {}
    def bind(file):
        file = Path(file)
        sources[file.relative_to(REPO).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        return file
    selected = json.loads(bind(REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_full_deepseek_kimi_rental_20261017/figure5_kimi_refresh_20261018/presentation_selected.json').read_text())['rows']
    paced = json.loads(bind(COMPACT_SOURCE.parent.parent/'verified_busy50_service_extension.json').read_text())['rows']
    bind(OUT/'plot_data.xlsx')
    bind(OUT/'plot_data.csv')
    names = {'ours': 'LightGen-II', 'qwen': 'Qwen3-VL-2B', 'glm': 'GLM-4.6V-Flash', 'clip': 'CLIP ViT-B/32', 'yolo': 'YOLO11s', 'full': 'DeepSeek-VL2', 'kimi': 'Kimi-VL-A3B-Instruct'}
    def base(x, condition, protocol):
        return dict(model=names[x['baseline']], baseline=x['baseline'], batch=x['batch_size_videos'],
                    condition=condition, srcc=x['measured_full558_srcc'],
                    utilization_percent=x.get('display_gpu_util_mean_percent'),
                    time_ms=x['time_16_videos_ms'], energy_j=x['energy_16_videos_j'],
                    platform=x.get('hardware_platform', 'Optical system'),
                    parameters_billion=x.get('model_total_parameters_billion'),
                    gpu_uuid=x.get('gpu_uuid'), checkpoint_sha256=x.get('checkpoint_sha256'),
                    cost_source=x.get('cost_source'), utilization_source=x.get('utilization_source'),
                    accuracy_source=x.get('cost_source'), accuracy_basis='Independent full558 conventional evaluation',
                    protocol=protocol, plotted=True, repeats=[], repeat_inputs=[],
                    color=COMPACT_COLORS[x['baseline']], marker=COMPACT_MARKERS[x['baseline']],
                    link_style='dashed' if x['baseline'] in ('qwen','full') else 'solid',
                    notes='')
    ours = next(x for x in selected if x['baseline']=='ours')
    conventional=[base(ours, 'Optical reference', 'Approved optical component-cost reference')]
    conventional[0]['notes']='N=16; approximately 50% spatial use is author-provided, not GPU temporal busy time.'
    resident=[dict(conventional[0])]
    for model, high in COMPACT_HIGH.items():
        points=[x for x in selected if x['baseline']==model and x['batch_size_videos'] in (1,high)]
        assert len(points)==2
        for x in points:
            row=base(x, 'B=1 reference' if x['batch_size_videos']==1 else 'Near-saturation reference', 'Dataset-once inference: first200 normalized to16')
            if model=='qwen':row['notes']='B1 finalized native anchor; B3 equivalence-validated execution optimization. Dashed link is descriptive, not batch-only.'
            if model=='full':row['notes']='B1 and B2 measured on different physical 80GB cards; descriptive dashed link.'
            conventional.append(row)
        x=next(v for v in points if v['batch_size_videos']==high)
        file=bind(REPO/x['utilization_source']);report=json.loads(file.read_text())
        assert report['status']=='complete' and len(report['repeats'])==3
        repeats=[]
        for i,r in enumerate(report['repeats'],1):
            m=r.get('measurement',r)
            calls=r.get('measured_forward_invocations',r.get('measured_forwards',r.get('forward_calls')))
            assert isinstance(calls,int) and calls>0 and m['duration_s']>=30
            telemetry=file.parent/(f'repeat{i:02d}/telemetry.csv' if model!='kimi' else 'telemetry.csv')
            raw=list(csv.DictReader(bind(telemetry).open(newline='')))
            a,b=m['start_monotonic_s'],m['end_monotonic_s']
            values=[v for v in raw if a<=float(v['host_monotonic_s'])<=b]
            assert len(values)==m['samples']
            watts=sum(float(v['watts']) for v in values)/len(values)
            util=sum(float(v['utilization_percent']) for v in values)/len(values)
            assert math.isclose(watts,m['mean_board_power_w'],abs_tol=1e-8)
            assert math.isclose(util,m['mean_gpu_utilization_percent'],abs_tol=1e-8)
            repeats.append(dict(duration_s=m['duration_s'], calls=calls, board_w=watts, utilization_percent=util, samples=len(values)))
        metrics=compact_resident_metrics(repeats,high)
        assert math.isclose(metrics['utilization_percent'],x['display_gpu_util_mean_percent'],abs_tol=1e-8)
        full=base(x,'Near-saturation','Resident service; complete measured windows')
        full.update(**metrics,host_proxy_w=338.2,
                    repeats=[r['utilization_percent'] for r in repeats],repeat_inputs=repeats,
                    cost_source=file.relative_to(REPO).as_posix(),link_style='solid',
                    notes='Costs recomputed from the same resident windows as utilization; not conventional first200 costs. Separate dates and process histories remain disclosed.')
        low=next((r for r in paced if r['baseline']==model and r['batch']==high),None)
        if low is not None:
            assert low['target50_reached'] and .45<=low['mean_busy_percent']/100<=.55
            if model=='kimi':
                identity=json.loads(bind((REPO/low['source']).parent.parent/'identity.json').read_text())
                low_head=identity['checkpoint_sha256']
            else:low_head=low['checkpoint_sha256']
            assert low_head==x['checkpoint_sha256']
            row=base(x,'Approximately 50%','Resident service; complete measured windows including pacing waits')
            row.update(utilization_percent=low['mean_busy_percent'],time_ms=low['service_T16_ms'],energy_j=low['service_E16_j'],
                       board_w=low['board_mean_w'],host_proxy_w=low['host_proxy_w'],duration_s=low['duration_s'],
                       calls=low['completed_forward_calls'],repeats=low['repeat_means_percent'],cost_source=low['source'],
                       utilization_source=low['source'],gpu_uuid=low['gpu_uuid'],link_style='solid',
                       notes='SRCC is a separate full558 reference, not re-evaluated on558 distinct videos under pacing. All imposed waits included.')
            assert row['gpu_uuid']==full['gpu_uuid'], 'Same-batch pair must use same physical GPU'
            resident.append(row)
        else:
            assert model=='full'
            missing=base(x,'Approximately 50%: unavailable','Not measured')
            missing.update(time_ms=None,energy_j=None,utilization_percent=None,srcc=None,plotted=False,
                           notes='No paced50% DeepSeek-VL2 full measurement exists. Tiny/Small cannot substitute.')
            resident.append(missing)
        resident.append(full)
    assert conventional[1]['utilization_percent']==63.133111480865225
    assert conventional[1]['time_ms']==1188.3861464262009
    assert conventional[1]['energy_j']==462.1134010767563
    payload=dict(status='complete',groups={'batch1_near_saturation':conventional,'busy50_near_saturation':resident},
                 source_sha256=sources,figure_width_mm=78,figure_height_mm=88,
                 omitted_models=['DeepSeek-VL2-Tiny','DeepSeek-VL2-Small','Rental Qwen80GB'],
                 missing_measurements=['DeepSeek-VL2 full paced50%'],
                 optical_spatial_use_basis='Author estimate approximately50%; not measured GPU temporal busy time')
    COMPACT_SOURCE.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(group_rows={k:len(v) for k,v in payload['groups'].items()},raw_window_replays=18,
                          missing=payload['missing_measurements'],no_GPU_rerun=True)))


def unified_inference_group_data():
    """Replay all ten raw runs; retain protected anchors and independent utilization."""
    from scipy.stats import spearmanr, pearsonr
    root=COMPACT_SOURCE.parent.parent
    archive=COMPACT_SOURCE.parent/'prior_report_archive'
    archive.mkdir(parents=True,exist_ok=True)
    archive_manifest=archive/'manifest.json'
    if not archive_manifest.exists():
        saved={}
        names=['README.md','LGVQ_batch1_near_saturation_plot_data.xlsx','LGVQ_busy50_near_saturation_plot_data.xlsx']
        for group in ('batch1_near_saturation','busy50_near_saturation'):
            names += [f'{group}_{kind}.{ext}' for kind in ('latency','energy') for ext in ('png','svg','pdf')]+[group+'_pair.png']
        for name in names:
            data=(OUT/name).read_bytes()
            with (archive/name).open('xb') as stream:stream.write(data)
            saved[name]=hashlib.sha256(data).hexdigest()
        archive_manifest.write_text(json.dumps(saved,indent=2)+'\n')
    else:
        for name,sha in json.loads(archive_manifest.read_text()).items():
            assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==sha
    old=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_resident_busy50_a100_20261017/attempt03_remaining_20261018/service_presentation/compact_groups_20261018.json'
    payload=json.loads(old.read_text());sources=dict(payload['source_sha256'])
    def bind(p):
        sources[p.relative_to(REPO).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    bind(old)
    state=json.loads(bind(root/'suite_status.json').read_text())
    folders={name:root/name for name in state['completed']}
    if state['status']=='failed':
        assert len(state['completed'])==7
        resumed=json.loads(bind(root/'resume03_20261018/suite_status.json').read_text())
        wrapper_name='LightGenV2/tasks/t06_video_quality_assessment/resume_unified_inference_recheck.py'
        wrapper=bind(REPO/wrapper_name)
        if resumed['status']=='complete':
            assert len(resumed['completed'])==3
            assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==resumed['continuation_source_sha256']
        else:
            assert resumed['status']=='failed' and resumed['completed']==['glm_paced_b02_r01']
            old_wrapper=subprocess.check_output(['git','show','18a818d78:'+wrapper_name],cwd=REPO)
            assert hashlib.sha256(old_wrapper).hexdigest()==resumed['continuation_source_sha256']
            last=json.loads(bind(root/'resume02_20261018/suite_status.json').read_text())
            assert last['status']=='complete' and last['completed']==['clip_paced_b10_r01','yolo_paced_b03_r01']
            assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==last['continuation_source_sha256']
            for name,sha in last['protected_original_sha256'].items():
                assert hashlib.sha256(bind(root/name).read_bytes()).hexdigest()==sha
            folders.update({name:root/'resume02_20261018'/name for name in last['completed']})
        for name,sha in resumed['protected_original_sha256'].items():
            assert hashlib.sha256(bind(root/name).read_bytes()).hexdigest()==sha
        folders.update({name:root/'resume03_20261018'/name for name in resumed['completed']})
    else:assert state['status']=='complete' and len(state['completed'])==10
    assert len(folders)==10
    if len(state['completed'])==7:
        ordered=state['completed']+['glm_paced_b02_r01','clip_paced_b10_r01','yolo_paced_b03_r01']
        assert set(ordered)==set(folders)
        folders={name:folders[name] for name in ordered}
    manifest=bind(REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_reproduction_a100_20261013/lgvq_train2250_test558.csv')
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
    expected=[r for r in csv.DictReader(manifest.open(newline='',encoding='utf-8-sig')) if r['split']=='test']
    assert len(expected)==558
    audited={};line_endings={};verified_sources=set()
    for name,folder in folders.items():
        report=json.loads(bind(folder/'report.json').read_text())
        identity=json.loads(bind(folder/'identity.json').read_text())
        prediction=list(csv.DictReader(bind(folder/'predictions.csv').open(newline='')))
        timing=list(csv.DictReader(bind(folder/'batch_timing.csv').open(newline='')))
        raw=list(csv.DictReader(bind(folder/'telemetry.csv').open(newline='')))
        bind(folder/'telemetry_stdout.jsonl')
        process_audit=json.loads(bind(folder/'process_audit.json').read_text())
        assert process_audit and all(str(p['pid'])==str(report['pid']) for a in process_audit for p in a['processes'])
        waits=json.loads(bind(folder/'excluded_pacing_waits.json').read_text())
        assert report['status']=='complete' and len(prediction)==558
        assert len({p['sample_id'] for p in prediction})==558
        assert [int(p['sample_index']) for p in prediction]==list(range(558))
        assert [p['sample_id'] for p in prediction]==[p['sample_id'] for p in expected]
        assert all(float(a['target_mos'])==float(b['temporal']) for a,b in zip(prediction,expected))
        assert sum(int(t['videos']) for t in timing)==558
        cost=[t for t in timing if t['in_cost_subset']=='True']
        assert sum(int(t['videos']) for t in cost)==200
        assert report['actual_checkpoint_verified'] and report['manifest_sha256']==identity['manifest_sha256']
        for p,h in identity['source_sha256'].items():
            raw_source=(REPO/p).read_bytes();local_sha=hashlib.sha256(raw_source).hexdigest()
            if (p,h) not in verified_sources and local_sha!=h:
                blob=subprocess.check_output(['git','show',identity['git_commit']+':'+p],cwd=REPO)
                assert hashlib.sha256(blob).hexdigest()==h and raw_source.replace(b'\r\n',b'\n')==blob, p
                line_endings[p]=dict(local_byte_sha256=local_sha,measured_LF_sha256=h,
                                     exact_measured_git_blob=identity['git_commit']+':'+p,only_difference='CRLF versus LF')
            sources[p]=local_sha;verified_sources.add((p,h))
        active=[v for v in raw if v['phase'].startswith('active:')]
        assert active
        cost_indices={int(v['batch_index']) for v in cost}
        assert all(int(v['phase'].split(':')[1]) in cost_indices for v in active)
        watts=sum(float(v['watts']) for v in active)/len(active)
        wall=sum(float(t['synchronized_wall_ms']) for t in cost)
        t16=wall*16/200;e16=(338.2+watts)*t16/1000
        srcc=float(spearmanr([float(p['target_mos']) for p in prediction],[float(p['prediction']) for p in prediction]).statistic)
        plcc=float(pearsonr([float(p['target_mos']) for p in prediction],[float(p['prediction']) for p in prediction]).statistic)
        for actual,metric_expected in [(t16,report['cost']['latency_16_videos_ms']),(e16,report['cost']['energy_16_videos_j']),
                                (watts,report['power']['active_mean_w']),(srcc,report['performance']['srcc']),(plcc,report['performance']['plcc'])]:
            assert math.isclose(actual,metric_expected,rel_tol=1e-10,abs_tol=1e-9)
        independent=report['independent_utilization'];util=bind(REPO/independent['utilization_source'])
        assert hashlib.sha256(util.read_bytes()).hexdigest()==independent['utilization_source_sha256']
        assert len(waits)==len(timing) and report['explicit_warmup_forwards']==0 and report['first_test_batch_included']
        excluded=sum(w['end_monotonic_s']-w['start_monotonic_s'] for w in waits)
        excluded_cost=sum(w['end_monotonic_s']-w['start_monotonic_s'] for w in waits if w['in_cost_subset'])
        assert math.isclose(excluded,report['excluded_pacing_total_s'],abs_tol=1e-8)
        requested=sum(w['requested_s'] for w in waits if w['in_cost_subset'])
        audited[name]=dict(name=name,baseline=report['baseline'],batch=report['batch'],kind=report['kind'],
            repeat=report['repeat'],time_ms=t16,energy_j=e16,board_w=watts,active_samples=len(active),
            cost_wall_sum_ms=wall,cost_video_count=200,srcc=srcc,plcc=plcc,
            first_call_ms=float(timing[0]['synchronized_wall_ms']),excluded_wait_total_s=excluded,
            excluded_cost_wait_s=excluded_cost,excluded_wait_16_ms=excluded_cost*1000*16/200,
            requested_cost_wait_s=requested,checkpoint_sha256=report['checkpoint_sha256'],
            utilization_percent=independent['independent_util_percent'],utilization_source=independent['utilization_source'],
            cost_source=(folder/'report.json').relative_to(REPO).as_posix(),host_proxy_w=338.2)
    conventional=payload['groups']['batch1_near_saturation']
    for row in conventional:
        if row['baseline']=='ours' or (row['baseline']=='qwen' and row['batch']==1):continue
        util_report=json.loads(bind(REPO/row['utilization_source']).read_text())
        repeats=[r.get('measurement',r) for r in util_report['repeats']]
        assert len(repeats)==3
        row['repeats']=[r['mean_gpu_utilization_percent'] for r in repeats]
        pooled=sum(r['samples']*r['mean_gpu_utilization_percent'] for r in repeats)/sum(r['samples'] for r in repeats)
        assert math.isclose(pooled,row['utilization_percent'],abs_tol=1e-8)
    # Rental models are audited locally only. No conversion from resident costs.
    for row in conventional:
        if row['baseline'] not in ('full','kimi'):continue
        report_file=bind(REPO/row['cost_source']);report=json.loads(report_file.read_text())
        timing=list(csv.DictReader(bind(report_file.parent/'batch_timing.csv').open(newline='')))
        raw=list(csv.DictReader(bind(report_file.parent/'telemetry.csv').open(newline='')))
        predictions=list(csv.DictReader(bind(report_file.parent/'predictions.csv').open(newline='')))
        cost=[r for r in timing if r['in_cost_subset']=='True'];active=[r for r in raw if r['phase'].startswith('active:')]
        assert len(predictions)==558 and sum(int(r['videos']) for r in cost)==200
        assert [r['sample_id'] for r in predictions]==[r['sample_id'] for r in expected]
        wall=sum(float(r['synchronized_wall_ms']) for r in cost)
        board=sum(float(r['watts']) for r in active)/len(active)
        assert math.isclose(row['time_ms'],wall*16/200,abs_tol=1e-8)
        assert math.isclose(row['energy_j'],(338.2+board)*wall*16/200/1000,abs_tol=1e-8)
        assert report['explicit_warmup_forwards']==0
        row.update(cost_wall_sum_ms=wall,cost_video_count=200,board_w=board,host_proxy_w=338.2,active_samples=len(active))
    for b in (1,10):
        values=[x for x in audited.values() if x['baseline']=='clip' and x['kind']=='native' and x['batch']==b]
        assert len(values)==3
        row=next(r for r in conventional if r['baseline']=='clip' and r['batch']==b)
        row.update(time_ms=sum(x['time_ms'] for x in values)/3,energy_j=sum(x['energy_j'] for x in values)/3,
            srcc=sum(x['srcc'] for x in values)/3,board_w=sum(x['board_w'] for x in values)/3,
            cost_source='; '.join(x['cost_source'] for x in values),accuracy_source='; '.join(x['cost_source'] for x in values),
            cost_wall_sum_ms=sum(x['cost_wall_sum_ms'] for x in values),cost_video_count=600,
            native_repeat_times_ms=[x['time_ms'] for x in values],native_repeat_energies_j=[x['energy_j'] for x in values],
            repeat_inputs=values,energy_aggregation='Arithmetic mean of three separately paired energy estimates',
            notes='Three cold fresh-process repeats, all retained; the same200 unique cost videos in each repeat,600 evaluations not600 unique videos. Sparse50ms power samples.')
    resident=[dict(conventional[0])]
    for model,b in COMPACT_HIGH.items():
        high=dict(next(r for r in conventional if r['baseline']==model and r['batch']==b))
        high.update(condition='Near-saturation reference')
        low=dict(high);low.update(condition='Approximately 50%',link_style='solid')
        if model in ('full','kimi'):
            low.update(plotted=False,time_ms=None,energy_j=None,srcc=None,utilization_percent=None,
                       cost_source=None,accuracy_source=None,utilization_source=None,board_w=None,
                       cost_wall_sum_ms=None,cost_video_count=None,active_samples=None,
                       notes='No compliant dataset-once paced cost; existing resident service cost is not substituted. No remeasurement authorized.')
        else:
            x=next(v for v in audited.values() if v['baseline']==model and v['kind']=='paced')
            assert x['checkpoint_sha256']==high['checkpoint_sha256']
            independent=json.loads(bind(REPO/x['utilization_source']).read_text())
            low.update(**{k:v for k,v in x.items() if k not in ('name','baseline','batch','kind','repeat')},
                       accuracy_source=x['cost_source'],repeats=independent['repeat_means_percent'],repeat_inputs=[x],
                       notes='558 distinct TEST predictions re-evaluated; inter-inference pacing excluded. Displayed utilization is independent resident measurement, not cost-run utilization.')
        for row in (low,high):
            row['protocol']='Dataset-once first200 inference boundary; pacing excluded; independent utilization'
            row.pop('duration_s',None);row.pop('calls',None)
        resident.extend([low,high])
    anchor=next(r for r in conventional if r['baseline']=='qwen' and r['batch']==1)
    assert (anchor['time_ms'],anchor['energy_j'],anchor['utilization_percent'])==(1188.3861464262009,462.1134010767563,63.133111480865225)
    payload.update(groups={'batch1_near_saturation':conventional,'busy50_near_saturation':resident},source_sha256=sources,
        missing_measurements=['DeepSeek-VL2 full paced dataset-once cost','Kimi paced dataset-once cost'],
        all_costs_dataset_once=True,pacing_waits_excluded=True,independent_utilization_not_same_cost_window=True,
        audit_points=list(audited.values()),protected_qwen_B1_unchanged=True)
    payload['local_line_ending_equivalence']=line_endings
    COMPACT_SOURCE.parent.mkdir(exist_ok=True)
    COMPACT_SOURCE.write_text(json.dumps(payload,indent=2)+'\n')
    for group,rows in payload['groups'].items():
        with (OUT/f'LGVQ_{group}_plot_data.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            fields=['model','condition','batch','srcc','utilization_percent','time_ms','energy_j','board_w','host_proxy_w',
                    'cost_wall_sum_ms','cost_video_count','active_samples','excluded_wait_total_s','parameters_billion',
                    'excluded_cost_wait_s','excluded_wait_16_ms',
                    'platform','gpu_uuid','plotted','cost_source','utilization_source','accuracy_source','checkpoint_sha256','protocol','notes']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (root/'verified_unified_inference.json').write_text(json.dumps(dict(status='complete',points=list(audited.values()),
        point_count=10,protected_qwen_B1_unchanged=True,source_sha256=sources,
        local_line_ending_equivalence=line_endings),indent=2)+'\n')
    print(json.dumps(dict(audited_points=10,unique_test_videos_per_point=558,all_repeats_retained=True)))


def compact_point_label(row):
    return '~50% spatial' if row['baseline']=='ours' else f"{row['utilization_percent']:.1f}%"


def compact_panel(fig,ax,rows,group,kind):
    key='energy_j' if kind=='energy' else 'time_ms'
    rows=[r for r in rows if r['plotted']]
    xmin=min(r[key] for r in rows)*.62;xmax=max(r[key] for r in rows)*1.6
    ax.set_xscale('log');ax.set_xlim(xmin,xmax);ax.set_ylim(.681,.843)
    ax.set_yticks([.70,.75,.80]);ax.set_ylabel('Temporal quality (SRCC)',labelpad=3)
    ax.set_xlabel(('Estimated energy (J / 16 videos)' if kind=='energy' else 'Latency (ms / 16 videos)') if group.startswith('batch1') or '--unified-inference' in sys.argv else
                  ('Estimated service energy (J / 16 videos)' if kind=='energy' else 'Service time (ms / 16 videos)'),labelpad=3)
    ax.grid(axis='both',which='major',color='#DBDFE3',lw=.45,zorder=0)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out',length=2.5,width=.6,pad=2,labelsize=7)
    ax._audited_points=[];annotations=[]
    for model in COMPACT_COLORS:
        points=[r for r in rows if r['baseline']==model]
        if not points:continue
        if len(points)>1:
            ax.plot([r[key] for r in points],[r['srcc'] for r in points],color=COMPACT_COLORS[model],
                    ls='--' if points[0]['link_style']=='dashed' else '-',lw=.9,alpha=.75,zorder=2)
        for r in points:
            x,y=r[key],r['srcc'];ax._audited_points.append((x,y))
            low=r['condition']=='Approximately 50%'
            ax.scatter(x,y,marker=r['marker'],s=49 if model=='ours' else 20,
                       facecolors='white' if low else r['color'],edgecolors=r['color'],linewidths=.8,zorder=4)
            label=compact_point_label(r)
            annotations.append(ax.annotate(label,(x,y),xytext=(.5,.5),textcoords='axes fraction',
                               fontsize=6.5,color=r['color'],ha='center',va='center',
                               bbox=dict(boxstyle='square,pad=.08',fc='white',ec='none',alpha=.96),
                               arrowprops=dict(arrowstyle='-',lw=.45,color=r['color'],alpha=.65),zorder=5))
    # Bounded label packing at the final physical figure size. Every endpoint is kept.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=ax.get_window_extent(renderer)
    placed=[];leaders=[];xy=[ax.transData.transform(p) for p in ax._audited_points]
    connectors=[PlotPath(line.get_transform().transform(line.get_xydata())) for line in ax.lines]
    grid=[(x,y) for x in np.linspace(.08,.92,31) for y in np.linspace(.06,.94,36)]
    for ann in annotations:
        old=ax.transAxes.transform(ann.get_position());box=ann.get_bbox_patch().get_window_extent(renderer)
        target=ax.transData.transform(ann.xy)
        choices=sorted(grid,key=lambda p:np.linalg.norm((ax.transAxes.transform(p)-target)*[1,1.2]))
        for p in choices:
            candidate=box.translated(*(ax.transAxes.transform(p)-old))
            if not(bounds.contains(candidate.x0,candidate.y0) and bounds.contains(candidate.x1,candidate.y1)):continue
            if any(candidate.padded(1.0).overlaps(v) for v in placed):continue
            if any(candidate.padded(2).contains(*v) for v in xy):continue
            if any(line.intersects_bbox(candidate.padded(1.5),filled=False) for line in connectors):continue
            if any(line.intersects_bbox(candidate.padded(.6),filled=False) for line in leaders):continue
            leader=PlotPath([target,ax.transAxes.transform(p)])
            if any(leader.intersects_bbox(box.padded(.6),filled=False) for box in placed):continue
            ann.set_position(p);placed.append(candidate);leaders.append(leader);break
        else:raise AssertionError('Compact labels do not fit at physical width78mm')
    fig.canvas.draw()
    return verify_label_geometry(fig,[ax],check_lines=True)


def compact_group_figures():
    data=json.loads(COMPACT_SOURCE.read_text());assert data['status']=='complete'
    for name,sha in data['source_sha256'].items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==sha, 'Source changed: '+name
    plt.rcParams.update({'font.family':'Arial','font.size':7.2,'svg.fonttype':'none','axes.linewidth':.6,
                         'savefig.facecolor':'white','pdf.fonttype':42})
    audits=[]
    for group,rows in data['groups'].items():
        handles=[Line2D([],[],marker=COMPACT_MARKERS[m],color=COMPACT_COLORS[m],lw=.8,
                        ls='--' if group.startswith('batch1') and m in ('qwen','full') else '-',
                        markersize=4,label='Kimi-VL-A3B' if m=='kimi' else next(r['model'] for r in rows if r['baseline']==m)) for m in COMPACT_COLORS]
        for kind in ('latency','energy'):
            fig,ax=plt.subplots(figsize=(78/25.4,88/25.4))
            fig.subplots_adjust(left=.19,right=.97,top=.965,bottom=.35)
            audits+=compact_panel(fig,ax,rows,group,kind)
            legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.07),ncol=2,
                       frameon=False,fontsize=6.5,columnspacing=.9,handlelength=1.4,labelspacing=.45)
            if group.startswith('batch1'):
                footer=fig.text(.19,.009,'Dataset-once costs; independent utilization labels.\nDashed: execution or GPU-session differences.',fontsize=5.3,color='#526170')
            elif '--unified-inference' in sys.argv:
                footer=fig.text(.19,.009,'Inference costs; pacing waits excluded.\nUtilization measured independently.\nDS/Kimi50% costs unavailable.',fontsize=5.3,color='#526170')
            else:
                footer=fig.text(.19,.009,'Resident service, not dataset-once inference.\nOpen: paced; filled: native. Optical: reference only.\nOptical uses a different protocol; DS50% unavailable.',fontsize=5.3,color='#526170')
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            legend_box=legend.get_window_extent(renderer)
            assert not legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer)), 'Legend covers axis title'
            assert not legend_box.overlaps(footer.get_window_extent(renderer)), 'Legend covers protocol note'
            for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{group}_{kind}.{ext}',dpi=600)
            plt.close(fig)
        # Same-size placement preview: two78-mm panels form one156-mm Word row.
        from PIL import Image
        first=Image.open(OUT/f'{group}_latency.png');second=Image.open(OUT/f'{group}_energy.png')
        paired=Image.new('RGB',(first.width+second.width,first.height),'white')
        paired.paste(first,(0,0));paired.paste(second,(first.width,0))
        paired.save(OUT/f'{group}_pair.png',dpi=(600,600))
    receipt=dict(status='passed',panels=audits,individual_width_mm=78,individual_height_mm=88,
                 identical_marker_by_model=True,all_available_same_model_points_connected=True,
                 low50_hollow_high_native_filled=True,group1_qwen_and_deepseek_descriptive_dashed_links=True,
                 utilization_only_point_labels=True,batch_and_optical_count_not_annotated=True,
                 group2_costs_all_resident_full_window='--unified-inference' not in sys.argv,
                 all_costs_dataset_once='--unified-inference' in sys.argv,excluded=data['omitted_models'],
                 missing=data['missing_measurements'],source_sha256=hashlib.sha256(COMPACT_SOURCE.read_bytes()).hexdigest())
    (COMPACT_SOURCE.parent/'compact_figure_audit_20261018.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(figures=4,formats=['png','svg','pdf'],geometry_audit='passed',width_mm=78)))


if __name__=='__main__':main()
