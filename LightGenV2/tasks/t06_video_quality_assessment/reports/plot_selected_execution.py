"""Replot verified sparse figure points, never reconstruct model measurements."""
from __future__ import annotations
import json
import hashlib
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO=Path(__file__).resolve().parents[4]
RUN=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_selected_execution_a100_20261016'
if '--yolo-clip-recheck' in sys.argv:RUN=RUN.with_name('lgvq_yolo_clip_recheck_a100_20261017')
if '--yolo-clip-sequential' in sys.argv:RUN=RUN.with_name('lgvq_yolo_clip_sequential_a100_20261017')
RENTAL='--rental-extension' in sys.argv
INCLUDE_CROSS_DEVICE_AUDIT='--include-cross-device-audit' in sys.argv
if RENTAL:RUN=RUN.with_name('lgvq_full_deepseek_kimi_rental_20261017')
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
        if len(connected)>1:ax.plot([x[key] for x in connected],[x['measured_full558_srcc'] for x in connected],color=color,lw=1,ls='--' if model in ('full','kimi') else '-',alpha=.55,zorder=2)
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


def verify_label_geometry(fig,axes):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();checks=[]
    for ax in axes:
        boxes=[text.get_bbox_patch().get_window_extent(renderer) for text in ax.texts if text.get_bbox_patch() is not None]
        assert len(boxes)==len(ax._audited_points)
        bounds=ax.get_window_extent(renderer)
        for i,box in enumerate(boxes):
            assert bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1),'Label outside axes'
            assert all(not box.padded(1).overlaps(other) for other in boxes[i+1:]),'Overlapping data labels'
            assert all(not box.padded(2).contains(*ax.transData.transform(point)) for point in ax._audited_points),f'Label covers data marker: annotation {i}'
        checks.append(dict(labels=len(boxes),overlaps=0,covered_markers=0,all_labels_within_axes=True,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())))
    return checks


def main():
    state=json.loads((RUN/'presentation_selected.json').read_text());assert state['status']=='complete'
    rows=paper_rows(state['rows']);assert 13<=len(rows)<=(20 if RENTAL else 18)
    if RENTAL and not INCLUDE_CROSS_DEVICE_AUDIT:
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
    legend=[Line2D([],[],marker=MARKERS[m],color=COLORS[m],lw=1,ls='--' if m in ('full','kimi') else '-',markerfacecolor='white' if m in ('full','kimi','qwen80') else COLORS[m],markersize=8 if m=='ours' else 5,label=next(x['model'] for x in rows if x['baseline']==m)+(' (optimized B=3,4)' if m=='qwen' else (' (80GB)' if m in ('full','kimi') else ''))) for m in COLORS];label_checks=[]
    if RENTAL:legend.insert(1,Line2D([],[],marker='o',color=COLORS['qwen'],markerfacecolor='white',lw=0,markersize=5,label='Qwen3-VL-2B B=1† (40GB reference)'))
    for kind,name in [('latency','latency_comparison_reference'),('energy','energy_comparison_reference')]:
        fig,ax=plt.subplots(figsize=(11.0,7.0) if RENTAL else (10.0,6.0));panel(ax,rows,kind)
        fig.subplots_adjust(left=.10,right=.95,top=.96,bottom=.30 if RENTAL else .24)
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=9)
        fig.text(.10,.035,'Independent resident utilization. YOLO B=3 / CLIP B=10: sequential model process. Qwen B=3,4: validated execution optimization.' if sequential else 'Labels: batch size and independent resident GPU utilization. Qwen B=3,4 use equivalence-validated execution optimization.',fontsize=8,color='#4B5563')
        fig.text(.10,.012,'† Qwen B=1 and LightGen-II are retained references. GPU costs normalize the first 200 TEST videos to 16 videos.',fontsize=8,color='#4B5563')
        if RENTAL:
            fig.text(.10,.045,'80GB series: A100-SXM4 /400W; other GPU points: PCIe40GB /250W. Dashed rental B1/B2: different card sessions.',fontsize=8,color='#4B5563')
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
        fig.text(.065,.050,'80GB series: A100-SXM4 /400W; other GPU points: PCIe40GB /250W. Rental B1/B2 use different cards; † denotes the retained40GB native Qwen B=1.',fontsize=8,color='#4B5563')
        for ax in axes:place_labels(fig,ax)
    label_checks+=verify_label_geometry(fig,axes)
    for name in ['srcc_cost_comparison_reference','batch_cost_comparison']:
        for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=240)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6.5) if RENTAL else (10,5.3));util_points=[]
    for m in COLORS:
        if m=='ours':continue
        points=[x for x in rows if x['baseline']==m]
        if m!='qwen':ax.plot([x['batch_size_videos'] for x in points],[x['display_gpu_util_mean_percent'] for x in points],lw=1,ls='--' if m in ('full','kimi') else '-',alpha=.6,color=COLORS[m])
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
        fig.text(.09,.015,'80GB series: SXM4 /400W; other GPU points: PCIe40GB /250W. Dashed rental links span different cards; † retains native40GB Qwen B=1.',fontsize=8,color='#4B5563')
    for ext in ['png','svg']:fig.savefig(OUT/('gpu_utilization_comparison.'+ext),dpi=240)
    plt.close(fig)
    (RUN/'figure_label_audit.json').write_text(json.dumps(dict(status='passed',panels=label_checks,paper_points=len(rows),excluded_from_paper=[] if INCLUDE_CROSS_DEVICE_AUDIT else ['qwen80'],raw_presentation_and_cross_device_measurements_preserved=True),indent=2)+'\n')
    print(json.dumps(dict(figures_updated=5,selected_points=len(rows),historical_Qwen_B1_preserved=True)))


if __name__=='__main__':main()
