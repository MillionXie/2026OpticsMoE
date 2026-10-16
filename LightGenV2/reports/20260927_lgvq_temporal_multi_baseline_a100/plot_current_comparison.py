"""Scientific figures from independently verified frozen costs; no model runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--preview-partial',action='store_true')
p.add_argument('--cost-only',action='store_true',help='Leave the independently measured utilization figure unchanged')
args=p.parse_args()
repo=Path(__file__).resolve().parents[3]
run=repo/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_frozen_batch_costs_a100_20261015'
data=json.loads((run/'verified_costs.json').read_text())
display_path=run/'presentation_update_20261016.json'
display=json.loads(display_path.read_text()) if display_path.exists() else None
selected_util={(r['model'],r['batch_size_videos']):r['display_gpu_util_mean_percent'] for r in display['rows']} if display else {}
if data['status']!='complete' and not args.preview_partial:
    raise RuntimeError('Do not publish incomplete measurements')
out=repo/'.codex_tmp' if args.preview_partial else repo/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
names=['Qwen3-VL-2B','GLM-4.6V-Flash','DeepSeek-VL2-Small','CLIP ViT-B/32','YOLO11s','DeepSeek-VL2-Tiny']
colors=['#0072B2','#CC79A7','#D55E00','#009E73','#E69F00','#5B6470']
markers=['o','P','s','D','^','v']
expected_batches={'Qwen3-VL-2B':[1,2,3,4,6,8,10],'GLM-4.6V-Flash':[1,2],
    'DeepSeek-VL2-Small':[1,2,3,4],'CLIP ViT-B/32':[1,2,3,4,6,8,9,10],
    'YOLO11s':[1,2,3],'DeepSeek-VL2-Tiny':[1,2,3,4,6,8]}
if not args.preview_partial:
    if not display or display['status']!='complete':
        raise RuntimeError('Require the approved presentation data, including finalized Qwen B1')
    if len(data['paired_rows'])!=30 or {x['model'] for x in data['paired_rows']}!=set(names):
        raise RuntimeError('Incomplete model coverage')
    with (out/'plot_data.csv').open(newline='',encoding='utf-8-sig') as f:
        plot_rows={(x['model'],int(x['batch_size_videos']) if x['batch_size_videos'] else None):x for x in csv.DictReader(f)}
    for name,batches in expected_batches.items():
        if [x['batch_size'] for x in data['paired_rows'] if x['model']==name]!=batches:
            raise RuntimeError('Missing model/batch point: '+name)
    for x in data['paired_rows']:
        record=plot_rows[(x['model'],x['batch_size'])]
        for key,source in [('time_16_videos_ms','time_16_videos_ms'),('energy_16_videos_j','energy_16_videos_j'),('measured_full558_srcc','srcc')]:
            if abs(float(record[key])-x[source])>1e-9:raise RuntimeError('CSV/source mismatch')
        if abs(float(record['display_gpu_util_mean_percent'])-selected_util[(x['model'],x['batch_size'])])>1e-9:
            raise RuntimeError('Selected utilization mismatch')
    archive=run/'presentation_prior_20261016'
    for stem in ['srcc_cost_comparison_reference','latency_comparison_reference','energy_comparison_reference']:
        for ext in ['png','svg']:
            source=out/(stem+'.'+ext)
            if source.exists():
                digest=hashlib.sha256(source.read_bytes()).hexdigest()
                preserved=archive/(digest[:12]+'_'+source.name)
                if not preserved.exists():shutil.copyfile(source,preserved)
                if hashlib.sha256(preserved.read_bytes()).hexdigest()!=digest:raise RuntimeError('Archive differs')
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':11,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})


def place_labels(ax,points,fig):
    """Fixed candidate offsets, all points labelled, no data-dependent omission."""
    renderer=fig.canvas.get_renderer();font=FontProperties(family='Arial',size=8)
    occupied=[]
    locations=[ax.transData.transform((v['x'],v['y'])) for v in points]
    obstacle=[Bbox.from_bounds(x-5,y-5,10,10) for x,y in locations]
    bounds=ax.get_window_extent()
    offsets=[(dx,dy) for dy in [19,-19,37,-37,55,-55,74,-74,94,-94]
             for dx in [0,-18,18,-36,36,-56,56]]
    for i,v in enumerate(points):
        text=v['text'];width=max(renderer.get_text_width_height_descent(t,font,False)[0] for t in text.split('\n'))+5
        height=len(text.split('\n'))*11*fig.dpi/72+3
        x,y=locations[i];options=[]
        for dx,dy in offsets:
            cx,cy=x+dx*fig.dpi/72,y+dy*fig.dpi/72
            box=Bbox.from_bounds(cx-width/2,cy-height/2,width,height)
            score=dx*dx+dy*dy
            for other in occupied+[o for j,o in enumerate(obstacle) if j!=i]:
                overlap=Bbox.intersection(box,other)
                if overlap is not None:score+=1e5+overlap.width*overlap.height*100
            if not (bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1)):score+=1e9
            options.append((score,dx,dy,box))
        _,dx,dy,box=min(options,key=lambda z:z[0]);occupied.append(box)
        ax.annotate(text,(v['x'],v['y']),xytext=(dx,dy),textcoords='offset points',
            ha='center',va='center',fontsize=8,color=v['color'],
            arrowprops=dict(arrowstyle='-',color=v['color'],lw=.45,alpha=.6),
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.2),zorder=6)


ours=data['ours']
def draw_panel(fig,ax,key,label,panel=None):
    legend=[]
    all_points=[];values=[ours[key]]
    for name,color,marker in zip(names,colors,markers):
        rows=[r for r in data['paired_rows'] if r['model']==name]
        if not rows:continue
        x=[r[key] for r in rows];y=[r['srcc'] for r in rows];values+=x
        line,=ax.plot(x,y,marker=marker,lw=1,ms=5,color=color,label=name,zorder=3)
        legend.append(line)
        for r in rows:
            u=selected_util.get((r['model'],r['batch_size']),r['independent_resident_util_mean_percent'])
            all_points.append(dict(x=r[key],y=r['srcc'],text=f"B={r['batch_size']}\n{u:.1f}%",color=color))
    star=ax.scatter([ours[key]],[ours['physical_reference_srcc']],marker='*',s=155,
                    color='#7828A8',label='LightGen-II reference (N=16)',zorder=5)
    legend.insert(0,star)
    ax.set_xscale('log');ax.set_xlim(min(values)*.7,max(values)*1.55);ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0,1.01,.2));ax.set_xlabel(label);ax.set_ylabel('Temporal quality SRCC')
    ax.grid(axis='both',which='major',color='#D8DDE3',lw=.55,zorder=0)
    ax.grid(axis='x',which='minor',color='#E8EBEF',lw=.3,zorder=0)
    if panel is not None:ax.text(-.065,1.025,chr(97+panel),transform=ax.transAxes,fontweight='bold',fontsize=14)
    all_points.append(dict(x=ours[key],y=ours['physical_reference_srcc'],text='N=16',color='#7828A8'))
    fig.canvas.draw();place_labels(ax,all_points,fig)
    if len(all_points)!=31 and not args.preview_partial:raise RuntimeError('Not all 31 points are drawn')
    if not all(ax.get_xlim()[0]<x<ax.get_xlim()[1] for x in values):raise RuntimeError('Clipped data coordinates')
    return legend

fig,axes=plt.subplots(1,2,figsize=(16,7.5))
fig.subplots_adjust(left=.06,right=.99,bottom=.22,top=.93,wspace=.17)
for panel,(ax,key,label) in enumerate(zip(axes,['energy_16_videos_j','time_16_videos_ms'],
        ['Energy per 16 videos (J)','Inference time per 16 videos (ms)'])):
    legend=draw_panel(fig,ax,key,label,panel)
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.52,.085),ncol=4,frameon=False,fontsize=9)
fig.text(.5,.047,'Labels show GPU batch B and independently measured resident GPU utilization; utilization is not used in these cost estimates.',ha='center',fontsize=9)
fig.text(.5,.019,'Qwen B=1 retains finalized historical cost and 63.13% utilization. Other utilization labels are three-repeat pooled means.',ha='center',fontsize=9)
if args.preview_partial:fig.suptitle(f"Partial verification preview: {data['verified_points']}/30 configurations",fontsize=11)
stem='lgvq_costs_partial_preview' if args.preview_partial else 'srcc_cost_comparison_reference'
fig.savefig(out/(stem+'.png'),dpi=300,facecolor='white')
if not args.preview_partial:fig.savefig(out/(stem+'.svg'),facecolor='white')
plt.close(fig)

if not args.preview_partial:
    for key,label,stem in [('time_16_videos_ms','Inference time per 16 videos (ms)','latency_comparison_reference'),
            ('energy_16_videos_j','Energy per 16 videos (J)','energy_comparison_reference')]:
        fig,ax=plt.subplots(figsize=(9,7.3));fig.subplots_adjust(left=.11,right=.97,bottom=.245,top=.96)
        legend=draw_panel(fig,ax,key,label)
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.53,.09),ncol=3,frameon=False,fontsize=9)
        fig.text(.53,.053,'Labels: GPU batch B and independent resident utilization.',ha='center',fontsize=9)
        fig.text(.53,.027,'Qwen B=1 retains finalized cost and 63.13% utilization.',ha='center',fontsize=9)
        fig.text(.53,.003,'LightGen-II: physical SRCC reference and component-cost estimate (N=16).',ha='center',fontsize=8)
        for ext in ['png','svg']:fig.savefig(out/(stem+'.'+ext),dpi=300,facecolor='white')
        plt.close(fig)
    paths=[out/(stem+'.'+ext) for stem in ['srcc_cost_comparison_reference','latency_comparison_reference','energy_comparison_reference'] for ext in ['png','svg']]
    receipt=dict(models=expected_batches,electronic_points_per_panel=30,optical_reference_points_per_panel=1,
        qwen_b1_finalized_util_percent=selected_util[('Qwen3-VL-2B',1)],workload_videos=16,
        axes=dict(x='log scale, complete min/max coverage',y='Temporal quality SRCC, 0 to 1'),
        source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'verified_costs.json',display_path,out/'plot_data.csv']},
        figures=[dict(file=str(p.relative_to(repo)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths])
    (run/'complete_comparison_figures_20261016.json').write_text(json.dumps(receipt,indent=2)+'\n')

if not args.preview_partial and not args.cost_only:
    util=json.loads((run.parent/'lgvq_utilization_execution_20261015/unified_repeat_suite01/verified_summary.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,6.8),sharey=True)
    for ax,name,color,marker in zip(axes.flat,names,colors,markers):
        r=[dict(v,mean_percent=selected_util.get((v['model'],v['batch']),v['mean_percent'])) for v in util['rows'] if v['model']==name]
        ax.plot([v['batch'] for v in r],[v['mean_percent'] for v in r],marker=marker,color=color,lw=1.2,ms=5)
        ax.axhline(98,color='#777777',lw=.6,ls='--')
        ax.set_title(name,fontsize=11);ax.set_ylim(0,102);ax.set_xlim(.65,max(v['batch'] for v in r)+.5)
        ax.set_xticks([v['batch'] for v in r]);ax.set_xlabel('Videos per GPU call, B');ax.grid(axis='y',color='#E0E4EA',lw=.5)
        for v in r:ax.annotate(f"{v['mean_percent']:.1f}",(v['batch'],v['mean_percent']),xytext=(0,-14 if v['mean_percent']>92 else 8),textcoords='offset points',ha='center',fontsize=8,color=color)
        if name=='Qwen3-VL-2B' and display:
            ax.scatter([1],[selected_util[(name,1)]],s=60,facecolors='white',edgecolors=color,zorder=5)
    axes[0,0].set_ylabel('GPU utilization (%)');axes[1,0].set_ylabel('GPU utilization (%)')
    fig.tight_layout(rect=(0,.08,1,1));fig.text(.5,.035,'Independent native resident queue8 load. Qwen B=1: finalized historical single run (open marker).',ha='center',fontsize=9)
    fig.text(.5,.01,'Other points: three identical repeats, pooled mean. Dashed line: 98% utilization; not SM occupancy.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(out/('gpu_utilization_comparison.'+ext),dpi=220,facecolor='white')
    plt.close(fig)
print(json.dumps(dict(preview=args.preview_partial,verified_points=data['verified_points'],figure=str(out/(stem+'.png')))))
