"""Replot verified sparse figure points, never reconstruct model measurements."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO=Path(__file__).resolve().parents[4]
RUN=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_selected_execution_a100_20261016'
OUT=REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
COLORS=dict(ours='#7828A8',qwen='#0072B2',glm='#92527F',small='#C25B18',clip='#007F63',yolo='#9B7000',deepseek='#5B6470')
MARKERS=dict(ours='*',qwen='o',glm='P',small='s',clip='D',yolo='^',deepseek='v')


def panel(ax,rows,kind):
    key='energy_16_videos_j' if kind=='energy' else 'time_16_videos_ms'
    offsets={1:(12,17),2:(-13,-35),3:(12,19),4:(-15,-36),6:(-12,-36),8:(12,19),9:(-17,-36),10:(15,20)}
    for model,color in COLORS.items():
        selected=[x for x in rows if x['baseline']==model]
        if not selected:continue
        connected=[x for x in selected if not(model=='qwen' and x['batch_size_videos']==1)]
        if len(connected)>1:ax.plot([x[key] for x in connected],[x['measured_full558_srcc'] for x in connected],color=color,lw=1,alpha=.55,zorder=2)
        for x in selected:
            historic=model=='qwen' and x['batch_size_videos']==1
            ax.scatter(x[key],x['measured_full558_srcc'],s=125 if model=='ours' else 39,marker=MARKERS[model],facecolors='white' if historic else color,edgecolors=color,linewidths=1.1,zorder=4)
            if model=='ours':text='N=16';xytext=(12,-6);align='left'
            else:
                b=x['batch_size_videos'];text=f'B={b}'+('†' if historic else '')+f"\n{x['display_gpu_util_mean_percent']:.1f}%"
                xytext=offsets[b];align='left' if xytext[0]>0 else 'right'
                if model=='small' and b==1:xytext=(-12,18);align='right'
                if model=='glm' and b==1:xytext=(12,18);align='left'
            ax.annotate(text,(x[key],x['measured_full558_srcc']),xytext=xytext,textcoords='offset points',ha=align,va='center',fontsize=8.5,color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.5,alpha=.6),zorder=5)
    ax.set_xscale('log');ax.set_xlim(.8 if kind=='energy' else 8,10000)
    ax.set_ylim(.5,.85);ax.set_yticks(np.arange(.5,.851,.05));ax.set_ylabel('Temporal quality (SRCC)')
    ax.set_xlabel('Estimated energy per 16 videos (J)' if kind=='energy' else 'Latency per 16 videos (ms)')
    ax.grid(which='major',color='#D8DDE2',linewidth=.6);ax.grid(which='minor',axis='x',color='#EBEEF0',linewidth=.4)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.tick_params(labelsize=9,direction='out')


def main():
    state=json.loads((RUN/'presentation_selected.json').read_text());assert state['status']=='complete'
    rows=state['rows'];assert 13<=len(rows)<=18
    assert [(x['batch_size_videos'],x['display_gpu_util_mean_percent']) for x in rows if x['baseline']=='qwen' and x['batch_size_videos']==1]==[(1,63.133111480865225)]
    plt.rcParams.update({'font.family':'Arial','font.size':10,'svg.fonttype':'none','axes.linewidth':.8,'savefig.facecolor':'white'})
    legend=[Line2D([],[],marker=MARKERS[m],color=COLORS[m],lw=1,markersize=8 if m=='ours' else 5,label=next(x['model'] for x in rows if x['baseline']==m)+(' (optimized B≥3)' if m=='qwen' else '')) for m in COLORS]
    for kind,name in [('latency','latency_comparison_reference'),('energy','energy_comparison_reference')]:
        fig,ax=plt.subplots(figsize=(10.0,6.0));panel(ax,rows,kind)
        fig.subplots_adjust(left=.10,right=.95,top=.96,bottom=.24)
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=9)
        fig.text(.10,.035,'Labels: batch size and independent resident GPU utilization. Qwen B≥3 uses equivalence-validated execution optimization.',fontsize=8,color='#4B5563')
        fig.text(.10,.012,'† Qwen B=1 and LightGen-II are retained references. GPU costs normalize the first200 TEST videos to16 videos.',fontsize=8,color='#4B5563')
        for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=240)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15.0,6.0),sharey=True)
    for ax,kind,label in zip(axes,['energy','latency'],['a','b']):
        panel(ax,rows,kind);ax.text(-.13,1.02,label,transform=ax.transAxes,fontweight='bold',fontsize=14)
    axes[1].set_ylabel('');fig.subplots_adjust(left=.065,right=.98,top=.95,bottom=.24,wspace=.16)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=9)
    fig.text(.065,.025,'GPU SRCC: 558 TEST videos. Costs: first200 normalized to16. Independent resident utilization. LightGen-II and † Qwen B=1: retained references.',fontsize=8,color='#4B5563')
    for name in ['srcc_cost_comparison_reference','batch_cost_comparison']:
        for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=240)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.3))
    for m in COLORS:
        if m=='ours':continue
        points=[x for x in rows if x['baseline']==m]
        if m!='qwen':ax.plot([x['batch_size_videos'] for x in points],[x['display_gpu_util_mean_percent'] for x in points],lw=1,alpha=.6,color=COLORS[m])
        for x in points:
            b=x['batch_size_videos'];u=x['display_gpu_util_mean_percent'];historic=m=='qwen' and b==1
            ax.scatter(b,u,color=COLORS[m],marker=MARKERS[m],s=35)
            offset=(5,-13) if m in ['glm','small'] else (5,7)
            if m=='qwen' and b==3:offset=(5,19)
            ax.annotate(f'{u:.1f}%'+('†' if historic else ''),(b,u),xytext=offset,textcoords='offset points',color=COLORS[m],fontsize=8)
    ax.axhline(98,color='#6B7280',ls='--',lw=.7);ax.set_xlim(.7,10.6);ax.set_ylim(20,105);ax.set_xticks(range(1,11))
    ax.set_xlabel('Video batch size');ax.set_ylabel('Resident GPU utilization (%)');ax.grid(color='#E5E7EB',lw=.5)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.subplots_adjust(left=.09,right=.97,bottom=.25,top=.95)
    fig.legend(handles=legend[1:],loc='lower center',bbox_to_anchor=(.5,.04),ncol=3,frameon=False,fontsize=9)
    for ext in ['png','svg']:fig.savefig(OUT/('gpu_utilization_comparison.'+ext),dpi=240)
    plt.close(fig)
    print(json.dumps(dict(figures_updated=5,selected_points=len(rows),historical_Qwen_B1_preserved=True)))


if __name__=='__main__':main()
