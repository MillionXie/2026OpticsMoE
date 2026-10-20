"""Render only the independently replayed cold, distinct-video MPS measurements."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO=Path(__file__).resolve().parents[4]
ROOT=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_dataset_once_a100_20261020/presentation'
OUT=REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
NAMES=dict(qwen='Qwen3-VL-2B',glm='GLM-4.6V-Flash',clip='CLIP ViT-B/32',yolo='YOLO11s')
COLORS=dict(qwen='#0072B2',glm='#92527F',clip='#007F63',yolo='#9B7000')
MARKERS=dict(qwen='o',glm='P',clip='D',yolo='^')
OPTICAL=dict(time16_ms=10.245569591370225,energy16_j=1.1776895097085185,srcc=.7977138739)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(ax,points,metric):
    xs=[OPTICAL[metric]]+[p[metric] for p in points]
    ys=[OPTICAL['srcc']]+[p['srcc'] for p in points]
    ax.scatter(OPTICAL[metric],OPTICAL['srcc'],s=52,marker='*',color='#7828A8',zorder=4)
    for model in NAMES:
        rr=sorted([p for p in points if p['model']==model],key=lambda r:-r['resource_cap'])
        assert len(rr)==2
        ax.plot([r[metric] for r in rr],[r['srcc'] for r in rr],color=COLORS[model],lw=.9,zorder=2)
        # Draw the larger open mark first so coincident conditions remain visible.
        # Coordinates and model-specific marker geometry are never displaced.
        for r in reversed(rr):
            ax.scatter(r[metric],r['srcc'],s=18 if r['resource_cap']==100 else 32,marker=MARKERS[model],
                       facecolor=COLORS[model] if r['resource_cap']==100 else 'white',
                       edgecolor=COLORS[model],linewidth=.9,zorder=3)
    ax.set_xscale('log');ax.set_xlim(min(xs)*.7,max(xs)*1.55)
    ymin=min(.65,min(ys)-.025);ymax=max(.85,max(ys)+.025)
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel('Temporal quality (SRCC)')
    ax.set_xlabel('Equivalent processing time (ms)' if metric=='time16_ms' else 'Estimated equivalent energy (J)')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(which='major',color='#D8DDE2',lw=.45);ax.grid(which='minor',axis='x',color='#ECEFF1',lw=.3)
    ax.tick_params(direction='out',length=2.5)
    assert all(ax.get_xlim()[0]<x<ax.get_xlim()[1] for x in xs)
    assert all(ymin<y<ymax for y in ys)
    return dict(plotted_points=len(xs),models=4,connected_pairs=4,x_limits=list(ax.get_xlim()),y_limits=list(ax.get_ylim()))


def legends(fig):
    models=[Line2D([],[],marker='*',ls='none',color='#7828A8',label='LightGen-II',markersize=7)]
    models.extend(Line2D([],[],marker=MARKERS[m],ls='-',color=COLORS[m],label=NAMES[m],markersize=4,lw=.8)for m in NAMES)
    fig.legend(handles=models,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,columnspacing=1.1,handlelength=1.2)
    caps=[Line2D([],[],marker='o',ls='none',color='#58646B',markerfacecolor=fc,label=label,markersize=size)
          for fc,label,size in [('#58646B','100% MPS allocation',4.2),('white','50% MPS allocation',5.6)]]
    fig.legend(handles=caps,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=1,frameon=False,handlelength=1.0,labelspacing=.2)


def main():
    source=ROOT/'dataset_once_audited.json';data=json.loads(source.read_text())
    assert data['status']=='complete' and len(data['points'])==8 and len(data['repeats'])==24
    optical_source=ROOT.parents[1]/'lgvq_strict_tf_rental_dual_20261020/presentation/eight_tables.json'
    prior=json.loads(optical_source.read_text())['optical']
    assert (prior['time_ms'],prior['energy_j'],prior['srcc'])==(OPTICAL['time16_ms'],OPTICAL['energy16_j'],OPTICAL['srcc'])
    points=data['points'];plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7.5,'svg.fonttype':'none','pdf.fonttype':42})
    receipt=dict(source_sha256=digest(source),marker_rendering='Larger open50 mark behind smaller filled100 mark, exact coordinates and same model geometry',optical_reference_source=optical_source.relative_to(REPO).as_posix(),
                 optical_reference_sha256=digest(optical_source),protocol='dataset_once_legacy_block_plus_nonpreprocessing_gaps_v1',files={},panels=[])
    for metric,kind in [('time16_ms','latency'),('energy16_j','energy')]:
        fig,ax=plt.subplots(figsize=(78/25.4,88/25.4));fig.subplots_adjust(left=.20,right=.97,bottom=.32,top=.84)
        receipt['panels'].append(panel(ax,points,metric));legends(fig)
        for ext in ('png','svg','pdf'):
            target=OUT/f'mps_dataset_once_comparison_{kind}.{ext}';fig.savefig(target,dpi=600);receipt['files'][target.name]=digest(target)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(156/25.4,88/25.4));fig.subplots_adjust(left=.09,right=.985,bottom=.28,top=.84,wspace=.30)
    for ax,metric,label in zip(axes,['time16_ms','energy16_j'],['a','b']):
        panel(ax,points,metric);ax.text(-.18,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
    legends(fig);target=OUT/'mps_dataset_once_comparison_pair.png';fig.savefig(target,dpi=600);receipt['files'][target.name]=digest(target);plt.close(fig)
    target=OUT/'mps_dataset_once_plot_data.csv'
    base_fields=['model','resource_cap','batch','time16_ms','energy16_j','srcc']
    fields=['model','model_name',*base_fields[1:],'srcc_repeats','time_ratio_to_optical','energy_ratio_to_optical','unique_cost_videos_per_repeat','full_test_videos_per_repeat','repeats','evidence_type','source_json','source_sha256']
    with target.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerow(dict(model='ours',model_name='LightGen-II',**OPTICAL,time_ratio_to_optical=1,energy_ratio_to_optical=1,
                        evidence_type='Prior optical component-cost reference',source_json=optical_source.relative_to(REPO).as_posix(),source_sha256=digest(optical_source)))
        for p in points:
            w.writerow({**{k:p[k] for k in base_fields},'model_name':NAMES[p['model']],'srcc_repeats':json.dumps(p['srcc_repeats']),
                        'time_ratio_to_optical':p['time16_ms']/OPTICAL['time16_ms'],
                        'energy_ratio_to_optical':p['energy16_j']/OPTICAL['energy16_j'],
                        'unique_cost_videos_per_repeat':200,'full_test_videos_per_repeat':558,'repeats':3,'evidence_type':'New cold distinct-video measurement',
                        'source_json':source.relative_to(REPO).as_posix(),'source_sha256':digest(source)})
    receipt['files'][target.name]=digest(target)
    (ROOT/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
