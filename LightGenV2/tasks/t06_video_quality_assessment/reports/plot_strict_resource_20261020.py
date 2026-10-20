"""Eight compact panels from independently audited strict interval data only."""
from __future__ import annotations
import hashlib
import json
import csv
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from . import strict_tf_resource_audit_20261020 as source
from .plot_selected_execution import compact_panel,COMPACT_COLORS,COMPACT_MARKERS

OUT=source.common.OUT

DEPLOYED_PRECISION={
    'qwen':('BF16','BF16 backbone; FP32 quality readout'),
    'glm':('BF16','BF16 backbone; FP32 quality readout'),
    'deepseek':('BF16','BF16 backbone; FP32 quality readout'),
    'kimi':('BF16','BF16 backbone; FP32 quality readout'),
    'clip':('FP16 mixed','FP16/FP32 parameters; FP16 autocast; FP32 quality readout'),
    'yolo':('FP16 mixed','FP32 parameters; FP16 autocast; FP32 quality readout'),
}
DEPLOYED_MODELS=('qwen','glm','clip','yolo','deepseek','kimi')


def selected_mps(data):
    """Select the original deployment, never choose precision by its result."""
    assert data['accounting_protocol']==source.ACCOUNTING and data['status']=='all48_complete'
    values=[]
    for model in DEPLOYED_MODELS:
        pair=[p for p in data['values'] if (p['model'],p['precision'],p['method'])==(model,'native','mps')]
        assert len(pair)==2 and {p['condition'] for p in pair}=={'mps_cap100','mps_cap50'}
        assert len({p['checkpoint_sha256'] for p in pair})==1
        assert len({p['accuracy_source_sha256'] for p in pair})==1
        for point in sorted(pair,key=lambda p:-p['resource_cap']):
            assert point['batch']==1 and point['gpu_count']==1 and len(point['repeats'])==3
            identity_path=(source.REPO/point['source']).parent.parent/'identity.json'
            identity=json.loads(identity_path.read_text())
            dtypes=identity.get('parameter_dtypes',identity.get('floating_parameter_dtypes'))
            expected=({'torch.float16','torch.float32'} if model=='clip' else
                      {'torch.float32'} if model=='yolo' else {'torch.bfloat16'})
            assert set(dtypes)==expected,(model,dtypes)
            assert identity['visible_multiprocessors'] in (108,54,[108],[54])
            n=sum(r['video_evaluations'] for r in point['repeats'])
            t=sum(r['elapsed_s'] for r in point['repeats'])
            e=sum(r['energy_j'] for r in point['repeats'])
            assert all(r['video_evaluations']==r['forward_calls'] for r in point['repeats'])
            assert math.isclose(t*16000/n,point['time16_ms'],rel_tol=1e-10)
            assert math.isclose(e*16/n,point['energy16_j'],rel_tol=1e-10)
            label,note=DEPLOYED_PRECISION[model]
            values.append(dict(point,deployment_precision=label,precision_detail=note,
                               backbone_parameter_dtypes=dtypes,quality_readout_precision='FP32',
                               identity_source=identity_path.relative_to(source.REPO).as_posix(),
                               identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest()))
    return dict(accounting_protocol=data['accounting_protocol'],selection='native MPS only; fixed B1',
                optical=data['optical'],values=values,
                accuracy_note='Separate same-batch full558 development reference, not a fresh evaluation per MPS cap.',
                historical_Qwen_B1_unchanged=True)


def deployed_mps_figures():
    """Current two-panel selection; preserve all eight historical precision panels."""
    original=source.ROOT/'presentation/eight_tables.json'
    data=selected_mps(json.loads(original.read_text()))
    data['source_sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
    target=source.ROOT/'presentation/mps_deployed.json'
    target.write_text(json.dumps(data,indent=2)+'\n')
    columns=['model_name','resource_cap','batch','deployment_precision','precision_detail',
             'quality_readout_precision','srcc_reference','time16_ms','energy16_j','busy_percent',
             'platform','checkpoint_sha256','source','source_sha256','accuracy_source',
             'accuracy_source_sha256','identity_source','identity_sha256']
    with (OUT/'mps_deployed_plot_data.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(data['values'])
    plt.rcParams.update({'font.family':'Arial','font.size':7.2,'svg.fonttype':'none',
                         'pdf.fonttype':42,'axes.linewidth':.6,'savefig.facecolor':'white'})
    panels=[]
    for metric,key in [('latency','time16_ms'),('energy','energy16_j')]:
        fig,ax=plt.subplots(figsize=(78/25.4,88/25.4))
        fig.subplots_adjust(left=.19,right=.97,top=.96,bottom=.40)
        optical=data['optical'];opkey='time_ms' if metric=='latency' else 'energy_j'
        xs=[optical[opkey]]+[p[key] for p in data['values']]
        ys=[optical['srcc']]+[p['srcc_reference'] for p in data['values']]
        ax.set_xscale('log');ax.set_xlim(min(xs)*.68,max(xs)*1.4);ax.set_ylim(.681,.843)
        ax.set_yticks([.70,.75,.80]);ax.set_ylabel('Temporal quality (SRCC)',labelpad=3)
        ax.set_xlabel('Equivalent processing time (ms)' if metric=='latency' else 'Estimated energy (J)',labelpad=3)
        ax.grid(which='major',color='#DBDFE3',lw=.45,zorder=0)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out',length=2.5,width=.6,pad=2,labelsize=7)
        ax.scatter(optical[opkey],optical['srcc'],marker='*',s=49,color=COMPACT_COLORS['ours'],zorder=4)
        handles=[Line2D([],[],marker='*',color=COMPACT_COLORS['ours'],lw=0,markersize=5,label='LightGen-II')]
        for model in DEPLOYED_MODELS:
            pair=[p for p in data['values'] if p['model']==model]
            style='full' if model=='deepseek' else model
            color=COMPACT_COLORS[style];marker=COMPACT_MARKERS[style]
            ax.plot([p[key] for p in pair],[p['srcc_reference'] for p in pair],color=color,lw=.9,alpha=.75,zorder=2)
            for point in pair:
                ax.scatter(point[key],point['srcc_reference'],marker=marker,s=21,
                           facecolors=color if point['resource_cap']==100 else 'white',
                           edgecolors=color,linewidths=.85,zorder=4)
            handles.append(Line2D([],[],marker=marker,color=color,lw=.8,markersize=4,label=pair[0]['model_name']))
        model_legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.075),
                               ncol=2,frameon=False,fontsize=6.1,handlelength=1.4,columnspacing=.8)
        resource_handles=[Line2D([],[],marker='o',color='#40474D',lw=0,markersize=4,label='100% MPS allocation'),
                          Line2D([],[],marker='o',markerfacecolor='white',color='#40474D',lw=0,markersize=4,label='50% MPS allocation')]
        allocation_legend=fig.legend(handles=resource_handles,loc='lower center',bbox_to_anchor=(.53,.005),
                                    ncol=1,frameon=False,fontsize=5.6,handlelength=1.4)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        assert not model_legend.get_window_extent(renderer).overlaps(ax.xaxis.label.get_window_extent(renderer))
        assert not model_legend.get_window_extent(renderer).overlaps(allocation_legend.get_window_extent(renderer))
        assert all(ax.get_xlim()[0]<x<ax.get_xlim()[1] for x in xs)
        assert all(ax.get_ylim()[0]<y<ax.get_ylim()[1] for y in ys)
        assert not ax.texts  # No utilization, spatial-use or batch point labels.
        for ext in ('png','svg','pdf'):fig.savefig(OUT/f'mps_deployed_comparison_{metric}.{ext}',dpi=600)
        panels.append(dict(metric=metric,points=len(xs),connectors=6,point_annotations=0,
                           xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())))
        plt.close(fig)
    a=Image.open(OUT/'mps_deployed_comparison_latency.png');b=Image.open(OUT/'mps_deployed_comparison_energy.png')
    pair=Image.new('RGB',(a.width+b.width,a.height),'white');pair.paste(a,(0,0));pair.paste(b,(a.width,0))
    pair.save(OUT/'mps_deployed_comparison_pair.png',dpi=(600,600))
    (source.ROOT/'presentation/mps_deployed_figure_receipt.json').write_text(json.dumps(dict(
        panels=panels,width_mm=78,height_mm=88,utilization_labels=False,spatial_labels=False,
        selected_source_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(selected_conditions=12,trial_windows=36,panels=2,batch=1)))


def normalized(data,method,precision):
    optical=dict(data['optical'],baseline='ours',plotted=True,condition='Optical reference',
                 marker=COMPACT_MARKERS['ours'],color=COMPACT_COLORS['ours'],link_style='solid')
    values=[optical]
    for point in data['values']:
        if (point['method'],point['precision'])!=(method,precision):continue
        if point.get('srcc_reference') is None:continue
        model='full' if point['model']=='deepseek' else point['model']
        low=point['condition'] in ('busy50','mps_cap50')
        values.append(dict(baseline=model,model=point['model_name'],plotted=True,
            condition='Approximately 50%' if low else 'Unrestricted reference',
            time_ms=point['time16_ms'],energy_j=point['energy16_j'],srcc=point['srcc_reference'],
            utilization_percent=point['busy_percent'],marker=COMPACT_MARKERS[model],
            color=COMPACT_COLORS[model],link_style='solid'))
    return values


def main():
    path=source.ROOT/'presentation/eight_tables.json';data=json.loads(path.read_text())
    if data['accounting_protocol']!=source.ACCOUNTING:raise ValueError('Latest interval replay required')
    plt.rcParams.update({'font.family':'Arial','font.size':7.2,'svg.fonttype':'none',
                         'pdf.fonttype':42,'axes.linewidth':.6,'savefig.facecolor':'white'})
    results=[]
    for method in ('temporal','mps'):
        for precision in ('native','fp32'):
            values=normalized(data,method,precision)
            modelnames={r['baseline']:r.get('model','LightGen-II') for r in values}
            handles=[Line2D([],[],marker=COMPACT_MARKERS[m],color=COMPACT_COLORS[m],lw=.8,
                            markersize=4,label=modelnames[m]) for m in COMPACT_COLORS if m in modelnames]
            prefix=f'strict_tf_{method}_{precision}'
            for metric in ('latency','energy'):
                fig,ax=plt.subplots(figsize=(78/25.4,88/25.4))
                fig.subplots_adjust(left=.19,right=.97,top=.92,bottom=.35)
                geometry=compact_panel(fig,ax,values,'strict_interval',metric)
                ax.set_xlabel('Estimated energy (J)' if metric=='energy' else 'Processing time (ms)',labelpad=3)
                ax.set_title(('Temporal duty control' if method=='temporal' else 'MPS resource provisioning')+
                             (' · FP32' if precision=='fp32' else ' · deployed precision'),fontsize=7.1)
                legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.075),
                                  ncol=2,frameon=False,fontsize=6.1,handlelength=1.4,columnspacing=.8)
                note=('First block→output + output→next input.\n'
                      'Frontend excluded; 16-video equivalents.\n'+
                      ('Open: time-limited; filled: unrestricted.' if method=='temporal' else
                       'Open: 50% resource cap; filled: 100% cap.')+
                      '\nLabels: full-wall busy (dual GPU: max card).')
                footer=fig.text(.19,.005,note,fontsize=4.8,color='#53616B')
                fig.canvas.draw();renderer=fig.canvas.get_renderer()
                assert not legend.get_window_extent(renderer).overlaps(ax.xaxis.label.get_window_extent(renderer))
                assert not legend.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer))
                for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{prefix}_{metric}.{ext}',dpi=600)
                results.append(dict(method=method,precision=precision,metric=metric,points=len(values),geometry=geometry))
                plt.close(fig)
            a=Image.open(OUT/f'{prefix}_latency.png');b=Image.open(OUT/f'{prefix}_energy.png')
            pair=Image.new('RGB',(a.width+b.width,a.height),'white');pair.paste(a,(0,0));pair.paste(b,(a.width,0))
            pair.save(OUT/f'{prefix}_pair.png',dpi=(600,600))
    receipt=dict(status='passed',panels=results,width_mm=78,height_mm=88,
                 source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 omitted_missing_accuracy=True,utilization_only_labels=True,
                 same_model_same_marker=True,all_available_endpoints_connected=True)
    (source.ROOT/'presentation/figure_geometry.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(panels=len(results),width_mm=78,geometry='passed')))


if __name__=='__main__':
    if '--mps-deployed' in sys.argv:deployed_mps_figures()
    else:main()
