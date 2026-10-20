"""Eight compact panels from independently audited strict interval data only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from . import strict_tf_resource_audit_20261020 as source
from .plot_selected_execution import compact_panel,COMPACT_COLORS,COMPACT_MARKERS

OUT=source.common.OUT


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


if __name__=='__main__':main()
