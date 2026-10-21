"""Compact, separately labelled figures for the existing FP32 duty-control data."""
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from . import temporal_fp32_reference as source
from .present_mps_dataset_once import NAMES,COLORS,MARKERS

ROOT=source.ROOT
OUT=source.audit.common.OUT
PREFIX='fp32_temporal_reference'


def panel(ax,data,key):
    optical=data['optical'];opkey='time_ms' if key=='time16_ms' else 'energy_j'
    xs=[optical[opkey]]+[p[key] for p in data['values']]
    ax.scatter(optical[opkey],optical['srcc'],marker='*',s=52,color='#7828A8',zorder=4)
    for model in source.MODELS:
        pair=sorted([p for p in data['values'] if p['model']==model],key=lambda p:p['condition']!='unrestricted')
        assert len(pair)==2 and pair[0]['display_reference_srcc']==pair[1]['display_reference_srcc']
        ax.plot([p[key] for p in pair],[p['display_reference_srcc'] for p in pair],lw=.9,color=COLORS[model],zorder=2)
        for p in reversed(pair):
            full=p['condition']=='unrestricted'
            ax.scatter(p[key],p['display_reference_srcc'],s=18 if full else 32,marker=MARKERS[model],
                       facecolor=COLORS[model] if full else 'white',edgecolor=COLORS[model],lw=.9,zorder=3)
    ax.set_xscale('log');ax.set_xlim(min(xs)*.7,max(xs)*1.55);ax.set_ylim(.65,.85)
    ax.set_ylabel('Reference temporal quality (SRCC)')
    ax.set_xlabel('Equivalent processing time (ms)' if key=='time16_ms' else 'Estimated equivalent energy (J)')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(which='major',color='#D8DDE2',lw=.45);ax.grid(which='minor',axis='x',color='#ECEFF1',lw=.3)
    ax.tick_params(direction='out',length=2.5)
    assert all(ax.get_xlim()[0]<x<ax.get_xlim()[1] for x in xs)
    assert all(.65<p['display_reference_srcc']<.85 for p in data['values'])
    return dict(points=13,connected_pairs=6,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()))


def legends(fig,paired):
    handles=[Line2D([],[],marker='*',ls='none',color='#7828A8',label='LightGen-II',markersize=7)]
    handles.extend(Line2D([],[],marker=MARKERS[m],color=COLORS[m],lw=.8,label=NAMES[m],markersize=4) for m in source.MODELS)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3 if paired else 2,
               frameon=False,columnspacing=.9,handlelength=1.2)
    caps=[Line2D([],[],marker='o',ls='none',color='#58646B',markerfacecolor=fc,label=text,markersize=4.2)
          for fc,text in [('#58646B','Unrestricted execution'),('white','Time-paced execution')]]
    fig.legend(handles=caps,loc='upper center',bbox_to_anchor=(.5,1),frameon=False,labelspacing=.2)


def main():
    data=source.audit.read(source.PAPER);proof=source.audit.read(ROOT/'temporal_fp32_reference_audit.json')
    assert proof['status']=='passed' and proof['reference_sha256']==source.audit.sha(source.TARGET)
    assert data['fp32_selection_sha256']==source.audit.sha(source.TARGET)
    assert data['display_accuracy_source_sha256']==source.audit.sha(source.NATIVE)
    receiptpath=ROOT/'temporal_fp32_reference_figure_receipt.json'
    expected={f'{PREFIX}_{kind}.{ext}' for kind in ('latency','energy') for ext in ('png','svg','pdf')}
    expected|={PREFIX+'_pair.png',PREFIX+'_plot_data.csv'}
    if receiptpath.exists():
        previous=source.audit.read(receiptpath)
        for name,digest in previous['files'].items():
            p=OUT/name
            assert source.audit.sha(p)==digest,'Preserve externally edited '+name
            saved=ROOT/(p.stem+'_prior_'+digest[:12]+p.suffix)
            if not saved.exists():shutil.copy2(p,saved)
    else:assert not any((OUT/name).exists() for name in expected),'Do not overwrite an unidentified export'
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7.5,'svg.fonttype':'none','pdf.fonttype':42})
    files={};geometry=[]
    for kind,key in [('latency','time16_ms'),('energy','energy16_j')]:
        fig,ax=plt.subplots(figsize=(78/25.4,88/25.4));fig.subplots_adjust(left=.20,right=.97,bottom=.32,top=.84)
        geometry.append(panel(ax,data,key));legends(fig,False)
        for ext in ('png','svg','pdf'):
            target=OUT/f'{PREFIX}_{kind}.{ext}';fig.savefig(target,dpi=600);files[target.name]=source.audit.sha(target)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(156/25.4,88/25.4));fig.subplots_adjust(left=.09,right=.985,bottom=.28,top=.84,wspace=.30)
    for ax,key,label in zip(axes,['time16_ms','energy16_j'],['a','b']):
        panel(ax,data,key);ax.text(-.18,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
    legends(fig,True);target=OUT/(PREFIX+'_pair.png');fig.savefig(target,dpi=600);files[target.name]=source.audit.sha(target);plt.close(fig)
    target=OUT/(PREFIX+'_plot_data.csv')
    fields=['model','condition','display_reference_srcc','fp32_reference_srcc','time16_ms','energy16_j','time_ratio_to_optical','energy_ratio_to_optical','batch','gpu_count','actual_busy_percent','parameters_billion','platform','cost_source','cost_sha256','accuracy_source','accuracy_sha256','precision','input_protocol','cost_repeat_count','display_accuracy_policy','display_accuracy_source','display_accuracy_source_sha256']
    with target.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        optical=data['optical'];writer.writerow(dict(model='LightGen-II',condition='Prior optical component reference',display_reference_srcc=optical['srcc'],time16_ms=optical['time_ms'],energy16_j=optical['energy_j'],time_ratio_to_optical=1,energy_ratio_to_optical=1,precision='Not applicable'))
        for p in data['values']:
            writer.writerow(dict(model=p['model_name'],condition=p['condition'],display_reference_srcc=p['display_reference_srcc'],fp32_reference_srcc=p['srcc_reference'],time16_ms=p['time16_ms'],energy16_j=p['energy16_j'],time_ratio_to_optical=p['time16_ms']/optical['time_ms'],energy_ratio_to_optical=p['energy16_j']/optical['energy_j'],batch=p['batch'],gpu_count=p['gpu_count'],actual_busy_percent=p['busy_percent'],parameters_billion=p['model_parameters']/1e9,platform=p['platform'],cost_source=p['source'],cost_sha256=p['source_sha256'],accuracy_source=p['accuracy_source'],accuracy_sha256=p['accuracy_source_sha256'],precision='FP32',input_protocol=data['input_protocol'],cost_repeat_count=3,display_accuracy_policy=data['display_accuracy_policy'],display_accuracy_source=data['display_accuracy_source'],display_accuracy_source_sha256=data['display_accuracy_source_sha256']))
    files[target.name]=source.audit.sha(target)
    receipt=dict(source_sha256=source.audit.sha(source.PAPER),fp32_selection_sha256=source.audit.sha(source.TARGET),display_accuracy_source_sha256=data['display_accuracy_source_sha256'],display_accuracy_policy=data['display_accuracy_policy'],files=files,panels=geometry,current_mps_unchanged=True)
    receiptpath.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(files=len(files),panels=geometry)))


if __name__=='__main__':main()
