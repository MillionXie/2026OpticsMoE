"""Offline rederivation; never imports the historical side-effecting builder.

No weights, CUDA, optical SDK or video decoding are used. Raw files stay intact.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT / 'recalculated_20261013'
RAW = ROOT / 'raw_remote'
SOURCE_FILES = set()
HOST_W = 338.2
COLORS = {'Qwen3-VL-2B':'#0072B2', 'DeepSeek-VL2-Tiny':'#D55E00',
          'CLIP ViT-B/32':'#009E73', 'YOLO11s':'#E69F00'}
REFERENCES = {'Qwen3-VL-2B':0.7663428036522781, 'DeepSeek-VL2-Tiny':0.5688169,
              'CLIP ViT-B/32':0.7251568, 'YOLO11s':0.7015822}

def read_json(path):
    SOURCE_FILES.add(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def read_csv(path):
    SOURCE_FILES.add(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, data):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def srcc(rows):
    return float(spearmanr([float(r['target_mos']) for r in rows],
                          [float(r['prediction']) for r in rows]).statistic)

def main():
    OUT.mkdir(exist_ok=True)
    latest_path = REPO/'LightGenV2/reports/20260928_nine_task_a100_table/raw_remote/ours_narrow_200/report.json'
    latest = read_json(latest_path)
    electronic = next(x for x in latest['formal_narrow_cuda_event'] if x['task']=='lgvq_temporal')
    t_opt = 6 * (0.714 + 0.300 + 0.0307)
    t_el = electronic['main_electronic_including_bridge_cuda_ms']
    ours_t = t_opt + t_el
    ours_e = (t_opt*(80.358+41.388)+t_el*(41.388+62.842))/1000
    assert abs(ours_t-10.245569591370225)<1e-10
    assert abs(ours_e-1.1776895097085185)<1e-10
    timing = []
    metrics = []
    qwen_subsets = {}
    qwen_fullsets = {}
    rerun_audit = []
    evidence = OUT/'server_prediction_evidence'
    manifest = read_json(evidence/'source_manifest.json')
    for entry in manifest['files']:
        relative = entry['local'].replace('\\','/').split('/server_prediction_evidence/',1)[1]
        recovered = evidence/relative
        assert recovered.stat().st_size == entry['bytes']
        assert hashlib.sha256(recovered.read_bytes()).hexdigest() == entry['sha256']
        SOURCE_FILES.add(recovered)
    SOURCE_FILES.update({Path(__file__), REPO/'LightGenV2/reports/20260928_nine_task_a100_table/README.md',
        REPO/'LightGenV2/tasks/t06_video_quality_assessment/CURRENT_VERSION_20261004.md', ROOT/'build_analysis.py'})
    for model in COLORS:
        for batch in (1,2,4,8):
            if model=='Qwen3-VL-2B':
                path=RAW/'frozen_batch_20261013'/f'qwen_batch{batch:02d}'/'report.json'
                report=read_json(path)
                assert report['status']=='complete' and report['test_videos']==558
                assert report['gpu_uuid']=='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
                assert report['checkpoint_sha256']=='e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2'
                wall=float(report['inference_batch_wall_ms']['mean'])
                wall_p95=float(report['inference_batch_wall_ms']['p95'])
                power=float(report['power']['active_mean_w'])
                n=int(report['cost_test_videos'])
                timed=read_csv(path.parent/'batch_timing.csv')
                cost_rows=[r for r in timed if r['in_cost_subset']=='True']
                assert sum(int(r['videos']) for r in cost_rows)==200
                calls=len(cost_rows)
                assert abs(report['cost']['latency_16_videos_ms']-wall*16/batch)<1e-8
                assert abs(report['cost']['energy_16_videos_j']-(HOST_W+power)*wall*16/batch/1000)<1e-8
                pred_path=path.parent/'predictions.csv'
                full=read_csv(pred_path)
                assert len(full)==558 and len({r['sample_id'] for r in full})==558
                preds=full[:200]
                assert len(preds)==200 and len({r['sample_id'] for r in preds})==200
                qwen_subsets[batch]=preds
                subset_srcc=srcc(preds)
                qwen_fullsets[batch]=full
                full_srcc=srcc(full)
                assert abs(full_srcc-report['performance']['srcc'])<1e-12
                full_source=str(pred_path.relative_to(REPO))
                SOURCE_FILES.update(path.parent.glob('*.csv'))
                cost_basis='2026-10-13 interleaved frozen reevaluation; first200 inference only'
                if batch==1:
                    rerun_audit.append(dict(model=model,batch_size=1,source=str(path.relative_to(REPO)),
                        cost=report['cost'],full558_srcc=full_srcc,
                        status='audit only; not substituted into final B1 table anchor'))
                    path=REPO/'LightGenV2/reports/20260927_demo_energy_efficiency_a100/raw_remote/baseline/01_lgvq_temporal/resolution448/frames4/scheme2_five_quality_tokens/report.json'
                    anchor=read_json(path)
                    wall=float(anchor['model_internal_synchronized_wall_ms_first_200_included']['mean'])
                    wall_p95=float(anchor['model_internal_synchronized_wall_ms_first_200_included']['p95'])
                    power=float(anchor['power']['active_mean_w'])
                    cost_basis='Historical nine-task table B1 anchor; fresh B1 reevaluation retained separately'
                dtype='bfloat16'
            else:
                folder={'CLIP ViT-B/32':'clip_vit_b32','DeepSeek-VL2-Tiny':'deepseek_vl2_tiny_independent','YOLO11s':'yolo11s_independent'}[model]
                if model=='CLIP ViT-B/32' and batch!=1:
                    folder='clip_vit_b32_independent'
                path=RAW/folder/'report.json' if model=='CLIP ViT-B/32' and batch==1 else RAW/folder/f'batch_{batch}'/'report.json'
                report=read_json(path)
                item=report['results'][0]
                assert int(item['batch_size_videos'])==batch
                wall=float(item['synchronized_wall_ms_per_batch']['mean'])
                wall_p95=float(item['synchronized_wall_ms_per_batch']['p95'])
                power=float(item['active_a100_board_power_w']['mean'])
                n=int(item['measured_videos'])
                calls=int(item['measured_calls'])
                subset_srcc=full_srcc=None
                full_source='timing-only profiler; trained-checkpoint batch predictions absent'
                cost_basis='Historical 2026-09-27 timing; separate sustained power pass; batch-specific SRCC unavailable'
                dtype='bfloat16' if model=='DeepSeek-VL2-Tiny' else 'float16'
            assert n==200 and calls==200//batch
            time16=wall*(16/batch)
            energy16=(HOST_W+power)*time16/1000
            timing.append(dict(model=model,batch_size=batch,measured_videos=n,measured_calls=calls,
                inference_precision=dtype,wall_ms_per_batch=wall,wall_p95_ms_per_batch=wall_p95,a100_active_mean_w=power,host_w=HOST_W,
                calls_for_16=16//batch,time_16_videos_ms=time16,energy_16_videos_j=energy16,
                amortized_time_ms_per_video=time16/16,amortized_energy_j_per_video=energy16/16,
                videos_per_j=16/energy16,speedup_vs_ours=time16/ours_t,energy_gain_vs_ours=energy16/ours_e,
                energy_reduction_vs_ours=1-ours_e/energy16,
                timing_source=str(path.relative_to(REPO)),cost_basis=cost_basis))
            metrics.append(dict(model=model,batch_size=batch,full_test_reference_srcc=REFERENCES[model],
                measured_full558_batch_srcc=full_srcc,measured_first200_batch_srcc=subset_srcc,
                full_batch_metric_source=full_source,
                status='2026-10-13 frozen full558 predictions; B1 cost uses preserved table anchor' if model=='Qwen3-VL-2B' else 'batch-specific metric unavailable'))
    first_ids=[r['sample_id'] for r in qwen_subsets[1]]
    for b, rr in qwen_subsets.items():
        assert [r['sample_id'] for r in rr]==first_ids
        assert [r['target_mos'] for r in rr]==[r['target_mos'] for r in qwen_subsets[1]]
        assert [r['selected_frame_positions'] for r in rr]==[r['selected_frame_positions'] for r in qwen_subsets[1]]
    for rr in qwen_fullsets.values():
        for key in ('sample_id','target_mos','selected_frame_positions'):
            assert [r[key] for r in rr]==[r[key] for r in qwen_fullsets[1]]
    ours=dict(model='LightGen-II',videos_per_field=16,optical_ms=t_opt,electronic_ms=t_el,
        time_16_videos_ms=ours_t,energy_16_videos_j=ours_e,amortized_energy_j_per_video=ours_e/16,
        videos_per_j=16/ours_e,simulation_reference_srcc=0.8043868643,physical_reference_srcc=0.7977138739,
        status='architecture/component timing estimate; no exact-PT end-to-end batch benchmark',
        source=str(latest_path.relative_to(REPO)))
    write_csv('timing_energy_recalculated.csv',timing)
    write_csv('batch_metric_audit.csv',metrics)
    qwen_figure=[r for r in metrics if r['model']=='Qwen3-VL-2B']
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,
                        'axes.linewidth':0.6,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
    for ax,key,label,our_value,panel in zip(axes,['energy_16_videos_j','time_16_videos_ms'],
            ['Energy per 16 videos (J)','Processing time per 16 videos (ms)'],[ours_e,ours_t],['a','b']):
        for model,color in COLORS.items():
            rs=[r for r in timing if r['model']==model]
            ax.plot([r['batch_size'] for r in rs],[r[key] for r in rs],'-o',color=color,lw=1,ms=3,label=model)
        ax.plot(16,our_value,'*',color='#7B2CBF',ms=9,label='LightGen-II estimate, N = 16')
        ax.set_xscale('log',base=2)
        ax.set_yscale('log')
        ax.set_xticks([1,2,4,8,16],[1,2,4,8,16])
        ax.set_xlim(.85,19)
        ax.set_xlabel('Videos per inference call')
        ax.set_ylabel(label)
        for s in ('top','right'): ax.spines[s].set_visible(False)
        ax.text(-.16,1.02,panel,transform=ax.transAxes,fontweight='bold',fontsize=8)
        ax.annotate(f'{our_value:.4f}',(16,our_value),xytext=(-3,9),textcoords='offset points',ha='right',fontsize=6)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=6)
    fig.subplots_adjust(left=.09,right=.985,bottom=.24,top=.80,wspace=.30)
    fig.text(.02,.08,'Qwen: preserved table B1; new interleaved B2/4/8. Other baselines: historical costs.',fontsize=6)
    fig.text(.02,.035,'LightGen-II: updated optical-cycle + electronic-component estimate; not bench end-to-end.',fontsize=6)
    for ext in ('png','svg'): fig.savefig(OUT/f'batch_cost_comparison.{ext}',dpi=600)
    plt.close(fig)
    # Retain the original quality-versus-cost layout, but do not masquerade
    # repeated reference scores as measured accuracy at each batch size.
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
    markers={'Qwen3-VL-2B':'o','DeepSeek-VL2-Tiny':'s','CLIP ViT-B/32':'D','YOLO11s':'^'}
    for ax,key,label,our_value,panel in zip(axes,['energy_16_videos_j','time_16_videos_ms'],
            ['Energy per 16 videos (J)','Processing time per 16 videos (ms)'],[ours_e,ours_t],['a','b']):
        for model,color in COLORS.items():
            rs=[r for r in timing if r['model']==model]
            actual=model=='Qwen3-VL-2B'
            ys=[next(m['measured_full558_batch_srcc'] for m in metrics if m['model']==model and m['batch_size']==r['batch_size'])
                if actual else REFERENCES[model] for r in rs]
            ax.plot([r[key] for r in rs],ys,marker=markers[model],color=color,lw=1,ms=3,
                linestyle='-' if actual else '--',markerfacecolor=color if actual else 'white',
                label=model+(' (measured at B)' if actual else ' (reference)'))
            if actual:
                for r,y in zip(rs,ys):
                    offset={1:(0,8),2:(8,-15),4:(-5,8),8:(-12,-10)}[r['batch_size']]
                    ax.annotate(f"B={r['batch_size']}",(r[key],y),xytext=offset,textcoords='offset points',ha='center',fontsize=5)
        ax.plot(our_value,ours['physical_reference_srcc'],'*',color='#7B2CBF',ms=9,
                label='LightGen-II (physical reference)')
        ax.annotate('0.7977',(our_value,ours['physical_reference_srcc']),xytext=(7,0),
                    textcoords='offset points',ha='left',va='center',fontsize=6)
        ax.set_xscale('log'); ax.set_ylim(.52,.82)
        ax.set_xlabel(label); ax.set_ylabel('Temporal quality SRCC')
        for s in ('top','right'): ax.spines[s].set_visible(False)
        ax.text(-.16,1.02,panel,transform=ax.transAxes,fontweight='bold',fontsize=8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=6)
    fig.subplots_adjust(left=.09,right=.985,bottom=.27,top=.76,wspace=.30)
    fig.text(.02,.12,'Open symbols/dashes: fixed full-TEST references, not measured SRCC(B). Qwen labels identify each batch.',fontsize=6)
    fig.text(.02,.075,'Qwen: new full558 SRCC; historical table B1 cost + new same-protocol B2/4/8. LightGen-II: component estimate.',fontsize=6)
    fig.text(.02,.03,'FP32 LightGen-II; BF16 Qwen/DeepSeek; FP16 CLIP/YOLO. Not equal-precision or bench end-to-end measurements.',fontsize=6)
    for ext in ('png','svg'): fig.savefig(OUT/f'srcc_cost_comparison_reference.{ext}',dpi=600)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,72/25.4))
    for ax,key,title in zip(axes,['measured_full558_batch_srcc','measured_first200_batch_srcc'],['Full TEST (558 videos)','Same first 200 TEST videos']):
        valid=[r for r in qwen_figure if r[key] is not None]
        ax.plot([r['batch_size'] for r in valid],[r[key] for r in valid],'o-',color='#0072B2',lw=1,ms=3)
        for r in valid: ax.annotate(f'{r[key]:.6f}',(r['batch_size'],r[key]),xytext=(0,8),textcoords='offset points',ha='center',fontsize=6)
        ax.set_xscale('log',base=2); ax.set_xticks([1,2,4,8],[1,2,4,8]); ax.set_xlim(.85,9.5)
        ax.set_xlabel('Batch size (videos)'); ax.set_ylabel('Temporal quality SRCC'); ax.set_title(title,fontsize=7)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.set_ylim((.750,.780) if key=='measured_full558_batch_srcc' else (.650,.675))
        for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.1,right=.98,bottom=.25,top=.86,wspace=.4)
    fig.text(.02,.08,'Qwen3-VL-2B: recovered full-TEST predictions; other models lack trained batch-specific predictions.',fontsize=6)
    fig.text(.02,.035,'No monotonic accuracy trend assumed. Both panels use identical videos within their respective subsets.',fontsize=6)
    for ext in ('png','svg'): fig.savefig(OUT/f'qwen_batch_metric_audit.{ext}',dpi=600)
    plt.close(fig)
    sources=[dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(SOURCE_FILES)]
    data=dict(ours=ours,timing=timing,metrics=metrics,sources=sources,rerun_audit=rerun_audit,
        offline_recalculation_used_gpu=False,source_gpu_reevaluation_date='2026-10-13',
        missing='Exact original trained temporal quality-row PTs for CLIP/YOLO/DeepSeek; batch-specific predictions unavailable',
        historical_builder='build_analysis.py; unchanged and not imported')
    (OUT/'recalculation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(OUT),ours=ours,rows=len(timing)),ensure_ascii=False))

if __name__=='__main__': main()
