"""Offline rederivation; never imports the historical side-effecting builder.

No weights, CUDA, optical SDK or video decoding are used. Raw files stay intact.
"""
from __future__ import annotations
import csv
import hashlib
import json
import statistics
import shutil
import sys
from collections import Counter
from math import ceil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT / 'recalculated_20261013'
RAW = ROOT / 'raw_remote'
REPRO = REPO / 'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_reproduction_a100_20261013'
SOURCE_FILES = set()
HOST_W = 338.2
COLORS = {'Qwen3-VL-2B':'#0072B2', 'DeepSeek-VL2-Tiny':'#D55E00',
          'CLIP ViT-B/32':'#009E73', 'YOLO11s':'#E69F00'}
REFERENCES = {'Qwen3-VL-2B':0.7663428036522781, 'DeepSeek-VL2-Tiny':0.5688169,
              'CLIP ViT-B/32':0.7251568, 'YOLO11s':0.7015822}
START_BLOCKS = {'Qwen3-VL-2B':'model.model.visual.blocks[0]',
                'CLIP ViT-B/32':'model.visual.transformer.resblocks[0]',
                'DeepSeek-VL2-Tiny':'model.vision.blocks[0]',
                'YOLO11s':'model.model[0] (first convolution block, not Transformer)'}

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
    def ranks(values):
        order=sorted(range(len(values)),key=values.__getitem__)
        result=[0.0]*len(values)
        start=0
        while start<len(order):
            end=start+1
            while end<len(order) and values[order[end]]==values[order[start]]: end+=1
            for i in order[start:end]: result[i]=(start+end+1)/2
            start=end
        return result
    return statistics.correlation(ranks([float(r['target_mos']) for r in rows]),
                                  ranks([float(r['prediction']) for r in rows]))

def audit_batches(report, timed, full, batch):
    """Audit actual distinct-video calls, including the non-padded TEST tail."""
    assert report['batch_size_videos']==batch
    assert report['explicit_warmup_forwards']==0 and report['first_test_batch_included']
    assert len(timed)==ceil(558/batch)
    assert [int(r['batch_index']) for r in timed]==list(range(len(timed)))
    assert [int(r['videos']) for r in timed]==[min(batch,558-i) for i in range(0,558,batch)]
    assert [int(r['sample_index']) for r in full]==list(range(558))
    assert [int(r['batch_index']) for r in full]==[i//batch for i in range(558)]
    assert [r['in_cost_subset']=='True' for r in timed]==[i<200//batch for i in range(len(timed))]
    cost=timed[:200//batch]
    assert abs(statistics.mean(float(r['synchronized_wall_ms']) for r in cost)
               -report['inference_batch_wall_ms']['mean'])<1e-10
    return dict(batch_size_videos=batch,full_test_videos=558,full_test_calls=len(timed),
                last_test_call_videos=int(timed[-1]['videos']),cost_videos=200,
                cost_calls=len(cost),calls_for_16=16//batch,frames_per_call=4*batch,
                first_call_included=True,explicit_warmup=0)

def log_axis(ax, axis, limits, ticks):
    getattr(ax,f'set_{axis}scale')('log')
    getattr(ax,f'set_{axis}lim')(*limits)
    getattr(ax,f'set_{axis}ticks')(ticks)
    getattr(ax,f'{axis}axis').set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))

def batch_labels(ax, points):
    """Keep callouts within the axes and clear of every marker/other label."""
    fig=ax.figure
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    occupied=[]
    for x,y,_,_ in points:
        px,py=ax.transData.transform((x,y))
        occupied.append(Bbox.from_extents(px-4,py-4,px+4,py+4))
    allowed=ax.get_window_extent(renderer).padded(-3)
    placements=[]
    for x,y,text,color in points:
        candidates=[(0,dy) for dy in (10,-10,20,-20,30,-30,40,-40,50,-50)]
        candidates += [(dx,dy) for dy in (10,-10,20,-20,30,-30,40,-40,50,-50) for dx in (-12,12,-24,24,-36,36)]
        chosen=None
        for offset in candidates:
            ann=ax.annotate(text,(x,y),xytext=offset,textcoords='offset points',
                            ha='center',va='center',fontsize=5.5,color=color,
                            bbox=dict(fc='white',ec='none',pad=.15))
            box=ann.get_window_extent(renderer).padded(2)
            if allowed.contains(box.x0,box.y0) and allowed.contains(box.x1,box.y1) and not any(box.overlaps(b) for b in occupied):
                chosen=(ann,box,offset)
                break
            ann.remove()
        if chosen is None:
            raise RuntimeError(f'No clear in-axis annotation position for {text}')
        ann,box,offset=chosen
        ann.remove()
        ax.annotate(text,(x,y),xytext=offset,textcoords='offset points',
                    ha='center',va='center',fontsize=5.5,color=color,
                    bbox=dict(fc='white',ec='none',pad=.15),
                    arrowprops=dict(arrowstyle='-',lw=.45,color=color,shrinkA=1,shrinkB=3))
        occupied.append(box)
        placements.append(dict(label=text,x=x,y=y,offset_points=offset))
    return placements

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
    call_audit = []
    split=read_csv(REPRO/'lgvq_train2250_test558.csv')
    assert Counter(r['split'] for r in split)=={'train':2250,'test':558}
    assert len({r['sample_id'] for r in split})==2808
    test_ids=[r['sample_id'] for r in split if r['split']=='test']
    qwen_subsets = {}
    qwen_fullsets = {}
    rerun_audit = []
    reproduction_audit = read_json(REPRO/'batch_audit.json')
    assert reproduction_audit['suite']['status']=='complete'
    assert len(reproduction_audit['batches'])==12
    assert all(not p['proc_exists'] for p in reproduction_audit['pid_checks'])
    for relative, expected in reproduction_audit['sha'].items():
        asset=REPRO/relative
        assert hashlib.sha256(asset.read_bytes()).hexdigest()==expected
        SOURCE_FILES.add(asset)
    reproduced_fullsets = {}
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
                    historical_calls=read_csv(path.parent/'per_video_predictions_and_timing.csv')
                    assert len(historical_calls)==200
                    for key in ('sample_id','target_mos','selected_frame_positions'):
                        assert [r[key] for r in historical_calls]==[r[key] for r in full[:200]]
                    wall=float(anchor['model_internal_synchronized_wall_ms_first_200_included']['mean'])
                    assert abs(statistics.mean(float(r['model_internal_synchronized_wall_ms'])
                               for r in historical_calls)-wall)<1e-10
                    wall_p95=float(anchor['model_internal_synchronized_wall_ms_first_200_included']['p95'])
                    power=float(anchor['power']['active_mean_w'])
                    cost_basis='Historical nine-task table B1 anchor; fresh B1 reevaluation retained separately'
                dtype='bfloat16'
            else:
                name={'CLIP ViT-B/32':'clip','DeepSeek-VL2-Tiny':'deepseek','YOLO11s':'yolo'}[model]
                path=REPRO/name/f'batch{batch:02d}'/'report.json'
                report=read_json(path)
                receipt_path=REPRO/name/'reproduction_receipt.json'
                receipt=read_json(receipt_path)
                assert receipt['origin']==report['checkpoint_origin']=='retrained_original_protocol'
                assert report['reproduction_receipt_sha256']==hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                assert report['status']=='complete' and report['test_videos']==558
                assert report['batch_size_videos']==batch and report['baseline']==name
                assert report['gpu_uuid']=='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
                assert report['actual_power_limit_w']==250
                assert report['manifest_sha256']=='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
                wall=float(report['inference_batch_wall_ms']['mean'])
                wall_p95=float(report['inference_batch_wall_ms']['p95'])
                power=float(report['power']['active_mean_w'])
                n=int(report['cost_test_videos'])
                timed=read_csv(path.parent/'batch_timing.csv')
                cost_rows=[r for r in timed if r['in_cost_subset']=='True']
                assert sum(int(r['videos']) for r in cost_rows)==200
                calls=len(cost_rows)
                assert abs(statistics.mean(float(r['synchronized_wall_ms']) for r in cost_rows)-wall)<1e-10
                power_rows=read_csv(path.parent/'power_samples.csv')
                active=[float(r['watts']) for r in power_rows if r['phase'].startswith('active:')]
                assert len(active)==report['power']['active_samples']
                assert abs(statistics.mean(active)-power)<1e-10
                pred_path=path.parent/'predictions.csv'
                full=read_csv(pred_path)
                assert len(full)==558 and len({r['sample_id'] for r in full})==558
                reproduced_fullsets[(model,batch)]=full
                full_srcc=srcc(full)
                subset_srcc=srcc(full[:200])
                assert abs(full_srcc-report['performance']['srcc'])<1e-12
                assert abs(report['cost']['latency_16_videos_ms']-wall*16/batch)<1e-8
                assert abs(report['cost']['energy_16_videos_j']-(HOST_W+power)*wall*16/batch/1000)<1e-8
                full_source=str(pred_path.relative_to(REPO))
                cost_basis='2026-10-13 retrained original-protocol head; interleaved frozen inference; first200 cost'
                dtype='bfloat16' if model=='DeepSeek-VL2-Tiny' else 'float16'
            assert [r['sample_id'] for r in full]==test_ids
            call_audit.append(dict(model=model,**audit_batches(report,timed,full,batch)))
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
                checkpoint_origin='original_frozen_quality_rows' if model=='Qwen3-VL-2B' else 'retrained_original_protocol',
                checkpoint_sha256=report['checkpoint_sha256'],
                delta_vs_historical_reference=full_srcc-REFERENCES[model],
                status='Original head; historical B1 cost, new B2/4/8' if model=='Qwen3-VL-2B' else 'Retrained head; TEST-selected development, no VAL'))
    first_ids=[r['sample_id'] for r in qwen_subsets[1]]
    for b, rr in qwen_subsets.items():
        assert [r['sample_id'] for r in rr]==first_ids
        assert [r['target_mos'] for r in rr]==[r['target_mos'] for r in qwen_subsets[1]]
        assert [r['selected_frame_positions'] for r in rr]==[r['selected_frame_positions'] for r in qwen_subsets[1]]
    for rr in qwen_fullsets.values():
        for key in ('sample_id','target_mos','selected_frame_positions'):
            assert [r[key] for r in rr]==[r[key] for r in qwen_fullsets[1]]
    reproduced_reference=next(iter(reproduced_fullsets.values()))
    for rr in reproduced_fullsets.values():
        for key in ('sample_id','target_mos','selected_frame_positions'):
            assert [r[key] for r in rr]==[r[key] for r in reproduced_reference]
        for key in ('sample_id','target_mos'):
            assert [r[key] for r in rr]==[r[key] for r in qwen_fullsets[1]]
    ours=dict(model='LightGen-II',videos_per_field=16,optical_ms=t_opt,electronic_ms=t_el,
        time_16_videos_ms=ours_t,energy_16_videos_j=ours_e,amortized_energy_j_per_video=ours_e/16,
        videos_per_j=16/ours_e,simulation_reference_srcc=0.8043868643,physical_reference_srcc=0.7977138739,
        status='architecture/component timing estimate; no exact-PT end-to-end batch benchmark',
        source=str(latest_path.relative_to(REPO)))
    plot_data=[dict(model='LightGen-II',batch_size_videos='',videos_per_optical_field=16,
        workload_videos=16,performance_videos=558,cost_sample_videos='',cost_sample_calls='',
        measured_full558_srcc=ours['physical_reference_srcc'],time_16_videos_ms=ours_t,
        energy_16_videos_j=ours_e,inference_precision='float32',
        performance_basis='Historical physical reference, not generated by timing proxy',
        cost_basis=ours['status'])]
    for t,m in zip(timing,metrics):
        assert (t['model'],t['batch_size'])==(m['model'],m['batch_size'])
        plot_data.append(dict(model=t['model'],batch_size_videos=t['batch_size'],videos_per_optical_field='',
            workload_videos=16,performance_videos=558,cost_sample_videos=200,cost_sample_calls=t['measured_calls'],
            measured_full558_srcc=m['measured_full558_batch_srcc'],time_16_videos_ms=t['time_16_videos_ms'],
            energy_16_videos_j=t['energy_16_videos_j'],inference_precision=t['inference_precision'],
            performance_basis=m['status'],cost_basis=t['cost_basis']))
    write_csv('plot_data.csv',plot_data)
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
        log_axis(ax,'y',(.7,3000) if key.startswith('energy') else (7,5000),
                 [1,10,100,1000,3000] if key.startswith('energy') else [10,100,1000,5000])
        ax.set_xticks([1,2,4,8,16],[1,2,4,8,16])
        ax.set_xlim(.85,19)
        ax.set_xlabel('Videos per call / optical field')
        ax.set_ylabel(label)
        for s in ('top','right'): ax.spines[s].set_visible(False)
        ax.text(-.16,1.02,panel,transform=ax.transAxes,fontweight='bold',fontsize=8)
        ax.annotate(f'{our_value:.4f}',(16,our_value),xytext=(-3,9),textcoords='offset points',ha='right',fontsize=6)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=6)
    fig.subplots_adjust(left=.09,right=.985,bottom=.24,top=.80,wspace=.30)
    fig.text(.02,.08,'Qwen: preserved table B1; new B2/4/8. CLIP/YOLO/DeepSeek: original-protocol retrained heads.',fontsize=6)
    fig.text(.02,.035,'LightGen-II: updated optical-cycle + electronic-component estimate; not bench end-to-end.',fontsize=6)
    for ext in ('png','svg'): fig.savefig(OUT/f'batch_cost_comparison.{ext}',dpi=600)
    plt.close(fig)
    # Retain the original quality-versus-cost layout, but do not masquerade
    # repeated reference scores as measured accuracy at each batch size.
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,105/25.4))
    markers={'Qwen3-VL-2B':'o','DeepSeek-VL2-Tiny':'s','CLIP ViT-B/32':'D','YOLO11s':'^'}
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.80,wspace=.30)
    annotation_audit={}
    for ax,key,label,our_value,panel in zip(axes,['energy_16_videos_j','time_16_videos_ms'],
            ['Energy per 16 videos (J)','Processing time per 16 videos (ms)'],[ours_e,ours_t],['a','b']):
        points=[]
        for model,color in COLORS.items():
            rs=[r for r in timing if r['model']==model]
            ys=[next(m['measured_full558_batch_srcc'] for m in metrics if m['model']==model and m['batch_size']==r['batch_size'])
                for r in rs]
            ax.plot([r[key] for r in rs],ys,marker=markers[model],color=color,lw=1,ms=3,
                label=model)
            for r,y in zip(rs,ys):
                points.append((r[key],y,f"B={r['batch_size']}",color))
        ax.plot(our_value,ours['physical_reference_srcc'],'*',color='#7B2CBF',ms=9,
                label='LightGen-II (physical reference)')
        points.append((our_value,ours['physical_reference_srcc'],'N=16','#7B2CBF'))
        log_axis(ax,'x',(.7,3000) if key.startswith('energy') else (7,5000),
                 [1,10,100,1000,3000] if key.startswith('energy') else [10,100,1000,5000])
        ax.set_ylim(0,1); ax.set_yticks([0,.2,.4,.6,.8,1])
        ax.set_xlabel(label); ax.set_ylabel('Temporal quality SRCC')
        for s in ('top','right'): ax.spines[s].set_visible(False)
        ax.text(-.16,1.02,panel,transform=ax.transAxes,fontweight='bold',fontsize=8)
        annotation_audit[panel]=batch_labels(ax,points)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=6)
    fig.text(.02,.12,'B = videos per GPU call; N = videos per optical field. Every baseline SRCC uses all 558 TEST videos.',fontsize=6)
    fig.text(.02,.075,'Qwen B1: table cost anchor. Other heads: original-protocol retraining. LightGen-II: component-cost estimate.',fontsize=6)
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
        ax.set_ylim(0,1); ax.set_yticks([0,.2,.4,.6,.8,1])
        for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.1,right=.98,bottom=.25,top=.86,wspace=.4)
    fig.text(.02,.08,'Qwen3-VL-2B: original frozen head reevaluated on all 558 TEST videos at each batch size.',fontsize=6)
    fig.text(.02,.035,'No monotonic accuracy trend assumed. Both panels use identical videos within their respective subsets.',fontsize=6)
    for ext in ('png','svg'): fig.savefig(OUT/f'qwen_batch_metric_audit.{ext}',dpi=600)
    plt.close(fig)
    sources=[dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(SOURCE_FILES)]
    descriptions={
      'Summary': 'The LGVQ split contains 2,250 TRAIN and 558 TEST videos (2,808 unique videos). Each of the 16 frozen model/batch settings is evaluated on the same 558 TEST videos; column C is the actual batch-specific SRCC and column I is historical reference only. Costs use the first 200 TEST videos, not a 16-video-only test. B=1/2/4/8 means 200/100/50/25 measured calls and 16/8/4/2 calls for the normalized 16-video workload. Each video contributes four frames. LightGen-II is one N=16 optical-field point: physical reference SRCC 0.7977138739, component-estimated time 10.245569591370225 ms and energy 1.1776895097085185 J. Qwen B=1 retains the total-table anchor: SRCC 0.7663428036522781, time 1188.3861464262009 ms and energy 462.1134010767563 J. Do not plot historical references or subset-200 SRCC as batch-specific full-TEST results.',
      'Batch results': 'Each batch setting runs in a fresh process on the specified NVIDIA A100-PCIE-40GB. For each batch, CPU decoding/cropping/processing and H2D transfer precede frozen inference; these audit stages are excluded from final costs. A host timer starts at the forward-pre-hook of the first visual Transformer block (Qwen, CLIP, DeepSeek), or first convolution block (YOLO), and stops after the MOS score is ready on GPU and CUDA synchronization completes. GPU-to-CPU prediction copying is excluded. There is no explicit warm-up; the first measured call is included. T16=(16/B)*mean batch wall time; E16=(338.2 W host proxy + measured active A100 W)*T16/1000. Full TEST uses 558/279/140/70 calls; the last B=4/8 calls contain 2/6 real videos, not padded examples. Column U is actual full558 SRCC; C is historical reference. The pre-hook has no internal CUDA sync, so this is an operational synchronized-wall boundary, not a pure first-block kernel-duration integral.',
      'Sources': 'All 16 full-TEST SRCC values are recomputed independently from the corresponding saved predictions and matched sample IDs, MOS labels and within-model selected frames. Qwen uses its original frozen quality rows. CLIP, YOLO and DeepSeek use newly reproduced original-protocol heads because historical PTs were not recovered; their highest-TEST selection is a development metric, with no VAL. Frozen evaluation adds no training. Historical Qwen B=1 costs are retained; the new B=1 audit (1150.4916036128998 ms, 442.9714733293941 J per 16 videos) is not used in final data. LightGen-II uses the previously approved six optical cycles (6.2682 ms) plus serial electronic components (3.9773695913702247 ms), not a new end-to-end optical measurement. Its electronic timing is FP32 CUDA-event component timing; baselines use FP16/BF16 synchronized wall timing. A100 power is a 50-ms tagged-window mean multiplied by boundary time, not calibrated per-call energy. Files, checkpoint origins and SHA-256 remain separately traceable.'}
    data=dict(ours=ours,timing=timing,metrics=metrics,sources=sources,rerun_audit=rerun_audit,
        plot_data=plot_data,table_descriptions=descriptions,call_audit=call_audit,start_blocks=START_BLOCKS,
        annotation_audit=annotation_audit,presentation_revision='2026-10-14; all scientific values unchanged',
        offline_recalculation_used_gpu=False,source_gpu_reevaluation_date='2026-10-13',
        missing='Original CLIP/YOLO/DeepSeek trained heads remain unrecovered; authorized new heads are explicitly retrained_original_protocol',
        reproduction_run=str(REPRO.relative_to(REPO)),
        reproduction_audit=str((REPRO/'batch_audit.json').relative_to(REPO)),
        historical_builder='build_analysis.py; unchanged and not imported')
    (OUT/'recalculation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(OUT),ours=ours,rows=len(timing)),ensure_ascii=False))

def annotate_utilization():
    """Figure-only correction from immutable report JSON; never changes measured data."""
    data=json.loads((OUT/'recalculation.json').read_text(encoding='utf8'))
    rows,ours=data['paired_rows'],data['ours']
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.linewidth':.65,'svg.fonttype':'none'})
    markers={'Qwen3-VL-2B':'o','DeepSeek-VL2-Tiny':'s','CLIP ViT-B/32':'D','YOLO11s':'^'}
    annotation_record={}
    for resident,stem in [(False,'srcc_cost_comparison_reference'),(True,'srcc_cost_comparison_resident')]:
        fig,axes=plt.subplots(1,2,figsize=(183/25.4,130/25.4))
        fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.78,wspace=.32)
        fig.suptitle('GPU-resident repeated inference' if resident else 'Interleaved preparation and inference',fontsize=9,y=.86)
        audit={}
        for ax,key,label,panel in zip(axes,['energy_16_videos_j','time_16_videos_ms'],['Energy per 16 videos (J)','Inference time per 16 videos (ms)'],['a','b']):
            field='sustained_'+key if resident else key
            utilization='sustained_gpu_util_mean_percent' if resident else 'gpu_util_mean_percent'
            points=[]
            for model,color in COLORS.items():
                rr=[r for r in rows if r['model']==model]
                ax.plot([r[field] for r in rr],[r['srcc'] for r in rr],color=color,marker=markers[model],ms=3.5,lw=1,label=model)
                points += [(r[field],r['srcc'],f"B={r['batch_size']}\nU={r[utilization]:.1f}%",color) for r in rr]
            ax.plot(ours[key],ours['physical_reference_srcc'],'*',ms=10,color='#7B2CBF',label='LightGen-II (reference)')
            points.append((ours[key],ours['physical_reference_srcc'],'N=16\nU: n.a.','#7B2CBF'))
            limits=(.7,2000) if key.startswith('energy') else (7,4000)
            # Some resident CLIP costs fall below the optical reference.
            log_axis(ax,'x',limits,[1,10,100,1000] if key.startswith('energy') else [10,100,1000])
            ax.set_ylim(0,1); ax.set_yticks([0,.2,.4,.6,.8,1]); ax.set_xlabel(label); ax.set_ylabel('Temporal quality SRCC')
            ax.text(-.17,1.02,panel,transform=ax.transAxes,fontweight='bold',fontsize=10)
            for side in ('top','right'): ax.spines[side].set_visible(False)
            audit[panel]=batch_labels(ax,points)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7)
        fig.text(.02,.14,'U = mean GPU kernel-busy utilization, not peak utilization or SM occupancy.',fontsize=6.5)
        fig.text(.02,.10,'Paired costs and U: 15-s resident-input loop; SRCC: separate full558 pass at the same PT/B.' if resident else
                 'Paired costs and U: first200 TEST inference-tagged windows. CPU preparation excluded from cost.',fontsize=6.5)
        fig.text(.02,.06,'LightGen-II: unchanged N=16 component-cost estimate and physical SRCC; GPU U not measured.',fontsize=6.5)
        fig.text(.02,.025,'14 October 2026 full02. No timing, energy, prediction or utilization measurement has been changed.',fontsize=6.5)
        for ext in ('png','svg'): fig.savefig(OUT/f'{stem}.{ext}',dpi=450)
        plt.close(fig)
        annotation_record[stem]=audit
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,100/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.75,wspace=.35)
    for ax,key,title in zip(axes,['gpu_util_mean_percent','sustained_gpu_util_mean_percent'],['Interleaved inference','GPU-resident repeated input']):
        points=[]
        for model,color in COLORS.items():
            rr=[r for r in rows if r['model']==model]
            ax.plot([r['batch_size'] for r in rr],[r[key] for r in rr],'-o',ms=3,lw=1,color=color,label=model)
            points += [(r['batch_size'],r[key],f"{r[key]:.1f}%",color) for r in rr]
        ax.set_xlim(.5,10.5); ax.set_ylim(-5,115); ax.set_xticks([1,2,3,4,8,9,10]); ax.set_yticks([0,20,40,60,80,100])
        ax.set_xlabel('Batch size (videos)'); ax.set_ylabel('Mean GPU utilization (%)'); ax.set_title(title,fontsize=8)
        for side in ('top','right'): ax.spines[side].set_visible(False)
        if key=='sustained_gpu_util_mean_percent':
            batch_labels(ax,points)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7)
    fig.text(.02,.09,'Means, not peaks. Resident input still uses per-call synchronization and CPU score copies.',fontsize=6.5)
    fig.text(.02,.045,'Each workload retains its own paired latency and energy; no values were substituted.',fontsize=6.5)
    for ext in ('png','svg'): fig.savefig(OUT/f'gpu_utilization_comparison.{ext}',dpi=450)
    plt.close(fig)
    (OUT/'utilization_annotation_audit.json').write_text(json.dumps(annotation_record,indent=2)+'\n')
    print('Annotated all19 electronic points with workload-matched mean GPU utilization; measurements unchanged.')


def refresh_paired():
    """Publish only the completed full02 round, with separate paired load regimes."""
    run=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_temporal_telemetry_a100_20261014/full_remeasurement02'
    archive=run/'report_before_refresh'
    if not archive.exists():
        archive.mkdir()
        records=[]
        for p in OUT.iterdir():
            if p.is_file():
                shutil.copy2(p,archive/p.name)
                records.append(dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        (archive/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    old=read_json(archive/'recalculation.json')
    ours=old['ours']
    audit=read_json(run/'measurement_audit_check07.json')
    assert audit['measurement_groups']==19 and audit['all_process_audits_only_own_pid']
    groups={'Qwen3-VL-2B':('qwen',(1,2,4,8)), 'CLIP ViT-B/32':('clip',(1,2,4,8,9,10)),
            'DeepSeek-VL2-Tiny':('deepseek',(1,2,4,8)), 'YOLO11s':('yolo',(1,2,3,4,8))}
    rows=[]; metric_rows=[]; reference=None
    for model,(name,batches) in groups.items():
        for batch in batches:
            folder=run/f'{name}_batch{batch:02d}'
            r=read_json(folder/'report.json'); p=read_csv(folder/'predictions.csv'); t=read_csv(folder/'batch_timing.csv')
            assert r['status']=='complete' and len(p)==len({x['sample_id'] for x in p})==558
            identity=[(x['sample_id'],x['target_mos']) for x in p]
            if reference is None: reference=identity
            assert identity==reference
            cost=[x for x in t if x['in_cost_subset']=='True']
            assert sum(int(x['videos']) for x in cost)==200
            time=sum(float(x['synchronized_wall_ms']) for x in cost)*16/200
            energy=(338.2+r['power']['active_mean_w'])*time/1000
            assert abs(time-r['cost']['latency_16_videos_ms'])<1e-9
            assert abs(energy-r['cost']['energy_16_videos_j'])<1e-9
            score=r['performance']['srcc']; assert abs(srcc(p)-score)<1e-12
            tele=r['telemetry']; s=r['sustained_resident_input']; st=s['telemetry']
            raw_tele=read_csv(folder/'telemetry.csv')
            active=[v for v in raw_tele if v['phase'].startswith('active:')]
            assert len(active)==tele['samples']
            for column,key in [('watts','mean_board_power_w'),('utilization_percent','mean_gpu_utilization_percent'),('memory_used_mib','mean_memory_used_mib')]:
                assert abs(statistics.mean(float(v[column]) for v in active)-tele[key])<1e-9
            raw_sust=read_csv(folder/'sustained_telemetry.csv')
            sustained_active=[v for v in raw_sust if v['phase'].startswith('sustained:')]
            assert len(sustained_active)==st['samples']
            for column,key in [('watts','mean_board_power_w'),('utilization_percent','mean_gpu_utilization_percent')]:
                assert abs(statistics.mean(float(v[column]) for v in sustained_active)-st[key])<1e-9
            assert abs(s['energy_16_videos_j']-(338.2+st['mean_board_power_w'])*s['latency_16_videos_ms']/1000)<1e-9
            row=dict(model=model,batch_size=batch,full_test_videos=558,cost_videos=200,cost_calls=len(cost),
                     full_test_calls=len(t),cost_last_call_videos=int(cost[-1]['videos']),test_last_call_videos=int(t[-1]['videos']),
                     srcc=score,first200_srcc=srcc(p[:200]),inference_precision='BF16' if name in ('qwen','deepseek') else 'FP16',
                     wall_mean_ms=r['inference_batch_wall_ms']['mean'],wall_p95_ms=r['inference_batch_wall_ms']['p95'],
                     cost_wall_sum_ms=sum(float(x['synchronized_wall_ms']) for x in cost),host_w=338.2,
                     time_16_videos_ms=time,energy_16_videos_j=energy,board_mean_w=tele['mean_board_power_w'],
                     board_peak_w=r['power']['active_peak_w'],gpu_util_mean_percent=tele['mean_gpu_utilization_percent'],
                     gpu_util_peak_percent=tele['peak_gpu_utilization_percent'],memory_mean_mib=tele['mean_memory_used_mib'],
                     memory_peak_mib=tele['peak_memory_used_mib'],active_samples=tele['samples'],
                     sustained_time_16_videos_ms=s['latency_16_videos_ms'],sustained_energy_16_videos_j=s['energy_16_videos_j'],
                     sustained_board_mean_w=st['mean_board_power_w'],sustained_gpu_util_mean_percent=st['mean_gpu_utilization_percent'],
                     sustained_gpu_util_peak_percent=st['peak_gpu_utilization_percent'],sustained_memory_peak_mib=st['peak_memory_used_mib'],
                     sustained_calls=s['calls'],sustained_duration_s=s['duration_seconds'],
                     speedup_vs_ours=time/ours['time_16_videos_ms'],energy_gain_vs_ours=energy/ours['energy_16_videos_j'],
                     checkpoint_sha256=r['checkpoint_sha256'],checkpoint_origin=r['checkpoint_origin'],
                     source=str((folder/'report.json').relative_to(REPO)).replace('\\','/'),measurement_date='2026-10-14',
                     regime='interleaved CPU preparation and GPU inference; costs exclude preparation')
            rows.append(row)
            metric_rows.append(dict(model=model,batch_size=batch,measured_full558_batch_srcc=score,
                measured_first200_batch_srcc=row['first200_srcc'],historical_reference_srcc=REFERENCES[model],
                delta_vs_historical_reference=score-REFERENCES[model],checkpoint_sha256=r['checkpoint_sha256'],
                checkpoint_origin=r['checkpoint_origin'],performance_source=str((folder/'predictions.csv').relative_to(REPO)),
                status='Frozen evaluation; inherited TEST-selected development head, no VAL'))
    diagnostics={}
    for name,relative in [('qwen','diagnostic_qwen'),('deepseek','diagnostic_deepseek_interfacefix')]:
        d=read_json(run/relative/'diagnosis.json'); status=read_json(run/relative/'status.json')
        assert status['status']=='complete'
        assert all(d['inputs'][str(b)]['inputs_equal_to_single'] and d['inputs'][str(b)]['frames_equal_to_single'] for b in (2,4))
        assert d['probes']['native']==d['probes']['native_repeat'] or all(
            d['probes']['native'][str(b)]['predictions']==d['probes']['native_repeat'][str(b)]['predictions'] for b in (1,2,4))
        diagnostics[name]={mode:{b:v['max_abs_mos_delta_vs_b1'] for b,v in bs.items()} for mode,bs in d['probes'].items()}
    plot=[dict(model='LightGen-II',batch_size_videos='',videos_per_optical_field=16,workload_videos=16,
               measured_full558_srcc=ours['physical_reference_srcc'],time_16_videos_ms=ours['time_16_videos_ms'],
               energy_16_videos_j=ours['energy_16_videos_j'],gpu_util_mean_percent='',gpu_util_peak_percent='',
               memory_peak_mib='',cost_basis='optical-cycle and electronic-component estimate; physical SRCC reference')]
    plot += [dict(model=r['model'],batch_size_videos=r['batch_size'],videos_per_optical_field='',workload_videos=16,
                  measured_full558_srcc=r['srcc'],time_16_videos_ms=r['time_16_videos_ms'],energy_16_videos_j=r['energy_16_videos_j'],
                  gpu_util_mean_percent=r['gpu_util_mean_percent'],gpu_util_peak_percent=r['gpu_util_peak_percent'],
                  memory_peak_mib=r['memory_peak_mib'],cost_basis='2026-10-14 full02 interleaved inference, same-round telemetry') for r in rows]
    write_csv('plot_data.csv',plot); write_csv('timing_energy_recalculated.csv',rows); write_csv('batch_metric_audit.csv',metric_rows)
    continuous=[dict(model=r['model'],batch_size_videos=r['batch_size'],srcc_full558_same_pt_batch=r['srcc'],
                    latency16_ms=r['sustained_time_16_videos_ms'],energy16_j=r['sustained_energy_16_videos_j'],
                    gpu_util_mean_percent=r['sustained_gpu_util_mean_percent'],gpu_util_peak_percent=r['sustained_gpu_util_peak_percent'],
                    board_mean_w=r['sustained_board_mean_w'],memory_peak_mib=r['sustained_memory_peak_mib'],
                    calls=r['sustained_calls'],duration_s=r['sustained_duration_s'],
                    workload='15-second repeated GPU-resident input; SRCC from separate full558 pass at same PT/B') for r in rows]
    write_csv('sustained_load_data.csv',continuous)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.linewidth':.65,'svg.fonttype':'none'})
    markers={'Qwen3-VL-2B':'o','DeepSeek-VL2-Tiny':'s','CLIP ViT-B/32':'D','YOLO11s':'^'}
    annotations={}
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,110/25.4))
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.81,wspace=.30)
    for ax,key,label,panel in zip(axes,['energy_16_videos_j','time_16_videos_ms'],['Energy per 16 videos (J)','Inference time per 16 videos (ms)'],['a','b']):
        points=[]
        for model,color in COLORS.items():
            rr=[r for r in rows if r['model']==model]
            ax.plot([r[key] for r in rr],[r['srcc'] for r in rr],color=color,marker=markers[model],ms=3.5,lw=1,label=model)
            points += [(r[key],r['srcc'],f"B={r['batch_size']}",color) for r in rr]
        ax.plot(ours[key],ours['physical_reference_srcc'],'*',ms=10,color='#7B2CBF',label='LightGen-II (reference)')
        points.append((ours[key],ours['physical_reference_srcc'],'N=16','#7B2CBF'))
        log_axis(ax,'x',(.7,2000) if key.startswith('energy') else (7,4000),[1,10,100,1000] if key.startswith('energy') else [10,100,1000])
        ax.set_ylim(0,1); ax.set_yticks([0,.2,.4,.6,.8,1]); ax.set_xlabel(label); ax.set_ylabel('Temporal quality SRCC')
        ax.text(-.17,1.02,panel,transform=ax.transAxes,fontweight='bold',fontsize=10)
        for side in ('top','right'): ax.spines[side].set_visible(False)
        annotations[panel]=batch_labels(ax,points)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7)
    fig.text(.02,.10,'Electronic baselines: frozen full02 measurements, 14 October 2026. B = videos per GPU call.',fontsize=6.5)
    fig.text(.02,.045,'LightGen-II: physical SRCC reference and component-cost estimate at N = 16; historical Qwen B1 excluded.',fontsize=6.5)
    for ext in ('png','svg'): fig.savefig(OUT/f'srcc_cost_comparison_reference.{ext}',dpi=450)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,78/25.4))
    for ax,key,title in zip(axes,['gpu_util_mean_percent','sustained_gpu_util_mean_percent'],['Interleaved inference','Continuous GPU-resident input']):
        for model,color in COLORS.items():
            rr=[r for r in rows if r['model']==model]
            ax.plot([r['batch_size'] for r in rr],[r[key] for r in rr],'-o',ms=3,lw=1,color=color,label=model)
        ax.set_xlim(.5,10.5); ax.set_ylim(0,100); ax.set_xticks([1,2,3,4,8,9,10]); ax.set_yticks([0,20,40,60,80,100])
        ax.set_xlabel('Batch size (videos)'); ax.set_ylabel('Mean GPU utilization (%)'); ax.set_title(title,fontsize=8)
        for side in ('top','right'): ax.spines[side].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7)
    fig.subplots_adjust(left=.10,right=.98,bottom=.21,top=.72,wspace=.35)
    fig.text(.02,.045,'Kernel-busy fraction over the driver window; each workload has its own paired latency and energy.',fontsize=6.5)
    for ext in ('png','svg'): fig.savefig(OUT/f'gpu_utilization_comparison.{ext}',dpi=450)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,75/25.4))
    qwen=[r for r in rows if r['model']=='Qwen3-VL-2B']
    for ax,key,title in zip(axes,['srcc','first200_srcc'],['Full TEST (558 videos)','Same first 200 TEST videos']):
        ax.plot([r['batch_size'] for r in qwen],[r[key] for r in qwen],'o-',color=COLORS['Qwen3-VL-2B'],lw=1,ms=3)
        for r in qwen: ax.annotate(f"{r[key]:.6f}",(r['batch_size'],r[key]),xytext=(0,8),textcoords='offset points',ha='center',fontsize=6)
        ax.set_xscale('log',base=2); ax.set_xlim(.8,10); ax.set_xticks([1,2,4,8],[1,2,4,8]); ax.set_ylim(0,1); ax.set_yticks([0,.2,.4,.6,.8,1])
        ax.set_xlabel('Batch size (videos)'); ax.set_ylabel('Temporal quality SRCC'); ax.set_title(title,fontsize=8)
        for side in ('top','right'): ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.24,top=.84,wspace=.35)
    fig.text(.02,.06,'Qwen3-VL-2B, frozen full02 measurements, 14 October 2026; identical TEST identities at each batch.',fontsize=6.5)
    for ext in ('png','svg'): fig.savefig(OUT/f'qwen_batch_metric_audit.{ext}',dpi=450)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,90/25.4))
    for ax,key,label in zip(axes,['energy_16_videos_j','time_16_videos_ms'],['Energy per 16 videos (J)','Inference time per 16 videos (ms)']):
        for model,color in COLORS.items():
            rr=[r for r in rows if r['model']==model]
            ax.plot([r['batch_size'] for r in rr],[r[key] for r in rr],'-o',color=color,ms=3,label=model)
        ax.plot(16,ours[key],'*',color='#7B2CBF',ms=9,label='LightGen-II (N=16 reference)')
        ax.set_yscale('log'); ax.set_ylim((.7,2000) if key.startswith('energy') else (7,4000)); ax.set_xlim(.5,17)
        ax.set_xticks([1,2,3,4,8,9,10,16]); ax.set_xlabel('Videos per GPU call / optical field'); ax.set_ylabel(label)
        for side in ('top','right'): ax.spines[side].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7)
    fig.subplots_adjust(left=.10,right=.98,bottom=.18,top=.78,wspace=.35)
    for ext in ('png','svg'): fig.savefig(OUT/f'batch_cost_comparison.{ext}',dpi=450)
    plt.close(fig)
    descriptions={
      'Summary':'LGVQ includes 2,250 TRAIN and 558 TEST videos. All 19 frozen model/batch settings use the same 558 TEST videos for independently computed SRCC. First-200 synchronized inference costs are normalized to 16 videos. The 20 plotted points include one unchanged N=16 LightGen-II component-cost estimate. Historical Qwen B1 is retained separately, without invented utilization. Main interleaved and 15-second resident-input utilization are distinct measurements and never exchanged.',
      'Batch results':'Each GPU batch is prepared on CPU and transferred before frozen inference. Timing starts at the first native backbone block and ends at GPU MOS readiness with synchronization; YOLO uses a convolution block. Decode, preprocessing and H2D remain audit-only. T16=sum of first-200 inference wall times times16/200, including partial batches at200 and558. E16=(338.2W host proxy+same-round tagged-active board meanW)*T16/1000. Continuous-load columns use their own 15-second resident-input timing and telemetry, not the main costs.',
      'Sources':'Measurements retain identical manifest and per-model checkpoint hashes. Qwen uses its exact historical head; CLIP, YOLO and DeepSeek use original-protocol reproduced heads, TEST-selected development metrics with no VAL. FP16/BF16 formal precision is unchanged. Eight-video probes have identical cross-batch inputs, deterministic repeats and markedly smaller MOS differences after casting deployed weights to FP32. Driver utilization is kernel-busy fraction, not SM occupancy. Power samples are coarse estimates, not calibrated short-call integrals.'}
    data=dict(ours=ours,paired_rows=rows,plot_data=plot,metrics=metric_rows,continuous_rows=continuous,
              diagnostics=diagnostics,annotation_audit=annotations,table_descriptions=descriptions,
              measurement_round=str(run.relative_to(REPO)).replace('\\','/'),measurement_date='2026-10-14',
              historical_qwen_b1=dict(srcc=.7663428036522781,time16_ms=1188.3861464262009,energy16_j=462.1134010767563,gpu_util=None),
              sources=[dict(path=str(p.relative_to(REPO)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(SOURCE_FILES)])
    (OUT/'recalculation.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    def table(headers,records):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(map(str,record))+' |' for record in records])
    paper_table=table(['Model','B / N','SRCC (558 TEST)','Time/16 (ms)','Energy/16 (J)','Active util mean (%)','Peak VRAM (MiB)'],
        [['LightGen-II','N=16',f"{ours['physical_reference_srcc']:.10f}",f"{ours['time_16_videos_ms']:.6f}",f"{ours['energy_16_videos_j']:.6f}",'not measured','not measured']]+
        [[r['model'],r['batch_size'],f"{r['srcc']:.10f}",f"{r['time_16_videos_ms']:.6f}",f"{r['energy_16_videos_j']:.6f}",f"{r['gpu_util_mean_percent']:.3f}",f"{r['memory_peak_mib']:.0f}"] for r in rows])
    sustained_table=table(['Model','B','Time/16 (ms)','Energy/16 (J)','Mean util (%)','Peak util (%)','Mean board power (W)','Peak VRAM (MiB)'],
        [[r['model'],r['batch_size'],f"{r['sustained_time_16_videos_ms']:.6f}",f"{r['sustained_energy_16_videos_j']:.6f}",f"{r['sustained_gpu_util_mean_percent']:.3f}",f"{r['sustained_gpu_util_peak_percent']:.0f}",f"{r['sustained_board_mean_w']:.3f}",f"{r['sustained_memory_peak_mib']:.0f}"] for r in rows])
    diag_table=table(['Model','Probe','B2–B1 max absolute MOS difference','B4–B1 max absolute MOS difference'],
        [[model,mode,f"{v['2']:.9f}",f"{v['4']:.9f}"] for model,modes in diagnostics.items() for mode,v in modes.items()])
    plan_table=table(['B','Full TEST calls','Final TEST call videos','First200 cost calls','Final cost call videos'],
        [[r['batch_size'],r['full_test_calls'],r['test_last_call_videos'],r['cost_calls'],r['cost_last_call_videos']] for r in rows if r['model']=='CLIP ViT-B/32']+
        [[3,next(r for r in rows if r['batch_size']==3)['full_test_calls'],1,67,2]])
    q=[r for r in rows if r['model']=='Qwen3-VL-2B']; q1=q[0]; q8=q[-1]
    text=f'''# LGVQ temporal quality: matched A100 inference costs and batch diagnostics

Updated 14 October 2026. All 19 frozen settings and both bounded numerical diagnoses are complete. The displayed curves now use only the uninterrupted `full_remeasurement02` measurements. Interrupted rounds and the initial failed DeepSeek input-fingerprint diagnostic remain preserved, excluded from final data. The original report location and three-workbook-sheet structure are retained.

## 两图含义与实验方法

图a展示每16视频推理能量与时间质量SRCC的关系，图b展示相同工作量的推理时间与SRCC的关系。横轴分别为Energy per 16 videos (J)和Inference time per 16 videos (ms)，使用对数坐标；纵轴为无量纲Temporal quality SRCC，范围0–1。B为每次GPU调用的视频数，N为每幅光场的视频数。每一个电子点都有其batch自己的558个TEST预测和同轮成本，不复制统一分数。LightGen-II仅为N=16物理SRCC参考和部件成本估算点，不是本轮重测光路。

LGVQ固定划分含2250 TRAIN、558 TEST，共2808个不同视频。每视频在10%、37%、63%、90%相对位置取4帧。Qwen/DeepSeek/CLIP/YOLO输入分别448/384/224/640像素正方形；后三者沿原短边65%中心裁剪。权重冻结，eval模式；各模型各B独立新进程。Qwen采用原历史head，其他三者采用2026-10-13原协议重训head，均保持PT/manifest SHA不变。重训head为50epoch/seed42/五级质量行/冻结backbone/CE，最高TEST选模为开发指标，无VAL。本轮无训练或选模。正式骨干精度Qwen与DeepSeek为BF16，CLIP与YOLO为FP16，质量头FP32；后三者保留FP16特征量化再FP32读出。

每批先CPU解码、裁剪和processor，再H2D，再推理，随后下一批。计时从首原生backbone block入口到GPU标量MOS就绪并完成同步；YOLO首block为卷积，其余为视觉Transformer。模型加载、CPU准备、H2D、之后CPU预测复制和审计开销不计最终成本。起点预钩子内无额外同步，因此为操作定义的同步墙钟边界，不是纯GPU kernel积分。无显式预热，第一调用保留。完整558视频统计SRCC，成本仅固定前200视频，并在200/558边界拆批，不丢尾样本。

T16=sum(first200 synchronized inference wall ms)×16/200；E16=(338.2 W主机代理+同轮tagged-active A100平均板卡W)×T16/1000。B=1/2/4/8时等价于mean batch ms×16/B；B=3/9/10含实际不足满批的尾批，不能用均值×16/B偷换。16仅为吞吐工作量归一化，不是仅测试16视频。

{plan_table}

功率、GPU利用率与显存同轮按50ms轮询并标记阶段。主表统计只取前200视频各调用的active标签窗口，与成本子集配套；后358视频仍计算性能，但不混入这些active均值。驱动自己的平均窗口并非50ms瞬时值，util是kernel-busy而非SM occupancy或峰值FLOPS使用率。短推理窗口可包含前序CPU间隔影响，CLIP/YOLO主交错util较低不代表算子没有GPU运行。均值功率×边界时间是沿用的能量估算，不是校准短调用积分。A100设250W；原始短功率读数中存在超过250W的峰值（DeepSeek约321–342W），保留不裁掉，不把设置值描述为瞬时硬限。

## 图a：能量与时间质量性能

LightGen-II物理SRCC为{ours['physical_reference_srcc']:.4f}，每16视频能量估算{ours['energy_16_videos_j']:.4f}J。新同轮Qwen B1为{q1['energy_16_videos_j']:.4f}J，B8为{q8['energy_16_videos_j']:.4f}J，分别对应{q1['energy_gain_vs_ours']:.2f}和{q8['energy_gain_vs_ours']:.2f}倍任务能效比（相同工作量的能量反比），能量降低{100*(1-1/q1['energy_gain_vs_ours']):.4f}%和{100*(1-1/q8['energy_gain_vs_ours']):.4f}%。多数基线增大batch后每视频成本下降，但这不是必须单调的理论：CLIP B9高于B8，包含尾批及运行噪声，不能人为平滑。性能曲线近水平源于SRCC变化很小，不是固定分数复制。

## 图b：时间与时间质量性能

LightGen-II每16视频时间估算{ours['time_16_videos_ms']:.4f}ms。新同轮Qwen B1与B8为{q1['time_16_videos_ms']:.4f}与{q8['time_16_videos_ms']:.4f}ms，时间比分别{q1['speedup_vs_ours']:.2f}与{q8['speedup_vs_ours']:.2f}。增大B通常使单调用更长，但完成固定工作量的调用数下降并摊销开销，吞吐改善随后可能饱和或因显存/计算限制退化；不要求所有成本点单调，也不存在同权重准确率随batch上升的理论趋势。

## 全部绘图数据及主交错遥测

数据来源同一轮14 October 2026。以下20点含19电子设置和1光学参考。成本保留6位用于核对，不表示6位有效精度。对应未提前舍入数据为`plot_data.csv`与`timing_energy_recalculated.csv`；后者还保留mean/peak功率、利用率、显存、调用数及SHA。

{paper_table}

## 连续负载：配套时延、能量与GPU利用率

另在同一进程、同一PT/B下，用前B个不同TEST视频已驻GPU的输入重复推理约15秒。此处没有每次CPU准备，成本和util来自连续负载自己的记录，严禁将本表高util配给上一表交错成本。时间折算mean boundary ms×16/B，B=3/9/10为吞吐等效值，不宣称整数次调用刚好处理16条。连续负载未独立计算新的全TEST性能，性能仍来自同PT/B的558视频独立前向。CLIP B9平均80.7617%触发补B10；B10平均89.1913%未达98%，不扫描更大B。YOLO B3平均85.6284%，B4为96.5436%，B8为97.0976%；不能把峰值98%当作平均98%。

{sustained_table}

## 精度变化的诊断证据

同权重精确数学推理不应因batch而改变输出，也无性能随B单调提升或下降的预期。第一步核所有558个样本ID、MOS及每模型抽帧一致；同模型PT与manifest SHA一致。Qwen和DeepSeek另对前8个固定TEST做B1/2/4输入tensor SHA及重复前向。逐视频有效文本、图像、抽帧相同，DeepSeek还核图像crop及图像token mask；原精度重复结果相同，原精度诊断与正式前向最大差0。

{diag_table}

FP32探针同时将部署BF16权重转FP32并使用math SDPA，不恢复权重被量化丢弃的位，也不替代正式结果。DeepSeek保留部署FP16特征取值；math SDPA不控制未调用torch SDPA的算子。Qwen math-only未消除差异，DeepSeek math-only无改变；FP32探针大幅减小差异，支持低精度计算路径是主要贡献，而不是输入改变或随机重训。范围仅8视频，未逐层分解每一算子，不能声称完整唯一因果证明。SRCC基于MOS排序，数值差异可改变秩；近水平曲线不意味着每条MOS完全相同。正式19组未改精度、权重或选模。

## 历史总表锚点与LightGen-II口径

历史Qwen B1仍保留SRCC0.7663428036522781、1188.3861464262009ms/16、462.1134010767563J/16，不写入新同轮曲线，也没有GPUutil记录。新B1为1180.8067452907562ms/16、458.3507566057941J/16，性能相同但成本为独立测量；不能新util与旧成本拼接冒称配套。九任务总表未被本轮修改。

LightGen-II沿此前表格参考：N=16同光场，六周期6×(.714+.300+.0307)=6.2682ms；电子部件3.9773695913702247ms是整幅16槽场计时，不是测一条视频再乘16。router .8302489486336708、CCD后处理2.0079718434810636、bridge .2413414404541254、head .8978073588013649ms。总10.245569591370225ms；E16=.0062682×(80.358+41.388)+.0039773695913702247×(41.388+62.842)=1.1776895097085185J。每视频摊销.0736055943567824J，13.5859238518309videos/J；.6403480994606391ms/video为摊销服务时间而非单视频调用时延。光周期与电子FP32 CUDA-event部件估算不冒称本轮实测精确正式PT光路端到端，不能与baseline同步墙钟说成相同仪器/精度。

## English Methods

We evaluated temporal video-quality prediction on the fixed LGVQ split of 2,250 training and 558 test videos. Each video contributed four frames at relative positions of 10%, 37%, 63% and 90%. All 19 frozen model–batch configurations were evaluated independently on the same 558 test videos. Inference cost was measured on the first 200 videos and normalized to a workload of 16 videos. Partial batches were retained at the 200- and 558-video boundaries. Each configuration ran in a fresh process on the same NVIDIA A100-PCIE-40GB. For each batch, CPU preprocessing and host-to-device transfer preceded inference; these stages were recorded for audit but excluded from the reported costs. Synchronized wall-clock timing began at the first native backbone block and ended after the scalar quality score was available on the GPU. No explicit warm-up was performed, and the first measured call was retained. Energy was estimated as inference duration multiplied by the sum of the 338.2-W host-power proxy and the same-run inference-active mean board power.

GPU utilization, board power and memory use were sampled together at a nominal 50-ms polling interval. Main telemetry statistics used only inference-active windows for the same first 200 videos as the cost subset. GPU utilization denotes the driver-window fraction of time during which kernels were active, not streaming-multiprocessor occupancy. A separate approximately 15-s continuous-load experiment repeatedly processed the first B distinct test inputs already resident on the GPU. Continuous-load latency, energy and utilization are reported together and are not assigned to the interleaved-inference costs. Qwen and DeepSeek retained BF16 backbone execution, CLIP and YOLO retained FP16 execution, and quality heads used FP32. The LightGen-II point retains the physical SRCC reference and the previously approved optical-cycle plus electronic-component cost estimate; it is not a new end-to-end optical measurement.

## English figure description

Temporal quality versus inference cost on LGVQ. (a) Energy and (b) inference time normalized to 16 videos. Electronic points use the complete matched A100 remeasurement of 14 October 2026; B indicates videos per GPU call. SRCC is calculated separately from all 558 test predictions at each batch size. The LightGen-II reference represents 16 videos per optical field (N=16), with physical SRCC 0.7977 and component-estimated costs of 1.1777 J and 10.2456 ms. Larger batches generally reduce amortized electronic cost, but this trend need not be monotonic. The nearly horizontal performance curves do not imply batch-invariant predictions: bounded same-input probes support low-precision computation as a major contributor to the observed batch-dependent differences. Historical measurements, preprocessing costs and continuous-load telemetry are not mixed into these points.

## Reproducibility and evidence

Formal run: `{run.relative_to(REPO).as_posix()}`. Measurement entry SHA28fc58f1; telemetry28ad0666. Manifest SHA607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc. Original Qwen PT e102adc6; reproduced CLIP28729f41/YOLO47606708/DeepSeek3748b08f. Exact per-file SHA and sources are in `recalculation.json` and the canonical run backup receipts. Old refreshed report files are recoverably archived in the same run's `report_before_refresh` with a manifest; all original measurements and failed attempts remain unchanged. The historical side-effecting `build_analysis.py` was not executed. Report refresh is CPU-only and launches no GPU processes.
'''
    (OUT/'README.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(rows=len(rows),plot_points=len(plot),diagnoses=list(diagnostics),output=str(OUT))))

if __name__=='__main__':
    if '--annotate-utilization' in sys.argv:
        annotate_utilization()
    else:
        refresh_paired() if '--paired-telemetry' in sys.argv else main()
