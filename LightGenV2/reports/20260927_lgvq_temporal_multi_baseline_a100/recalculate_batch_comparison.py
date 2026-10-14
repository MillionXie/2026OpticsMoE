"""Offline rederivation; never imports the historical side-effecting builder.

No weights, CUDA, optical SDK or video decoding are used. Raw files stay intact.
"""
from __future__ import annotations
import csv
import hashlib
import json
import statistics
from collections import Counter
from math import ceil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr

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
    return float(spearmanr([float(r['target_mos']) for r in rows],
                          [float(r['prediction']) for r in rows]).statistic)

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
        candidates=[(0,dy) for dy in (10,-10,20,-20,30,-30)]
        candidates += [(dx,dy) for dy in (10,-10,20,-20,30,-30) for dx in (-12,12,-24,24)]
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

if __name__=='__main__': main()
