"""Offline raw-evidence replay and four precision-specific panels, never infer gaps."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import shutil

REPO=Path(__file__).resolve().parents[4]
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
RENTAL=TASK/'runs/simulation/lgvq_rental_precision_20261019'
LAB=TASK/'runs/simulation/lgvq_fp32_a100_20261019'
OUT=REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
OLD=TASK/'runs/simulation/lgvq_unified_inference_recheck_20261018/presentation/compact_groups_20261018.json'


def read(path):return json.loads(path.read_text())
def rows(path):return list(csv.DictReader(path.open(newline='',encoding='utf-8-sig')))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def local(path):
    for prefix in ('/data/2026OpticsMoE/','/DATA/DATA1/guest3/2026OpticsMoE/'):
        if path.startswith(prefix):return REPO/path[len(prefix):]
    raise ValueError('Unknown source root')


def audit_util(folder):
    report=read(folder/'report.json');raw=rows(folder/'telemetry.csv');groups=read(folder/'forward_groups.json')
    assert report['status']=='complete' and len(report['repeats'])==3
    events=[json.loads(line) for line in (folder/'telemetry_stdout.jsonl').read_text().splitlines()]
    assert len(events)==len(raw) and all(v['valid_numeric_row'] for v in events)
    assert all(0<=float(r['utilization_percent'])<=100 and float(r['watts'])>0 for r in raw)
    total=0;count=0;means=[]
    for index,r in enumerate(report['repeats'],1):
        start,end=r['start_monotonic_s'],r['end_monotonic_s'];assert end-start>=30 and r['warmup_s']>=5
        selected=[s for s in raw if start<=float(s['host_monotonic_s'])<end]
        assert len(selected)==r['measurement']['samples']
        mean=statistics.mean(float(s['utilization_percent']) for s in selected)
        close(mean,r['measurement']['mean_gpu_utilization_percent']);total+=sum(float(s['utilization_percent']) for s in selected);count+=len(selected);means.append(mean)
        measured=[g for g in groups if g['phase']==f'measured{index}']
        assert measured and all(g['forwards']==8 and g['duty']==report['frozen_duty'] for g in measured)
        for g in measured:
            expected=(g['compute_end_monotonic_s']-g['start_monotonic_s'])*(1/g['duty']-1) if report['paced'] else 0
            close(expected,g['requested_idle_s'])
    close(total/count,report['mean_gpu_busy_percent'])
    return report,means


def audit_cost(folder):
    from scipy.stats import spearmanr,pearsonr
    report=read(folder/'evaluation/report.json');identity=read(folder/'identity.json')
    assert report['status']=='complete' and report['test_videos']==558 and report['cost_videos']==200
    assert report['explicit_warmup_forwards']==0 and report['first_batch_included']
    assert 'inter-inference imposed waits' in report['excluded']
    prediction=rows(folder/'evaluation/predictions.csv');timing=rows(folder/'evaluation/batch_timing.csv')
    raw=rows(folder/'evaluation/telemetry.csv');waits=read(folder/'evaluation/excluded_waits.json')
    expected=rows(TASK/'runs/simulation/lgvq_temporal_reproduction_a100_20261013/lgvq_train2250_test558.csv')
    expected=[r for r in expected if r['split']=='test']
    assert len(prediction)==558 and [r['sample_id'] for r in prediction]==[r['sample_id'] for r in expected]
    assert [int(r['sample_index']) for r in prediction]==list(range(558))
    assert [float(r['target_mos']) for r in prediction]==[float(r['temporal']) for r in expected]
    assert sum(int(r['videos']) for r in timing)==558
    subset=[r for r in timing if r['in_cost_subset']=='True'];assert sum(int(r['videos']) for r in subset)==200
    expected_sizes=[min(identity['batch'],end-i) for start,end in ((0,200),(200,558)) for i in range(start,end,identity['batch'])]
    assert [int(r['videos']) for r in timing]==expected_sizes
    active=[r for r in raw if r['phase'].startswith('active:')];assert active
    board=statistics.mean(float(r['watts']) for r in active)
    wall=sum(float(r['synchronized_wall_ms']) for r in subset);t16=wall*16/200;e16=(338.2+board)*t16/1000
    close(t16,report['cost']['latency_16_videos_ms']);close(e16,report['cost']['energy_16_videos_j'])
    target=[float(r['target_mos']) for r in prediction];pred=[float(r['prediction']) for r in prediction]
    srcc=float(spearmanr(target,pred).statistic);plcc=float(pearsonr(target,pred).statistic)
    close(srcc,report['performance']['srcc']);close(plcc,report['performance']['plcc'])
    utilpath=local(report['independent_utilization_source']);util,repeats=audit_util(utilpath.parent)
    assert util['model']==identity['model'] and util['batch']==identity['batch'] and util['precision']==identity['precision']
    assert len(waits)==len(timing)
    for wait,t in zip(waits,timing):close(wait['requested_s'],float(t['synchronized_wall_ms'])/1000*(1/report['frozen_duty']-1))
    for path,digest in identity.get('sources',{}).items():
        blob=subprocess.check_output(['git','show',identity['git_commit']+':'+path.replace('\\','/')],cwd=REPO)
        assert hashlib.sha256(blob).hexdigest()==digest,path
    owned=None
    for event in read(folder/'process_audit.json'):
        ids={str(r[0]).strip() for r in event['processes']}
        if not ids and owned is None:continue  # CPU-only admission before CUDA.
        if owned is None:
            assert len(ids)==1,'Cannot bind sole CUDA process';owned=ids
        assert ids==owned,'Foreign/missing/replaced GPU process'
    assert owned is not None
    return dict(model_key=identity['model'],batch=identity['batch'],precision=identity['precision'],
        condition='Approximately 50%' if util['paced'] else 'Near-saturation reference',srcc=srcc,plcc=plcc,
        utilization_percent=util['mean_gpu_busy_percent'],util_repeats=repeats,time_ms=t16,energy_j=e16,
        board_w=board,host_proxy_w=338.2,cost_wall_sum_ms=wall,cost_video_count=200,active_samples=len(active),
        excluded_wait_16_ms=sum(w['end']-w['start'] for w in waits if w['in_cost_subset'])*80,
        source=folder.relative_to(REPO).as_posix(),cost_source=(folder/'evaluation/report.json').relative_to(REPO).as_posix(),
        utilization_source=utilpath.relative_to(REPO).as_posix(),checkpoint_sha256=identity['checkpoint_sha256'],
        gpu_uuid=identity['gpu_uuid'],platform='A100-SXM4-80GB /400W' if folder.is_relative_to(RENTAL) else 'A100-PCIE-40GB /250W',
        no_implicit_resident_cost_substitution=True,source_commit=identity['git_commit'])


def build():
    old=read(OLD);native=[dict(r,precision='optical-reference' if r['baseline']=='ours' else 'native') for r in old['groups']['busy50_near_saturation'] if r['baseline'] in ('ours','qwen','glm','clip','yolo')]
    # Recheck original-platform costs without changing any accepted old observation.
    for r in native:
        if r['baseline']=='ours':continue
        if r['baseline']=='clip' and r['condition']=='Near-saturation reference':
            assert len(r['repeat_inputs'])==3
            close(r['time_ms'],statistics.mean(x['time_ms'] for x in r['repeat_inputs']))
            close(r['energy_j'],statistics.mean(x['energy_j'] for x in r['repeat_inputs']))
            r['host_proxy_w']=338.2
            continue
        source=REPO/r['cost_source'];folder=source.parent
        t=rows(folder/'batch_timing.csv');raw=rows(folder/'telemetry.csv')
        cost=[v for v in t if v['in_cost_subset']=='True'];active=[v for v in raw if v['phase'].startswith('active:')]
        assert sum(int(v['videos']) for v in cost)==200 and active
        wall=sum(float(v['synchronized_wall_ms']) for v in cost);watts=statistics.mean(float(v['watts']) for v in active)
        close(wall*16/200,r['time_ms']);close((338.2+watts)*wall*16/200/1000,r['energy_j'])
        r.update(cost_wall_sum_ms=wall,cost_video_count=200,board_w=watts,host_proxy_w=338.2,active_samples=len(active))
    templates={r['baseline']:r for r in old['groups']['busy50_near_saturation'] if r['condition']=='Near-saturation reference'}
    audited=[];missing=[];fp32=[dict(native[0],precision='optical-reference')]
    for root in (RENTAL,LAB):
        if not root.exists():continue
        for folder in root.iterdir():
            if not folder.is_dir() or not(folder/'status.json').exists():continue
            status=read(folder/'status.json')
            if status['status']!='complete' or not(folder/'evaluation/report.json').exists():continue
            value=audit_cost(folder);audited.append(value)
            key='full' if value['model_key']=='deepseek' else value['model_key']
            template=templates[key]
            assert value['checkpoint_sha256']==template['checkpoint_sha256'],'Frozen comparison head mismatch'
            new=dict(template,**{k:v for k,v in value.items() if k!='model_key'})
            new.update(plotted=True,repeats=value['util_repeats'],link_style='solid',accuracy_source=value['cost_source'])
            (native if value['precision']=='native' else fp32).append(new)
    for precision,values in [('native',native),('fp32',fp32)]:
        for model in ('qwen','glm','clip','yolo','full','kimi'):
            for condition in ('Approximately 50%','Near-saturation reference'):
                if not any(r['baseline']==model and r['condition']==condition for r in values):missing.append(dict(precision=precision,model=model,condition=condition))
    for r in native+fp32:
        if r['baseline']=='ours':continue
        if r['condition']=='Approximately 50%' and abs(r['utilization_percent']-50)>5:
            r.update(plotted=False,target50_failed=True)
            missing.append(dict(precision=r['precision'],model=r['baseline'],condition=r['condition'],reason='Measured independent utilization outside accepted50±5pp; raw evidence retained'))
    target=RENTAL/'presentation';target.mkdir(parents=True,exist_ok=True)
    payload=dict(status='complete' if not missing else 'partial',groups={'native':native,'fp32':fp32},audited_new_points=audited,missing=missing,
        Qwen_B1_unchanged=True,pacing_waits_excluded=True,utilization_independent=True,all_completed_repeats_retained=True)
    (target/'precision_comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(status=payload['status'],new_points=len(audited),missing=len(missing))))


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from PIL import Image
    data=read(RENTAL/'presentation/precision_comparison.json')
    archive=RENTAL/'presentation/prior_report_archive';archive.mkdir(parents=True,exist_ok=True)
    manifest=archive/'manifest.json'
    if not manifest.exists():
        saved={}
        names=['README.md','LGVQ_busy50_near_saturation_plot_data.xlsx','busy50_near_saturation_pair.png']
        names += [f'busy50_near_saturation_{kind}.{ext}' for kind in ('latency','energy') for ext in ('png','svg','pdf')]
        for name in names:
            target=archive/name
            if not target.exists():shutil.copy2(OUT/name,target)
            saved[name]=sha(target)
        manifest.write_text(json.dumps(saved,indent=2)+'\n')
    for name,digest in read(manifest).items():assert sha(archive/name)==digest
    from .plot_selected_execution import COMPACT_COLORS,COMPACT_MARKERS
    colors=COMPACT_COLORS;markers=COMPACT_MARKERS
    plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.6})
    geometry=[]
    for precision,values in data['groups'].items():
        prefix='busy50_near_saturation' if precision=='native' else 'fp32_busy50_near_saturation'
        values=[r for r in values if r.get('plotted',True)]
        for metric,field,label in [('latency','time_ms','Inference latency (ms / 16 videos)'),('energy','energy_j','Estimated inference energy (J / 16 videos)')]:
            fig,ax=plt.subplots(figsize=(78/25.4,88/25.4));fig.subplots_adjust(left=.19,right=.97,top=.92,bottom=.34)
            ax.set_xscale('log');ax.set_xlim(min(r[field] for r in values)*.55,max(r[field] for r in values)*1.85)
            low=min(r['srcc'] for r in values);high=max(r['srcc'] for r in values)
            ax.set_ylim(min(.68,low-.025),max(.84,high+.025));ax.set_ylabel('Temporal quality (SRCC)');ax.set_xlabel(label)
            ax.set_title('Deployed precision' if precision=='native' else 'FP32 execution',fontsize=8)
            ax.grid(color='#DEE2E6',lw=.4,which='major');ax.spines[['top','right']].set_visible(False)
            annotations=[];handles=[]
            for model,color in colors.items():
                points=[r for r in values if r['baseline']==model]
                if not points:continue
                handles.append(Line2D([],[],marker=markers[model],color=color,lw=.7,label=points[0]['model'],markersize=3.5))
                if len(points)>1:ax.plot([r[field] for r in points],[r['srcc'] for r in points],color=color,lw=.8,zorder=2)
                for r in points:
                    paced=r['condition']=='Approximately 50%'
                    ax.scatter(r[field],r['srcc'],marker=markers[model],s=45 if model=='ours' else 18,
                        facecolor='white' if paced else color,edgecolor=color,lw=.8,zorder=3)
                    text='~50% spatial' if model=='ours' else f"{r['utilization_percent']:.1f}%"
                    ann=ax.annotate(text,(r[field],r['srcc']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=6.2,color=color,
                        bbox=dict(facecolor='white',edgecolor='none',pad=.5),arrowprops=dict(arrowstyle='-',lw=.4,color=color))
                    annotations.append(ann)
            # Lay labels on a fine axes grid, avoiding points, lines and labels.
            fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=ax.get_window_extent(renderer);placed=[];leaders=[]
            ax._audited_points=[(r[field],r['srcc']) for r in values]
            pointxy=[ax.transData.transform((r[field],r['srcc'])) for r in values]
            from matplotlib.path import Path as MP
            lines=[MP(line.get_transform().transform(line.get_xydata())) for line in ax.lines]
            for ann in annotations:
                box=ann.get_bbox_patch().get_window_extent(renderer);anchor=ax.transData.transform(ann.xy)
                choices=[(x,y) for x in np.linspace(.07,.93,35) for y in np.linspace(.05,.95,36)]
                choices.sort(key=lambda p:np.linalg.norm(ax.transAxes.transform(p)-anchor))
                origin=ann.get_transform().transform(ann.get_position())
                for choice in choices:
                    new=ax.transAxes.transform(choice);candidate=box.translated(*(new-origin))
                    if not bounds.contains(candidate.x0,candidate.y0) or not bounds.contains(candidate.x1,candidate.y1):continue
                    if any(candidate.padded(1).overlaps(b) for b in placed):continue
                    if any(candidate.padded(2).contains(*p) for p in pointxy):continue
                    if any(line.intersects_bbox(candidate.padded(1),filled=False) for line in lines):continue
                    if any(line.intersects_bbox(candidate.padded(.5),filled=False) for line in leaders):continue
                    leader=MP([anchor,new])
                    if any(leader.intersects_bbox(b.padded(.5),filled=False) for b in placed):continue
                    ann.set_anncoords('axes fraction');ann.set_position(choice);placed.append(candidate);leaders.append(leader);break
                else:raise RuntimeError('Label packing failed')
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.075),ncol=2,frameon=False,fontsize=6,handlelength=1.2,columnspacing=.7)
            count=sum(1 for r in data['missing'] if r['precision']==precision)
            note='Independent utilization; pacing waits excluded.'
            if precision=='fp32':note+='\nFP32 expanded weights; backend/platform differences disclosed.'
            if count:note+=f'\n{count} requested endpoint(s) unavailable; no interpolation.'
            fig.text(.19,.008,note,fontsize=5,color='#53616B')
            from .plot_selected_execution import verify_label_geometry
            geometry.extend(verify_label_geometry(fig,[ax],check_lines=True))
            for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{prefix}_{metric}.{ext}',dpi=600)
            plt.close(fig)
        a=Image.open(OUT/f'{prefix}_latency.png');b=Image.open(OUT/f'{prefix}_energy.png')
        joined=Image.new('RGB',(a.width+b.width,a.height),'white');joined.paste(a,(0,0));joined.paste(b,(a.width,0));joined.save(OUT/f'{prefix}_pair.png',dpi=(600,600))
    (RENTAL/'presentation/figure_geometry.json').write_text(json.dumps(dict(status='passed',panels=geometry),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');a=p.parse_args()
    render() if a.render else build()
