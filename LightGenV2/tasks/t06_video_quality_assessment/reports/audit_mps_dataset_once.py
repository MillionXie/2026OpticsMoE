"""Independent cold dataset-once replay; never relabel resident observations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from . import strict_tf_resource_audit_20261020 as old

TASK=Path(__file__).resolve().parents[1]
REPO=TASK.parents[2]
ROOT=TASK/'runs/simulation/lgvq_mps_dataset_once_a100_20261020'
_sources={}


def rows(path):return list(csv.DictReader(path.open(newline='',encoding='utf-8-sig')))
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-7,rel_tol=1e-10),(a,b)


def audit_point(folder):
    from scipy.stats import spearmanr,pearsonr
    report=read(folder/'report.json');identity=read(folder/'identity.json');calls=rows(folder/'calls.csv')
    predictions=rows(folder/'predictions.csv');raw=rows(folder/'telemetry.csv')
    assert report['status']=='complete' and identity['batch']==1
    assert identity['protocol']['timing_protocol']=='dataset_once_legacy_block_plus_nonpreprocessing_gaps_v1'
    assert identity['protocol']['explicit_warmup_forwards']==identity['explicit_warmup_forwards']==0
    assert identity['first_test_batch_included'] and identity['timer_start_placement'].startswith('legacy first-block host entry')
    assert len(calls)==len(predictions)==558
    manifest=ROOT.parent/'lgvq_strict_tf_rental_dual_20261020/transfer/lgvq_train2250_test558.csv'
    assert sha(manifest)==identity['manifest_sha256']=='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
    expected=[r for r in rows(manifest) if r['split']=='test']
    assert [r['sample_id'] for r in calls]==[r['sample_id'] for r in predictions]==[r['sample_id'] for r in expected]
    assert [int(r['sample_index']) for r in calls]==list(range(558))
    assert len({r['sample_id'] for r in calls[:200]})==200
    assert identity['visible_multiprocessors']==(108 if identity['cap']==100 else 54)
    for path,digest in identity['sources'].items():
        key=(identity['git_commit'],path)
        if key not in _sources:
            value=subprocess.check_output(['git','show',key[0]+':'+path.replace('\\','/')],cwd=REPO)
            _sources[key]=hashlib.sha256(value).hexdigest()
        assert _sources[key]==digest
    observer=read(folder/'telemetry_observer_identity.json')
    assert observer['cuda_context_created'] is False and observer['version']=='independent_process_receipt_clock_v1'
    events=[json.loads(s) for s in (folder/'telemetry_stdout.jsonl').read_text().splitlines()]
    assert len(events)==len(raw) and all(r['valid_numeric_row'] for r in events)
    intervals=[];compute=gap=0.
    for i,c in enumerate(calls):
        p,a,s,e=(float(c[k]) for k in ('preprocess_start','model_call_start','first_block_start','gpu_output'))
        assert p<=a<=s<e<=float(c['call_return'])+1e-5
        assert (c['in_cost_subset']=='True') is (i<200)
        close((e-s)*1000,c['legacy_host_ms'])
        close(s-a,c['excluded_model_preblock_s'])
        assert json.loads(c['frame_positions'])==json.loads(predictions[i]['frame_positions'])
        assert len(json.loads(c['frame_positions']))==4
        close(predictions[i]['target_mos'],expected[i]['temporal'])
        if i<200:
            intervals.append((s,e));compute+=e-s
            if i:
                previous=float(calls[i-1]['gpu_output']);assert previous<=p
                intervals.append((previous,p));gap+=p-previous
    intervals=sorted((a,b) for a,b in intervals if a<b)
    assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
    charged=compute+gap;board=old.observed_energy(intervals,raw);energy=338.2*charged+board
    for k,v in dict(charged_s=charged,compute_s=compute,nonpreprocessing_gap_s=gap,
                    time16_ms=charged*80,energy16_j=energy*.08,legacy_compute_only_time16_ms=compute*80).items():
        close(report['cost'][k],v)
    y=[float(r['target_mos']) for r in predictions];pred=[float(r['prediction']) for r in predictions]
    close(report['performance']['srcc'],spearmanr(y,pred).statistic)
    close(report['performance']['plcc'],pearsonr(y,pred).statistic)
    identity_value=read(folder/'identity.json');assert identity_value['checkpoint_sha256']==report['checkpoint_sha256']
    for event in read(folder/'process_audit.json'):
        allowed={str(identity['pid'])}|{str(p) for p in event['servers']}
        assert all(str(r[0]).strip() in allowed for r in event['processes'])
        assert all(str(r[1]).strip()==identity['gpu_uuid'] for r in event['processes'])
    return dict(model=identity['model'],resource_cap=identity['cap'],repeat=identity['repeat'],batch=1,
                srcc=report['performance']['srcc'],plcc=report['performance']['plcc'],
                time16_ms=charged*80,energy16_j=energy*.08,compute_only_time16_ms=compute*80,
                nonpreprocessing_gap16_ms=gap*80,charged_s=charged,gpu_energy_j=board,
                video_evaluations=200,visible_sms=identity['visible_multiprocessors'],
                actual_busy_percent=report['full_wall_telemetry']['mean_gpu_utilization_percent'],
                checkpoint_sha256=identity['checkpoint_sha256'],parameter_dtypes=identity['floating_parameter_dtypes'],
                readout_dtypes=identity['quality_readout_dtypes'],source=(folder/'report.json').relative_to(REPO).as_posix(),
                source_sha256=sha(folder/'report.json'),identity_sha256=sha(folder/'identity.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    values=[]
    for model in ('qwen','glm','clip','yolo'):
        for cap in (100,50):
            for repeat in (1,2,3):
                folder=ROOT/f'{model}_cap{cap}_repeat{repeat:02d}'
                if not (folder/'report.json').exists():
                    if a.partial:continue
                    raise RuntimeError('Missing '+str(folder))
                values.append(audit_point(folder));print('REPLAY_PASS',folder.name,flush=True)
    points=[]
    for model in ('qwen','glm','clip','yolo'):
        for cap in (100,50):
            rr=[r for r in values if (r['model'],r['resource_cap'])==(model,cap)]
            if len(rr)!=3:continue
            assert len({r['checkpoint_sha256'] for r in rr})==1
            assert len({tuple(r['parameter_dtypes']) for r in rr})==1
            n=sum(r['video_evaluations'] for r in rr)
            points.append(dict(model=model,resource_cap=cap,batch=1,
                               time16_ms=sum(r['charged_s'] for r in rr)*16000/n,
                               energy16_j=sum(338.2*r['charged_s']+r['gpu_energy_j'] for r in rr)*16/n,
                               srcc=statistics.mean(r['srcc'] for r in rr),
                               srcc_repeats=[r['srcc'] for r in rr],repeats=rr))
    folder=ROOT/'presentation';folder.mkdir(exist_ok=True)
    (folder/'dataset_once_audited.json').write_text(json.dumps(dict(
        status='complete' if len(points)==8 else 'partial',points=points,repeats=values,
        historical_Qwen_B1_unchanged=True,method='three fresh processes; ordered558; first200 cost; no warmup'),indent=2)+'\n')
    print(json.dumps(dict(points=len(points),repeats=len(values),status='complete' if len(points)==8 else 'partial')))


if __name__=='__main__':main()
