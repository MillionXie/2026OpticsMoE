"""Offline complete-window reconciliation, eight views, no missing-point synthesis."""
from __future__ import annotations
import json
import csv
import statistics
import subprocess
import math
from datetime import datetime,timezone,timedelta
from pathlib import Path
from . import precision_comparison_20261019 as audit

ROOT=audit.TASK/'runs/simulation/lgvq_busy_resource_20261020'
LAB=audit.TASK/'runs/simulation/lgvq_busy_resource_lab_20261020'
NAMES={'qwen':'Qwen3-VL-2B','glm':'GLM-4.6V-Flash','clip':'CLIP ViT-B/32','yolo':'YOLO11s','deepseek':'DeepSeek-VL2','kimi':'Kimi-VL-A3B'}


def backup_audit():
    unique=set();archives=[]
    for receipt in (ROOT/'transfer').glob('*.tar.receipt.json'):
        record=audit.read(receipt);archive=receipt.with_name(receipt.name.replace('.receipt.json','.gz'))
        assert audit.sha(archive)==record['archive_sha256']
        for item in record['files']:
            file=(audit.REPO/item['path']).resolve()
            assert file.is_relative_to(ROOT.resolve()) and audit.sha(file)==item['sha256']
            unique.add(item['path'])
        archives.append(dict(name=archive.name,sha256=record['archive_sha256'],files=len(record['files'])))
    return dict(archives=archives,unique_raw_files=len(unique),all_local_raw_sha_verified=True)


def check(folder,condition):
    identity=audit.read(folder/'identity.json');report=audit.read(folder/condition/'report.json')
    raw=audit.rows(folder/condition/'telemetry.csv');events=audit.read(folder/condition/'groups.json')
    clocks=audit.rows(folder/condition/'clock_telemetry.csv')
    expected_limit=250 if identity['gpu_uuid']=='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55' else 400
    assert clocks and all(abs(float(r['power_limit_w'])-expected_limit)<.1 for r in clocks)
    assert report['status']=='complete' and report['cost_and_utilization_same_window']
    assert report['internal_wait_included'] and report['terminal_wait_excluded'] and len(report['repeats'])==3
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['git_commit']+':'+name.replace('\\','/')],cwd=audit.REPO)
        assert audit.hashlib.sha256(blob).hexdigest()==digest
    for r in report['repeats']:
        assert len(r['output'])==identity['batch'] and all(math.isfinite(v) for v in r['output'])
        elapsed=r['final_gpu_output']-r['first_tf_start'];audit.close(elapsed,r['elapsed_s']);assert elapsed>=30
        rows=[v for v in raw if r['first_tf_start']<=float(v['host_monotonic_s'])<r['final_gpu_output']]
        assert len(rows)==r['measurement']['samples']
        audit.close(statistics.mean(float(v['watts']) for v in rows),r['measurement']['mean_board_power_w'])
        audit.close(statistics.mean(float(v['utilization_percent']) for v in rows),r['measurement']['mean_gpu_utilization_percent'])
        groups=[g for g in events if g['repeat']==r['repeat']]
        assert len(groups)==r['groups'] and all(g['forwards']==8 for g in groups)
        assert groups[0]['start']<=r['first_tf_start']<=groups[0]['compute_end']
        assert groups[-1]['terminal'] and groups[-1]['requested_wait_s']==0
        audit.close(groups[-1]['compute_end'],r['final_gpu_output'])
        assert r['video_evaluations']==len(groups)*8*identity['batch']
        audit.close(elapsed*16000/r['video_evaluations'],r['time16_ms'])
        audit.close((338.2+r['measurement']['mean_board_power_w'])*elapsed*16/r['video_evaluations'],r['energy16_j'])
    assert all(abs(a-b)<=.0002 for r in report['repeats'][1:] for a,b in zip(report['repeats'][0]['output'],r['output']))
    n=sum(r['video_evaluations'] for r in report['repeats']);seconds=sum(r['elapsed_s'] for r in report['repeats'])
    energy=sum((338.2+r['measurement']['mean_board_power_w'])*r['elapsed_s'] for r in report['repeats'])
    audit.close(seconds*16000/n,report['time16_ms']);audit.close(energy*16/n,report['energy16_j'])
    samples=sum(r['measurement']['samples'] for r in report['repeats'])
    busy=sum(r['measurement']['mean_gpu_utilization_percent']*r['measurement']['samples'] for r in report['repeats'])/samples
    audit.close(busy,report['mean_gpu_busy_percent'])
    if identity['method']=='mps' and identity['cap']==50:assert .4<=identity['visible_sm_ratio_to100']<=.55
    return dict(model=identity['model'],model_name=NAMES[identity['model']],method=identity['method'],precision=identity['precision'],
        condition=condition,batch=identity['batch'],resource_cap=identity['cap'] if identity['method']=='mps' else None,
        visible_sms=identity['visible_multiprocessors'],busy_percent=busy,time16_ms=report['time16_ms'],energy16_j=report['energy16_j'],
        measured_at=datetime.fromtimestamp(float(next(v for v in raw if float(v['host_monotonic_s'])>=report['repeats'][0]['first_tf_start'])['host_time_unix']),timezone(timedelta(hours=8))).isoformat(),
        repeats=report['repeats'],checkpoint_sha256=identity['checkpoint_sha256'],
        assets_sha256=audit.hashlib.sha256(json.dumps(identity['assets'],sort_keys=True).encode()).hexdigest(),
        source=(folder/condition/'report.json').relative_to(audit.REPO).as_posix(),source_sha256=audit.sha(folder/condition/'report.json'),
        platform='A100-PCIE-40GB /250W' if identity['gpu_uuid']=='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55' else 'A100-SXM4-80GB /400W',status='measured' if report['target50_reached'] is not False else 'outside50±5pp',
        srcc=None,accuracy_note='No full558 accuracy evaluation in this repeating cost window; do not substitute another precision SRCC.')


def main():
    values=[]
    for root in (ROOT,LAB):
        if not root.exists():continue
        for folder in root.iterdir():
            if not folder.is_dir() or not(folder/'identity.json').exists():continue
            if not(folder/'status.json').exists() or audit.read(folder/'status.json')['status']!='complete':continue
            for condition in ('mps_cap100','mps_cap50','unrestricted','busy50'):
                if (folder/condition/'report.json').exists():values.append(check(folder,condition))
    keys=[(r['model'],r['method'],r['precision'],r['condition']) for r in values]
    if len(keys)!=len(set(keys)):raise RuntimeError('Ambiguous duplicate comparison point; do not select a run implicitly')
    old=audit.read(audit.OLD);optical=old['groups']['busy50_near_saturation'][0]
    anchor=next(r for r in old['groups']['batch1_near_saturation'] if r['baseline']=='qwen' and r.get('batch')==1)
    for key,expected in dict(srcc=.7663428036522781,utilization_percent=63.133111480865225,time_ms=1188.3861464262009,energy_j=462.1134010767563).items():audit.close(anchor[key],expected)
    # Independent accuracy references are explicit, never promoted to MPS measurements.
    refs={}
    for group in ('batch1_near_saturation','busy50_near_saturation'):
        for row in old['groups'].get(group,[]):
            if row.get('batch') and row.get('srcc') is not None:
                key='deepseek' if row['baseline']=='full' else row['baseline']
                refs[(key,row['batch'])]=dict(srcc=row['srcc'],source=row.get('accuracy_source') or row.get('cost_source'))
    for row in values:
        if row['precision']=='native' and (row['model'],row['batch']) in refs:
            ref=refs[(row['model'],row['batch'])];row.update(srcc_reference=ref['srcc'],accuracy_source=ref['source'],
                accuracy_note='Independent frozen native full558 reference at this batch; not a new MPS/restricted-condition accuracy evaluation.')
            if row['model'] in ('deepseek','kimi'):
                file=audit.REPO/ref['source'];reference=audit.read(file)
                assert reference['status']=='complete' and reference['test_videos']==558
                assert reference['identity']['checkpoint_sha256']==row['checkpoint_sha256']
                audit.close(reference['performance']['srcc'],row['srcc_reference'])
                row['accuracy_source_sha256']=audit.sha(file)
    accuracy=ROOT/'accuracy_kimi_fp32_batch1'
    if (accuracy/'status.json').exists() and audit.read(accuracy/'status.json')['status']=='complete':
        import numpy as np
        from scipy.stats import spearmanr,kendalltau,pearsonr
        identity=audit.read(accuracy/'identity.json');file=accuracy/'evaluation/report.json';report=audit.read(file)
        predictions=audit.rows(accuracy/'evaluation/predictions.csv')
        assert identity['precision']=='fp32' and identity['batch']==1 and report['test_videos']==558 and len(predictions)==558
        assert [int(r['sample_index']) for r in predictions]==list(range(558))
        target=np.array([float(r['target_mos']) for r in predictions]);prediction=np.array([float(r['prediction']) for r in predictions])
        assert np.isfinite(target).all() and np.isfinite(prediction).all()
        difference=prediction-target
        metrics=dict(srcc=float(spearmanr(target,prediction).statistic),krcc=float(kendalltau(target,prediction).statistic),
            plcc=float(pearsonr(target,prediction).statistic),rmse=float(np.sqrt(np.mean(np.square(difference)))),mae=float(np.mean(np.abs(difference))))
        for key,value in metrics.items():audit.close(value,report['performance'][key])
        for row in values:
            if row['model']=='kimi' and row['precision']=='fp32' and row['batch']==1:
                assert row['checkpoint_sha256']==identity['checkpoint_sha256']
                assert row['assets_sha256']==audit.hashlib.sha256(json.dumps(identity['assets'],sort_keys=True).encode()).hexdigest()
                row.update(srcc_reference=report['performance']['srcc'],accuracy_source=file.relative_to(audit.REPO).as_posix(),
                    accuracy_source_sha256=audit.sha(file),accuracy_note='Independent frozen FP32 full558 B1 evaluation; not a fresh per-MPS-cap accuracy test.')
    target=ROOT/'presentation';target.mkdir(parents=True,exist_ok=True)
    data=dict(status='partial',values=values,optical=optical,models=NAMES,
              contract='Two controls × two precisions × time/energy; full same window includes internal waits, excludes terminal wait; first-B inputs reused, real inference every call; three full windows, all retained.',
              capacity_missing='DeepSeek-VL2 FP32 weights require 109,920,536,992 bytes (102.4 GiB), exceeding a single 80 GiB device. MPS does not reduce weight storage. No CPU offload.',
              B1_historical_anchor_unchanged=True,accuracy_reference_not_per_control=True,
              rental_backup_audit=backup_audit())
    (target/'eight_tables.json').write_text(json.dumps(data,indent=2)+'\n')
    columns=['model','method','precision','condition','batch','resource_cap','visible_sms','busy_percent','time16_ms','energy16_j','srcc_reference','platform','measured_at','status','source','source_sha256','accuracy_source','accuracy_note']
    with (target/'paired_measurements.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(values)
    with (target/'paired_repeats.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['model','method','precision','condition','batch','resource_cap','visible_sms','repeat','elapsed_s','video_evaluations','mean_board_power_w','gpu_busy_percent','samples','first_tf_start','final_gpu_output','time16_ms','energy16_j','source','source_sha256']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for point in values:
            for r in point['repeats']:
                writer.writerow({**point,**r,'mean_board_power_w':r['measurement']['mean_board_power_w'],
                    'gpu_busy_percent':r['measurement']['mean_gpu_utilization_percent'],'samples':r['measurement']['samples']})
    print(json.dumps(dict(audited_points=len(values),models=[(x['model'],x['method'],x['precision'],x['condition']) for x in values])))


if __name__=='__main__':main()
