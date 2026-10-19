"""Offline complete-window reconciliation, eight views, no missing-point synthesis."""
from __future__ import annotations
import json
import statistics
import subprocess
from pathlib import Path
from . import precision_comparison_20261019 as audit

ROOT=audit.TASK/'runs/simulation/lgvq_busy_resource_20261020'
NAMES={'qwen':'Qwen3-VL-2B','glm':'GLM-4.6V-Flash','clip':'CLIP ViT-B/32','yolo':'YOLO11s','deepseek':'DeepSeek-VL2','kimi':'Kimi-VL-A3B'}


def check(folder,condition):
    identity=audit.read(folder/'identity.json');report=audit.read(folder/condition/'report.json')
    raw=audit.rows(folder/condition/'telemetry.csv');events=audit.read(folder/condition/'groups.json')
    assert report['status']=='complete' and report['cost_and_utilization_same_window']
    assert report['internal_wait_included'] and report['terminal_wait_excluded'] and len(report['repeats'])==3
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['git_commit']+':'+name.replace('\\','/')],cwd=audit.REPO)
        assert audit.hashlib.sha256(blob).hexdigest()==digest
    for r in report['repeats']:
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
        repeats=report['repeats'],checkpoint_sha256=identity['checkpoint_sha256'],
        source=(folder/condition/'report.json').relative_to(audit.REPO).as_posix(),source_sha256=audit.sha(folder/condition/'report.json'),
        platform='A100-SXM4-80GB /400W',status='measured' if report['target50_reached'] is not False else 'outside50±5pp',
        srcc=None,accuracy_note='No full558 accuracy evaluation in this repeating cost window; do not substitute another precision SRCC.')


def main():
    values=[]
    if ROOT.exists():
        for folder in ROOT.iterdir():
            if not folder.is_dir() or not(folder/'identity.json').exists():continue
            if not(folder/'status.json').exists() or audit.read(folder/'status.json')['status']!='complete':continue
            for condition in ('mps_cap100','mps_cap50','unrestricted','busy50'):
                if (folder/condition/'report.json').exists():values.append(check(folder,condition))
    old=audit.read(audit.OLD);optical=old['groups']['busy50_near_saturation'][0]
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
    target=ROOT/'presentation';target.mkdir(parents=True,exist_ok=True)
    data=dict(status='partial',values=values,optical=optical,models=NAMES,
              contract='Two controls × two precisions × time/energy; full same window includes internal waits, excludes terminal wait; first-B inputs reused, real inference every call; three full windows, all retained.',
              capacity_missing='DeepSeek full FP32 needs109920536992 weight bytes, beyond80GB even under50% MPS; no offload.',
              B1_historical_anchor_unchanged=True,accuracy_reference_not_per_control=True)
    (target/'eight_tables.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(audited_points=len(values),models=[(x['model'],x['method'],x['precision'],x['condition']) for x in values])))


if __name__=='__main__':main()
