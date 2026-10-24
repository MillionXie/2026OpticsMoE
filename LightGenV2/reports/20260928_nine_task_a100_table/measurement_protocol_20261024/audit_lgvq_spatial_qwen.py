"""CPU replay of the unchanged spatial baseline; no inference or cost override."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
from measure_lgvq_spatial_qwen import PIN,REFERENCE_SHA,GPU
from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry


def close(a,b):
    if not math.isclose(float(a),float(b),abs_tol=1e-6,rel_tol=1e-9):
        raise ValueError('Independent spatial replay mismatch')


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def audit(folder,manifest):
    from scipy.stats import spearmanr,pearsonr
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    folder=Path(folder);report=entry.read(folder/'report.json');state=entry.read(folder/'status.json')
    expected=entry.input_rows(manifest);calls=csv_rows(folder/'calls.csv');raw=csv_rows(folder/'telemetry.csv')
    if (report['status']!='complete' or state['status']!='complete' or state['samples']!=200
            or report['checkpoint_sha256']!=PIN or report['reference_sha256']!=REFERENCE_SHA
            or report['gpu_uuid']!=GPU or report['mps_percent']!=100
            or report['explicit_warmup_forwards']!=0 or report['batch']!=1
            or report['visible_multiprocessors']!=108 or abs(report['power_limit_w']-250)>.1):
        raise ValueError('Wrong completion/model/resource/protocol identity')
    for record in entry.read(folder/'files_manifest.json'):
        p=folder/record['file']
        if entry.sha(p)!=record['sha256'] or p.stat().st_size!=record['bytes']:raise ValueError('Raw SHA/size changed')
    if [r['sample_id'] for r in calls]!=[r['sample_id'] for r in expected]:raise ValueError('Wrong200 input order')
    if [float(r['target_mos']) for r in calls]!=[float(r['spatial']) for r in expected]:raise ValueError('Wrong spatial targets')
    if report['manifest_sha256']!=entry.sha(manifest):raise ValueError('Wrong manifest')
    evidence=entry.read(folder/'input_audit.json')
    if entry.sha(folder/'input_audit.json')!=report['input_audit_sha256']:raise ValueError('Input audit SHA')
    if [r['sample_id'] for r in evidence['samples']]!=[r['sample_id'] for r in expected]:raise ValueError('Incomplete input evidence')
    for r in evidence['samples']:
        if r['tensors']['input_ids']['shape'][0]!=1 or r['small_cpu_values']['video_grid_thw']!=[[2,28,28]]:
            raise ValueError('Not four original448x448 frames')
        mask=r['small_cpu_values']['attention_mask'][0]
        if sum(mask)!=r['effective_input_tokens']:raise ValueError('Effective token denominator')
    if sum(r['effective_input_tokens'] for r in evidence['samples'])!=report['effective_input_tokens200']:raise ValueError('Common token total')
    if set(report['input_files'])!={r['sample_id'] for r in expected}:raise ValueError('Raw input SHA coverage')
    samples=[Sample(float(r['host_time_unix']),float(r['host_monotonic_s']),r['phase'],float(r['watts']),
                    float(r['utilization_percent']),float(r['memory_used_mib'])) for r in raw]
    value=entry.summarize(calls,samples)
    for key in ('continuous_wall_s','ms_per_video','equivalent_ms_per_input_frame','estimated_joules_per_video',
                'estimated_joules_per_input_frame','gpu_energy_j'):close(value[key],report['cost'][key])
    close(observed_energy([(float(calls[0]['preprocess_start']),float(calls[-1]['call_return']))],raw),value['gpu_energy_j'])
    close(value['gpu_energy_j']/value['continuous_wall_s'],report['mean_board_power_w'])
    truth=[float(r['target_mos']) for r in calls];prediction=[float(r['prediction']) for r in calls]
    close(spearmanr(truth,prediction).statistic,report['performance200']['srcc'])
    close(pearsonr(truth,prediction).statistic,report['performance200']['plcc'])
    for name,digest in report['sources'].items():
        blob=subprocess.check_output(['git','show',report['source_commit']+':'+name.replace('\\','/')],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Source not published at recorded Git')
    events=entry.read(folder/'process_audit.json')
    if len(events)<2:raise ValueError('Missing process telemetry')
    bound=False
    for event in events:
        clients=set(event['clients'])
        if clients-{report['pid']}:raise ValueError('Foreign MPS client')
        if clients:bound=True
        elif bound:raise ValueError('Own MPS client disappeared after binding')
        if any(int(p[0]) not in {report['pid'],*event['servers']} or p[1].strip()!=GPU for p in event['processes']):
            raise ValueError('Foreign GPU compute')
    if not bound:raise ValueError('Missing own bound MPS client')
    return dict(status='passed',report_sha256=entry.sha(folder/'report.json'),calls_sha256=entry.sha(folder/'calls.csv'),
        telemetry_sha256=entry.sha(folder/'telemetry.csv'),repeat=report['repeat'],checkpoint_sha256=PIN,
        cost=value,measured200=report['performance200'],full558_reference=report['full558_reference'],
        common_input_tokens200=report['effective_input_tokens200'],mean_board_power_w=report['mean_board_power_w'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.folder,args.manifest)
    with args.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(status=result['status'],repeat=result['repeat'],cost=result['cost'])))


if __name__=='__main__':main()
