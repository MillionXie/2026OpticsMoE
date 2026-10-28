"""Independent local replay and pooled summary of the new cold Qwen triple."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))

def audit(root, destination):
    import run_temporal_qwen_fullscope_20261029 as entry
    root=Path(root).resolve();entry.configure(root)
    suite=json.loads((root/'suite_status.json').read_text())
    expected=[f'qwen_cap50_repeat{i:02d}' for i in (1,2,3)]
    if suite['status']!='complete' or suite['completed']!=expected or not suite['mps_released']:
        raise ValueError('Three complete original rounds and MPS release required')
    manifest=REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv'
    points=[entry.base.audit_point(root/name,manifest) for name in expected]
    seconds=sum(p['cost']['continuous_wall_s'] for p in points)
    gpu_j=sum(p['cost']['gpu_energy_j'] for p in points)
    system_j=sum(p['cost']['estimated_system_energy_j'] for p in points)
    result=dict(status='passed',protocol=entry.NEW_PROTOCOL,videos=1536,groups16=96,
        total_continuous_s=seconds,time16_s=seconds/96,LG_100x_threshold_s=seconds/9600,
        round_time16_s=[p['cost']['continuous_wall_s']/32 for p in points],
        GPU_average_W=gpu_j/seconds,host_proxy_W=338.2,energy16_J=system_j/96,
        model_loading_excluded=True,explicit_inference_warmups=0,all_file_preparation_and_internal_gaps_included=True,
        weighting='sum elapsed / sum groups; no fastest round selection',round_audits=points)
    with Path(destination).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=True,allow_nan=False)
    return result

if __name__=='__main__':
    r=audit(sys.argv[1],sys.argv[2]);print(json.dumps({k:v for k,v in r.items() if k!='round_audits'},ensure_ascii=True))
