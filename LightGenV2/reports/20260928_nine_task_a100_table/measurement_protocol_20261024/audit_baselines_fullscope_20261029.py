"""Independent local data audit and static pooled full-input baseline values."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,value):
    with Path(p).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=True,allow_nan=False)

def audit_point(leaf,destination):
    import run_baselines_fullscope_20261029 as entry
    leaf=Path(leaf).resolve();entry.configure(leaf.parent)
    manifest=REPO/'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv'
    result=entry.base.audit_point(leaf,manifest)
    if result['measured_videos']!=512 or result['status']!='passed':raise ValueError('Independent complete512 audit required')
    save(destination,result);return result

def collect(root,qwen50,destination,LG_summary=None):
    root=Path(root);qwen50=Path(qwen50);state=read(root/'suite_status.json')
    import run_baselines_fullscope_20261029 as entry
    expected=[f'{m}_cap{c}_repeat{r:02d}' for m,c,r in entry.missing_plan('lab')]
    if state['status']!='complete' or state['completed']!=expected or not state['mps_released']:
        raise ValueError('All missing21 complete and controller released required')
    if read(qwen50/'local_independent_pooled_audit01.json')['status']!='passed':raise ValueError('Retained Qwen50 audit required')
    lg=read(LG_summary) if LG_summary else None
    records=[]
    for model in ('qwen','glm','clip','yolo'):
        for cap in (50,100):
            points=[];audits=[]
            for repeat in (1,2,3):
                parent=qwen50 if (model,cap)==('qwen',50) else root
                leaf=parent/f'{model}_cap{cap}_repeat{repeat:02d}';report=read(leaf/'report.json')
                if parent==root:
                    audit=read(leaf/'local_independent_fullscope_audit01.json')
                    if audit['report_sha256']!=sha(leaf/'report.json') or audit['status']!='passed':raise ValueError('Original point report/audit differs')
                else:audit=read(leaf/'cpu_audit.json')
                if (report['batch'],report['cap'],report['explicit_warmup_forwards'],report['measured_videos'])!=(1,cap,0,512):
                    raise ValueError('Inconsistent cold baseline scope')
                points.append(report);audits.append(audit)
            elapsed=sum(p['cost']['continuous_wall_s'] for p in points)
            gpu_j=sum(p['cost']['gpu_energy_j'] for p in points)
            system_j=sum(p['cost']['estimated_system_energy_j'] for p in points)
            tokens=sum(p['effective_input_tokens512'] for p in points)
            row=dict(model=model,MPS_percent=cap,repeats=3,video_calls=1536,groups16=96,
                time16_s=elapsed/96,frame_s=elapsed/(1536*4),frames_per_s=1536*4/elapsed,
                native_effective_input_tokens=tokens,native_tokens_per_s=tokens/elapsed if tokens else None,
                GPU_mean_W=gpu_j/elapsed,host_proxy_W=338.2,hardware_power_cap_W=250,
                energy16_J=system_j/96,energy_per_frame_J=system_j/(1536*4),
                round_time16_s=[p['cost']['continuous_wall_s']/32 for p in points],
                round_report_SHA256=[a['report_sha256'] for a in audits],
                whole_file_input_to_CPU_output_measured=True,model_loading_separate=True)
            if lg:
                row.update(time_ratio_to_LightGenII=row['time16_s']/lg['LG_time16_s'],
                    tokens_per_s_ratio_LightGenII_to_native=lg['LG_tokens_per_s']/row['native_tokens_per_s'] if tokens else None,
                    comparison_is_same_scope_physical_E2E=False,LightGenII_energy_ratio=None)
            records.append(row)
    result=dict(status='PASSED_8_CONFIG_24_POINT_FULL_INPUT_BASELINES',new_points=21,retained_Qwen50_points=3,
        point_video_count=512,records=records,DS_Kimi_not_remeasured=True,
        tokens_policy='Each baseline retains its own actual nonpadding input count; only LightGen-II shares Qwen numerator',
        energy_policy='matched NVML board integral plus338.2W host proxy once, not wattmeter total')
    save(destination,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('point','collect'),required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qwen50',type=Path);p.add_argument('--LG-summary',dest='LG_summary',type=Path);args=p.parse_args()
    result=audit_point(args.source,args.output) if args.phase=='point' else collect(args.source,args.qwen50,args.output,args.LG_summary)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cost','records')},ensure_ascii=True))
