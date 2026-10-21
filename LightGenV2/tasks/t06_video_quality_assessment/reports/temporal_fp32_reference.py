"""Select and replay existing FP32 duty-control evidence without new inference."""
from __future__ import annotations
import csv
import json
from . import strict_tf_resource_audit_20261020 as audit

MODELS=('qwen','glm','clip','yolo','deepseek','kimi')
ROOT=audit.ROOT/'presentation'
TARGET=ROOT/'temporal_fp32_reference.json'


def select(data):
    assert data['status']=='all48_complete' and data['accounting_protocol']==audit.ACCOUNTING
    selected=[]
    for model in MODELS:
        pair=[p for p in data['values'] if (p['model'],p['precision'],p['method'])==(model,'fp32','temporal')]
        assert len(pair)==2 and {p['condition'] for p in pair}=={'unrestricted','busy50'}
        unrestricted=next(p for p in pair if p['condition']=='unrestricted')
        assert len({p['batch'] for p in pair})==1
        assert len({p['checkpoint_sha256'] for p in pair})==1
        assert len({p['accuracy_source_sha256'] for p in pair})==1
        for point in sorted(pair,key=lambda p:p['condition']!='unrestricted'):
            assert len(point['repeats'])==3 and point['target50_reached'] is not False
            assert point['gpu_count']==(2 if model=='deepseek' else 1)
            assert point['srcc_reference']==unrestricted['srcc_reference']
            selected.append(point)
    return selected


def main():
    source=ROOT/'eight_tables.json';before=audit.sha(source)
    data=audit.read(source);points=select(data);checked=[];accuracy_cache={}
    for point in points:
        raw_path=audit.REPO/point['source']
        assert audit.sha(raw_path)==point['source_sha256']
        replay=audit.check(raw_path.parent.parent,point['condition'])
        for key in ('time16_ms','energy16_j','busy_percent'):
            audit.close(replay[key],point[key])
        assert len(replay['repeats'])==3
        for a,b in zip(replay['repeats'],point['repeats']):
            for key in ('elapsed_s','compute_s','output_to_next_model_input_s','internal_wait_s','gpu_energy_j','video_evaluations'):
                audit.close(a[key],b[key])
        acc_path=audit.REPO/point['accuracy_source']
        if str(acc_path) not in accuracy_cache:
            accuracy_cache[str(acc_path)]=audit.verify_accuracy(acc_path,point['checkpoint_sha256'])
        acc=accuracy_cache[str(acc_path)]
        accuracy_record=audit.read(acc_path)
        accuracy_identity=accuracy_record.get('identity',accuracy_record)
        assert accuracy_identity['precision']=='fp32' and accuracy_identity['batch']==point['batch']
        audit.close(acc['srcc_reference'],point['srcc_reference'])
        assert acc['accuracy_source_sha256']==point['accuracy_source_sha256']
        assert acc['accuracy_predictions_sha256']==point['accuracy_predictions_sha256']
        if point['model'] not in ('deepseek','kimi'):
            audit.verify_lab_fp32_runtime(audit.read(raw_path.parent.parent/'identity.json'),audit.read(acc_path.with_name('identity.json')))
        cost_identity=audit.read(raw_path.parent.parent/'identity.json')
        sample_reference=audit.rows(acc_path.with_name('predictions.csv'))[:point['batch']]
        assert cost_identity['sample_ids']==[r['sample_id'] for r in sample_reference]
        assert cost_identity['frame_positions']==[audit.frame_positions(r) for r in sample_reference]
        output_delta=max(abs(float(v)-float(record['prediction'])) for repeat in point['repeats'] for v,record in zip(repeat['output'],sample_reference))
        audit.close(output_delta,point['resident_reference_max_abs_mos_delta'])
        checked.append(dict(model=point['model'],condition=point['condition'],source=point['source'],sha256=point['source_sha256'],resident_batch_reference_max_abs_mos_delta=output_delta))
        print('REPLAY_PASS',point['model'],point['condition'],flush=True)
    assert audit.sha(source)==before
    output=dict(status='complete',selection='FP32 temporal duty-control reference only',
        source=source.relative_to(audit.REPO).as_posix(),source_sha256=before,
        values=points,optical=data['optical'],
        accuracy_policy='Independent full558 unrestricted FP32 reference at the same batch shared between paced and unrestricted costs.',
        input_protocol='One prepared TEST batch resident on GPU and reused. Individually synchronized forwards. Not dataset-once inference.',
        timing='Sum first-native-block to synchronized GPU output plus output-to-next-model-call intervals. Exclude every model-call frontend. Pacing waits are inside intercall gaps, not added twice.',
        energy='Sum all used-GPU raw board-power integrals on charged intervals; add338.2W host proxy once. Driver-window busy reported separately.',
        hardware='DeepSeek FP32 uses two A100-SXM4-80GB/400W GPUs; Kimi uses one; laboratory models use one A100-PCIE-40GB/250W.',
        qualification='Unrestricted means no pacing, not guaranteed100% busy. DeepSeek unrestricted is below98%.',
        comparison_boundary='Historical warmed resident-input reference. Do not merge with cold distinct-video native MPS data or apply the historical BF16 Qwen B1 anchor to FP32.',
        no_new_gpu_inference=True)
    TARGET.write_text(json.dumps(output,indent=2)+'\n')
    (ROOT/'temporal_fp32_reference_audit.json').write_text(json.dumps(dict(status='passed',conditions=12,repeats=36,
        accuracy_references=len(accuracy_cache),source_sha256=before,reference_sha256=audit.sha(TARGET),verified=checked),indent=2)+'\n')
    print(json.dumps(dict(status='passed',conditions=12,repeats=36,accuracy_references=len(accuracy_cache))))


if __name__=='__main__':main()
