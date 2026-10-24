"""No-model independent original LGVQ first200 field, stage, input and power audit."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from measure_lgvq_cost import live,GPU,PROTOCOL,proof_contract,sha,write
from audit_t12_cost import source_blobs,rows,close,check_platform
from audit_abo_title_student import ledger_key


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def mapped(value):
    prefix='/DATA/DATA1/guest3/2026OpticsMoE/'
    value=str(value).replace('\\','/')
    if not value.startswith(prefix) or '..' in Path(value[len(prefix):]).parts:raise ValueError('Unknown original repo artifact')
    return REPO/value[len(prefix):]


def expected_fields(target):
    # Independent grouping, not importing the measurement field/cost helpers.
    if target not in ('temporal','spatial'):raise ValueError('Only original temporal/spatial')
    slots=16 if target=='temporal' else 1
    indices=list(range(200));flags=[True]*200
    extra=(-200)%slots;indices+=list(range(extra));flags+=[False]*extra
    return [(indices[i:i+slots],flags[i:i+slots]) for i in range(0,len(indices),slots)]


def audit(folder):
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    root=Path(folder);identity=read(root/'identity.json');summary=read(root/'summary.json');state=read(root/'status.json')
    target=identity['target'];plan=expected_fields(target)
    if (target not in ('temporal','spatial') or identity['protocol_id']!=PROTOCOL or identity['system']!='ours'
            or identity['body_sha256']!=live.PINS[target][1] or identity['manifest_sha256']!=live.MANIFEST_SHA
            or identity['front_binding_sha256']!=live.BINDING_SHA or identity['gpu_uuid']!=GPU
            or identity['visible_multiprocessors']!=108 or identity['power_limit_w']!=250 or identity['mps_percent']!=100
            or identity['batch']!=1 or identity['explicit_warmup_forwards']!=0 or identity['first_field_retained'] is not True
            or identity['frontend_transformer_layers_executed']!=0 or state['status']!='complete'
            or state['valid_samples']!=200 or state['fields']!=len(plan) or summary['status']!='complete'):
        raise ValueError('Approved original field/model/physical platform differs')
    manifest=read(root/'files_manifest.json');names=[r['file'] for r in manifest]
    required={'identity.json','summary.json','status.json','input_files.json','video_views.json','prepared_inputs.json',
        'fields.csv','stages.csv','field_predictions.npz','stage_ledger.json','nn_module_shapes.json','process_audit.json','telemetry.csv'}
    if not required<=set(names) or len(names)!=len(set(names)):raise ValueError('Complete sealed raw file manifest required')
    for item in manifest:
        if Path(item['file']).name!=item['file']:raise ValueError('Non-flat raw manifest path')
        p=root/item['file']
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Sealed raw changed')
    source_blobs(identity)
    proofpath=mapped(identity['cpu_contract']);proof=proof_contract(proofpath,check_sources=False);source_blobs(proof)
    if sha(proofpath)!=identity['cpu_contract_sha256']:raise ValueError('Actual original CPU proof changed')
    expected_dtypes={'body':['torch.float32'],'native_patch_position':['torch.bfloat16']}
    if target=='spatial':expected_dtypes['original_FrameStem192']=['torch.float32']
    if identity['parameter_dtypes']!=expected_dtypes:raise ValueError('Original body/front precision differs')
    if sha(live.BINDING)!=live.BINDING_SHA:raise ValueError('Original front binding changed')
    binding=read(live.BINDING)
    if len(identity['original_front_tensors'])!=len(binding['matched_front_tensors']):raise ValueError('Missing actual native component hashes')
    for actual,pin in zip(identity['original_front_tensors'],binding['matched_front_tensors']):
        if any(actual[k]!=pin[k] for k in ('name','shape','dtype','numel','sha256')):raise ValueError('Actual native component differs')
    if identity['native_prompt']!=next(c for c in binding['matched_prompt_checks'] if c['target']==target):
        raise ValueError('Actual original task prompt binding differs')
    if target=='spatial' and identity['original_stem_sha256']!=live.STEM_SHA:raise ValueError('Original FrameStem identity differs')
    if sha(live.MANIFEST)!=live.MANIFEST_SHA:raise ValueError('Original TEST manifest changed')
    with live.MANIFEST.open(encoding='utf-8-sig',newline='') as f:tests=[r for r in csv.DictReader(f) if r['split']=='test']
    if len(tests)!=558:raise ValueError('Original TEST558 required')
    expected=[r['sample_id'] for r in tests[:200]]
    refpath=mapped(identity['qualified_Qwen_input_reference']);reference=read(refpath)
    if sha(refpath)!=identity['qualified_Qwen_report_sha256']:raise ValueError('Qualified same-input Qwen evidence changed')
    refinputs=read(refpath.parent/'input_audit.json')
    if sha(refpath.parent/'input_audit.json')!=reference['input_audit_sha256']:raise ValueError('Original Qwen video dimensions evidence changed')
    files=read(root/'input_files.json')
    if set(files)!=set(expected):raise ValueError('Wrong first200 original video identities')
    for sid,item in files.items():
        if (item['sha256']!=reference['input_files'][sid]['sha256'] or item['bytes']!=reference['input_files'][sid]['size_bytes']
                or {k:v for k,v in item.items() if k!='sha256'}!=identity['video_metadata_before'][sid]):
            raise ValueError('Input bytes not matched to qualified original Qwen')
    views=read(root/'video_views.json')
    if set(views)!=set(expected):raise ValueError('Missing original video view identity')
    for sid,view in views.items():
        source=refinputs['original_video_metadata'][sid];n=view['reported_frame_count']
        if any(view[k]!=source[k] for k in ('width','height','reported_frame_count')):raise ValueError('Original video dimensions differ')
        positions=[min(n-1,max(0,round((n-1)*(.10+i*.80/3)))) for i in range(4)]
        if (view['Qwen_front_source_selected_positions']!=positions or view['Qwen_front_crop']!=.65
                or view['Qwen_front_size']!=448 or view['metadata_read_after_T1'] is not True):
            raise ValueError('Original crop448/temporal landmarks differ')
        if target=='spatial' and (view['raw224_source_selected_positions']!=[min(n-1,max(0,round((n-1)*f))) for f in (.10,.37,.63,.90)]
                or view['raw224_crop']!=1.):raise ValueError('Original independent raw224 view differs')
    fields=rows(root/'fields.csv');stages=rows(root/'stages.csv');prepared=read(root/'prepared_inputs.json')
    input_contract=next(c['inputs'] for c in proof['checks'] if c['target']==target)
    output=np.load(root/'field_predictions.npz',allow_pickle=False)
    if len(fields)!=len(plan) or len(prepared)!=len(plan):raise ValueError('Incomplete original field records')
    all_predictions=[];gaps=front=0.;previous=None
    for i,(record,(indices,valid)) in enumerate(zip(fields,plan)):
        a,b,c=(float(record[k]) for k in ('input_start','frontend_end','cpu_result_return'))
        if not a<b<c or previous is not None and previous>a:raise ValueError('Invalid continuous field window')
        gap=0. if previous is None else a-previous;close(gap*1000,record['internal_gap_ms'])
        close((b-a)*1000,record['frontend_wall_ms']);front+=(b-a);gaps+=gap;previous=c
        ids=json.loads(record['test_indices']);flags=json.loads(record['valid']);source_ids=json.loads(record['source_ids'])
        if int(record['field'])!=i or ids!=indices or flags!=valid or source_ids!=[expected[j] for j in indices]:
            raise ValueError('Wrong temporal tail membership/validity or original TEST order')
        native=json.loads(record['original_native_prompt']);pin=identity['native_prompt']
        if native['input_ids']!=pin['input_ids'] or native['attention_mask']!=pin['attention_mask'] or native['source_ids']!=source_ids:
            raise ValueError('Original prompt/tokenizer input changed')
        preds=json.loads(record['predictions'])
        if len(preds)!=len(indices) or not np.isfinite(preds).all():raise ValueError('Full original MOS output missing')
        all_predictions.extend(preds)
        shapes={k:v['shape'] for k,v in prepared[i].items()}
        if prepared[i].keys()!=input_contract.keys() or any(
                any(value[k]!=input_contract[name][k] for k in ('shape','dtype')) for name,value in prepared[i].items()):
            raise ValueError('Actual prepared input shape/dtype differs from original CPU contract')
        if target=='temporal':
            if shapes['vision_tokens']!=[1,16,4,49,1024] or shapes['quality_tokens']!=[1,16,4,49,14]:
                raise ValueError('Original temporal16x4 feature shape differs')
        elif shapes['vision_tokens']!=[1,4,196,1024] or shapes['quality_tokens']!=[1,4,196,192] or shapes['raw_frames']!=[1,4,3,224,224]:
            raise ValueError('Original spatial192/raw224 shape differs')
        if any(len(v['sha256'])!=64 for v in prepared[i].values()):raise ValueError('Post-window actual tensor SHA missing')
    if (output['predictions'].dtype!=np.float32 or not np.isfinite(output['predictions']).all()
            or not np.array_equal(output['predictions'],np.asarray(all_predictions,np.float32))
            or output['test_indices'].tolist()!=[j for ids,_ in plan for j in ids]
            or output['valid'].tolist()!=[v for _,flags in plan for v in flags]):
        raise ValueError('Full output/validity NPZ differs from original field records')
    charged=0.;count={};intervals={i:[] for i in range(len(plan))}
    for row in stages:
        i=int(row['sample_index']);a,b=float(row['start_monotonic']),float(row['end_monotonic'])
        if i not in intervals or not float(fields[i]['input_start'])<=a<=b<=float(fields[i]['cpu_result_return']):
            raise ValueError('Original observed span outside its request')
        close((b-a)*1000,row['synchronized_wall_ms']);intervals[i].append((a,b))
        count[i,row['stage']]=count.get((i,row['stage']),0)+1
        if row['region']=='frontend':
            if b>float(fields[i]['frontend_end']):raise ValueError('Frontend detail outside inclusive frontend')
        elif row['region']=='body':
            if a<float(fields[i]['frontend_end']):raise ValueError('Body span overlaps charged frontend')
            if row['charged']=='True':charged+=(b-a)
        else:raise ValueError('Unknown span region')
        if row['family'] in ('pure_fusion','optical_simulation_FFT') and row['charged']!='False':
            raise ValueError('Excluded fusion/FFT charged')
        if live.residual_name(row['stage']) and row['charged']!='False':raise ValueError('Parallel residual reclassified')
        if row['family'] not in ('frontend_cpu','output') and row['cuda_event_ms']=='':raise ValueError('Paired CUDA Event missing')
    for i,spans in intervals.items():
        if any(x[1]>y[0]+1e-8 for x,y in zip(spans,spans[1:])):raise ValueError('Overlapping summed scopes')
        if sum(1 for r in stages if int(r['sample_index'])==i and r['family']=='optical_simulation_FFT')!=6:
            raise ValueError('Original six passes per physical field missing')
        if any(count.get((i,k))!=1 for k in ('frame_merger','input.original_CPU_features_H2D','output.full_field_CPU_MOS')):
            raise ValueError('Actual merger/inclusive transfers missing')
    adopted={ledger_key(v) for c in proof['checks'] if c['target']==target for v in c['source_ledger']}
    if {ledger_key(v) for v in read(root/'stage_ledger.json')}!=adopted:raise ValueError('Selected CPU-proven original NN spans changed')
    elapsed=float(fields[-1]['cpu_result_return'])-float(fields[0]['input_start'])
    board=observed_energy([(float(fields[0]['input_start']),float(fields[-1]['cpu_result_return']))],rows(root/'telemetry.csv'))
    total=front+charged+gaps;optical=len(plan)*6*1.0447/1000;seconds=total+optical;joules=104.230*total+121.746*optical
    for key,value in dict(seconds200=seconds,joules200=joules,sample_ms=seconds*5,sample_j=joules/200,
            equivalent_frame_ms=seconds*1.25,equivalent_frame_j=joules/800,frontend_wall_ms_total=front*1000,
            charged_electronic_ms_total=total*1000,optical_ms_total=optical*1000,actual_software_wall_seconds=elapsed,
            diagnostic_software_board_j200=board,weighted_software_board_w=board/elapsed).items():close(value,summary[key])
    if (summary['field_count']!=len(plan) or summary['valid_videos']!=200
            or summary['executed_video_slots']!=sum(len(v) for _,v in plan)
            or summary['common_input_tokens200'] is not None):
        raise ValueError('Wrong denominator or unsupported common-token value')
    events=read(root/'process_audit.json');anchor=identity['clock_anchor'];offset=anchor['unix']-anchor['monotonic'];bound=False
    if len(events)<2 or events[0]['unix']>float(fields[0]['input_start'])+offset or events[-1]['unix']<float(fields[-1]['cpu_result_return'])+offset:
        raise ValueError('Full-window process/platform evidence missing')
    for event in events:
        platform=event['platform'];check_platform(','.join(str(platform[k]) for k in ('uuid','name','power_limit_w','memory_total_mib')))
        clients=set(event['clients'])
        if clients-{identity['pid']} or bound and not clients:raise ValueError('Foreign/disappeared private MPS client')
        bound=bound or bool(clients)
        if any(int(r[0]) not in {identity['pid'],*event['servers']} or r[1].strip()!=GPU for r in event['processes']):
            raise ValueError('Foreign A100 process')
    if not bound or {s['field'] for s in read(root/'nn_module_shapes.json')}!=set(range(len(plan))):
        raise ValueError('Original NN shapes/own MPS client absent')
    return dict(status='passed',target=target,repeat=identity['repeat'],valid_videos=200,fields=len(plan),
        executed_video_slots=sum(len(v) for _,v in plan),sample_ms=seconds*5,sample_j=joules/200,
        equivalent_frame_ms=seconds*1.25,equivalent_frame_j=joules/800,seconds200=seconds,joules200=joules,
        frontend_wall_s=front,selected_body_boundary_s=charged,all_field_gaps_s=gaps,parameter_optical_s=optical,
        summary_sha256=sha(root/'summary.json'),predictions_sha256=sha(root/'field_predictions.npz'),
        cpu_contract_sha256=identity['cpu_contract_sha256'],same_input_Qwen_sha256=identity['qualified_Qwen_report_sha256'],
        no_model_or_GPU_forward=True,formal_accuracy='approved original full558, not evaluated by this audit')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve original independent audit')
    write(args.output,audit(args.folder))
