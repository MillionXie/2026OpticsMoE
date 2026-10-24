"""Independent CPU raw-data audit; never load Torch/a model or execute GPU."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from openmoji_e45_graph import GPU,sha,validate_stage_counts
from openmoji_grid_audit import metrics,native_bf16_edit
from protocol_identity import effective_protocol


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Independent numeric replay differs')
def artifact(path):
    value=str(path).replace('\\','/');prefix='/DATA/DATA1/guest3/2026OpticsMoE/'
    return REPO/value[len(prefix):] if value.startswith(prefix) else Path(path)
def key(r):return tuple(r[k] for k in ('stage','family','charged','method','source_sha256','first_source_line','last_source_line'))


def audit(folder):
    folder=Path(folder);identity=read(folder/'identity.json');summary=read(folder/'summary.json');state=read(folder/'status.json')
    row=next(r for r in effective_protocol()['rows'] if r['id']=='07')
    if (identity['protocol_id']!=effective_protocol()['protocol_id'] or identity['selected_body_sha256']!=row['ours_pt_sha256']
            or identity['gpu_uuid']!=GPU or identity['batch']!=1 or identity['mps_percent']!=100
            or identity['visible_multiprocessors']!=108 or abs(identity['power_limit_w']-250)>.1
            or identity['explicit_warmup_forwards']!=0 or not identity['first_sample_retained']
            or not identity['amp_enabled'] or identity['autocast_dtype']!='bfloat16'
            or state['status']!='complete' or state['samples']!=200 or summary['valid_samples']!=200
            or not summary['not_a_device_measured_e2e'] or summary['common_Qwen_input_tokens200'] is not None):
        raise ValueError('Original epoch45 / precision / complete workload contract differs')
    for record in read(folder/'files_manifest.json'):
        p=folder/record['file']
        if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Original raw artifact changed')
    path=artifact(identity['stage_contract']);contract=read(path)
    if (sha(path)!=identity['stage_contract_sha256'] or contract['status']!='passed' or not contract['no_gpu']
            or [r['test_index'] for r in contract['checks']]!=[200,999]
            or not all(r['bitwise_equal'] for r in contract['checks'])):raise ValueError('Actual paired CPU neural-stage contract not bound')
    ledger=read(folder/'stage_ledger.json')
    if {key(r) for r in ledger if r['family'] not in ('input','output')}!={key(r) for r in contract['stage_ledger']}:
        raise ValueError('Measured stages differ from exact original CPU graph')
    manifest=folder/'original_TEST.jsonl'
    if sha(manifest)!=row['test_jsonl_sha256']:raise ValueError('Not the adopted new-layout TEST')
    original=[json.loads(s) for s in manifest.read_text().splitlines()]
    calls=rows(folder/'calls.csv');stages=rows(folder/'stages.csv');inputs=read(folder/'original_input_manifest.json')
    if len(original)!=1000 or len(calls)!=200 or len(inputs)!=200:raise ValueError('Not the complete first200 of original1000')
    pngs=np.load(folder/'original_source_PNG_bytes.npz',allow_pickle=False)
    outputs=np.load(folder/'grid_outputs.npz',allow_pickle=False)
    expected_shapes=dict(category_logits=(200,17,6,6),edit_logits=(200,6,6),task_logits=(200,4),
        prediction=(200,6,6),generated=(200,6,6),edited=(200,6,6))
    for name,shape in expected_shapes.items():
        if outputs[name].shape!=shape or not np.isfinite(outputs[name]).all():raise ValueError('Original output shape/content differs')
    if not np.array_equal(outputs['generated'],outputs['category_logits'].argmax(1)):raise ValueError('Native category argmax differs')
    if not np.array_equal(outputs['edited'],native_bf16_edit(outputs['edit_logits'])):
        raise ValueError('Original native-BF16 sigmoid decision differs; never silently cast or change threshold')
    all_metrics=[];tokens=0
    for i,(call,record,raw) in enumerate(zip(calls,original,inputs)):
        if int(call['index'])!=i or call['sample_id']!=record['sample_id'] or raw['sample_id']!=record['sample_id']:
            raise ValueError('Changed first200 identities/order')
        if call['instruction']!=record['instruction'] or call['instruction_utf8_sha256']!=hashlib.sha256(record['instruction'].encode()).hexdigest():
            raise ValueError('Original plain UTF8 input changed')
        ids=json.loads(call['plain_input_ids']);tokens+=len(ids)
        if len(ids)!=int(call['plain_input_tokens']) or not 0<len(ids)<=64:
            raise ValueError('Original plain token shapes/count differ')
        if json.loads(call['input_shapes'])!=[[1,3,224,224],[len(ids),2048]]:raise ValueError('Original live frontend dimensions differ')
        data=pngs[f'PNG_{i:03d}'].tobytes()
        if len(data)!=raw['bytes'] or hashlib.sha256(data).hexdigest()!=raw['sha256']:raise ValueError('Original PNG SHA differs')
        with Image.open(BytesIO(data)) as image:
            if image.size!=(224,224):raise ValueError('Original PNG geometry differs')
            image.convert('RGB').load()
        expected=np.where(outputs['edited'][i],outputs['generated'][i],np.asarray(record['source_grid']))
        if not np.array_equal(expected,outputs['prediction'][i]):raise ValueError('Original source-grid composition differs')
        if call['edit_logits_dtype']!='torch.bfloat16':raise ValueError('Original AMP output precision changed')
        all_metrics.append(metrics(outputs['prediction'][i],outputs['edited'][i],int(outputs['task_logits'][i].argmax()),record))
    for name in all_metrics[0]:close(sum(r[name] for r in all_metrics)/200,summary['measured_first200_metrics'][name])
    close(tokens,summary['actual_plain_input_tokens200'])
    counts=Counter();intervals=defaultdict(list);charged=0.
    for stage in stages:
        i=int(stage['sample_index']);a=float(stage['start_monotonic']);b=float(stage['end_monotonic'])
        if (i not in range(200) or not math.isfinite(a+b) or a>b
                or a<float(calls[i]['input_start']) or b>float(calls[i]['cpu_result_return'])):
            raise ValueError('Stage lies outside its real request')
        close((b-a)*1000,stage['synchronized_wall_ms']);counts[stage['stage']]+=1;intervals[i].append((a,b))
        if stage['family'] in ('parallel_residual','simulation_only','fusion','layout') and stage['charged']!='False':
            raise ValueError('Excluded original spans incorrectly charged')
        if stage['family']!='input' and stage['cuda_event_ms']=='':raise ValueError('Missing paired CUDA Event')
        if stage['charged']=='True':charged+=(b-a)*1000
    validate_stage_counts(counts,200)
    if any(counts[k]!=200 for k in ('input.PNG_plain_tokenizer_H2D','input.original_embedding_float_conversion','output.original_grid_and_CPU_result')):
        raise ValueError('Missing real input/output occurrence')
    for values in intervals.values():
        values.sort()
        if any(a[1]>b[0]+1e-8 for a,b in zip(values,values[1:])):raise ValueError('Overlapping charged/audit stage spans')
    gaps=0.
    for i,call in enumerate(calls):
        gap=0. if not i else float(call['input_start'])-float(calls[i-1]['cpu_result_return'])
        if gap<0:raise ValueError('Overlapping real queries')
        close(gap*1000,call['internal_gap_ms']);gaps+=gap
    charged+=1000*gaps;optical=200*6*1.0447;seconds=(charged+optical)/1000;energy=(121.746*optical+104.230*charged)/1000
    for a,b in ((charged,summary['charged_electronic_ms_total']),(optical,summary['optical_ms_total']),
            (seconds,summary['projected_seconds200']),(energy,summary['phase_estimated_j200'])):close(a,b)
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    elapsed=float(calls[-1]['cpu_result_return'])-float(calls[0]['input_start'])
    board=observed_energy([(float(calls[0]['input_start']),float(calls[-1]['cpu_result_return']))],rows(folder/'telemetry.csv'))
    close(elapsed,summary['diagnostic_software_wall_s']);close(board,summary['diagnostic_software_board_integral']['energy_j'])
    bound=False
    for event in read(folder/'process_audit.json'):
        if set(event['clients'])-{identity['pid']}:raise ValueError('Foreign private MPS client')
        if event['clients']:bound=True
        elif bound:raise ValueError('Own client disappeared inside measured window')
        if any(int(p[0]) not in {identity['pid'],*event['servers']} or p[1].strip()!=GPU for p in event['processes']):
            raise ValueError('Foreign A100 compute process')
    if not bound:raise ValueError('Own private MPS client never observed')
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original measured source differs from published Git')
    shapes=read(folder/'nn_module_shapes.json')
    if {int(r['sample_index']) for r in shapes}!=set(range(200)):raise ValueError('Missing actual NN shape observations')
    return dict(status='passed',repeat=identity['repeat'],calls=200,report_sha256=sha(folder/'summary.json'),
        output_sha256=sha(folder/'grid_outputs.npz'),stage_contract_sha256=identity['stage_contract_sha256'],
        measured_first200_metrics=summary['measured_first200_metrics'],actual_plain_input_tokens200=tokens,
        common_Qwen_input_tokens200=None,common_tokens_pending_actual_Qwen_matching_calls=True,
        projected_seconds200=seconds,phase_estimated_j200=energy,sample_ms=seconds*5,sample_j=energy/200,
        optical_ms_total=optical,charged_electronic_ms_total=charged,internal199_gaps_s=gaps,
        diagnostic_software_wall_s=elapsed,diagnostic_software_board_j200=board,
        actual_module_shape_records=len(shapes),stage_counts=dict(counts),not_a_device_measured_e2e=True)
