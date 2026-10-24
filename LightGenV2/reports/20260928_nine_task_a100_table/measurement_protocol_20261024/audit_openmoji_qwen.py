"""Independent CPU arithmetic/identity audit of cost trials, no model rerun."""
from __future__ import annotations
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from audit_openmoji_e45 import read,rows,close,artifact
from openmoji_qwen_live import GPU,sha,REFERENCE_SHA
from openmoji_grid_audit import metrics,native_bf16_edit
from protocol_identity import effective_protocol


def audit(folder):
    folder=Path(folder);identity=read(folder/'identity.json');summary=read(folder/'summary.json');state=read(folder/'status.json')
    row=next(r for r in effective_protocol()['rows'] if r['id']=='07')
    if (identity['protocol_id']!=effective_protocol()['protocol_id'] or identity['selected_body_sha256']!=row['qwen_pt_sha256']
            or identity['gpu_uuid']!=GPU or identity['batch']!=1 or identity['mps_percent']!=100
            or identity['visible_multiprocessors']!=108 or abs(identity['power_limit_w']-250)>.1
            or identity['explicit_warmup_forwards']!=0 or not identity['first_sample_retained']
            or not identity['amp_enabled'] or identity['autocast_dtype']!='bfloat16'
            or state['status']!='complete' or state['samples']!=200 or summary['valid_samples']!=200
            or not identity['original_native_CPU_feature_return_preserved']
            or identity['training_cache_used_as_online_input']):
        raise ValueError('Adopted epoch30 complete real-input cost scope differs')
    for record in read(folder/'files_manifest.json'):
        p=folder/record['file']
        if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Original raw artifact changed')
    proof_path=artifact(identity['cpu_contract']);proof=read(proof_path)
    if (sha(proof_path)!=identity['cpu_contract_sha256'] or proof['status']!='passed' or not proof['no_gpu']
            or [r['test_index'] for r in proof['checks']]!=[200,999] or not all(r['bitwise_equal'] for r in proof['checks'])):
        raise ValueError('Complete original native CPU contract missing')
    manifest=folder/'original_TEST.jsonl'
    if sha(manifest)!=row['test_jsonl_sha256']:raise ValueError('Not the same new-layout full TEST identity')
    reference=folder/'original_full_TEST_reference.json'
    if sha(reference)!=REFERENCE_SHA or read(reference)['metrics']['overall']!=summary['formal_full1000_reference_metrics']:
        raise ValueError('Formal accuracy must be original full1000, not cost subset')
    original=[json.loads(s) for s in manifest.read_text().splitlines()]
    calls=rows(folder/'calls.csv');inputs=read(folder/'original_input_manifest.json')
    if len(original)!=1000 or len(calls)!=200 or len(inputs)!=200:raise ValueError('Not the exact first200 cost workload')
    pngs=np.load(folder/'original_source_PNG_bytes.npz',allow_pickle=False);out=np.load(folder/'grid_outputs.npz',allow_pickle=False)
    expected_shapes=dict(category_logits=(200,17,6,6),edit_logits=(200,6,6),task_logits=(200,4),
        prediction=(200,6,6),generated=(200,6,6),edited=(200,6,6))
    if any(out[k].shape!=shape or not np.isfinite(out[k]).all() for k,shape in expected_shapes.items()):raise ValueError('Original output dimensions/content differ')
    if (not np.array_equal(out['generated'],out['category_logits'].argmax(1))
            or not np.array_equal(out['edited'],native_bf16_edit(out['edit_logits']))):raise ValueError('Original native output decisions differ')
    tokens=0;values=[];gaps=0.
    expected_counts=proof['checks'][0]['native_counts']
    for i,(call,record,raw) in enumerate(zip(calls,original,inputs)):
        if int(call['index'])!=i or call['sample_id']!=record['sample_id'] or raw['sample_id']!=record['sample_id']:
            raise ValueError('Changed original first200 IDs/order')
        if call['instruction']!=record['instruction'] or call['instruction_utf8_sha256']!=hashlib.sha256(record['instruction'].encode()).hexdigest():
            raise ValueError('Changed original UTF8 instruction')
        native=json.loads(call['native_inputs_json']);ids=native['input_ids'];mask=native['attention_mask']
        if (len(ids)!=1 or len(mask)!=1 or len(ids[0])!=len(mask[0]) or any(v!=1 for v in mask[0])
                or not all(isinstance(x,int) for x in ids[0]) or int(call['valid_input_tokens'])!=sum(mask[0])
                or native['shapes']['input_ids']!=[1,len(ids[0])]
                or json.loads(call['native_layer_counts_json'])!=expected_counts
                or json.loads(call['visual_feature_shape_json'])!=[196,1024]
                or json.loads(call['language_feature_shape_json'])[-1]!=2048):
            raise ValueError('Actual native token/mask/layer/feature shape contract differs')
        tokens+=sum(mask[0]);data=pngs[f'PNG_{i:03d}'].tobytes()
        if len(data)!=raw['bytes'] or hashlib.sha256(data).hexdigest()!=raw['sha256']:raise ValueError('Original PNG SHA differs')
        with Image.open(BytesIO(data)) as image:
            if image.size!=(224,224):raise ValueError('Original native image geometry differs')
            image.convert('RGB').load()
        expected=np.where(out['edited'][i],out['generated'][i],np.asarray(record['source_grid']))
        if not np.array_equal(expected,out['prediction'][i]) or call['edit_logits_dtype']!='torch.bfloat16':raise ValueError('Original composition/AMP differs')
        values.append(metrics(out['prediction'][i],out['edited'][i],int(out['task_logits'][i].argmax()),record))
        a,b=float(call['input_start']),float(call['cpu_result_return'])
        gap=0. if not i else a-float(calls[i-1]['cpu_result_return'])
        if a>b or gap<0:raise ValueError('Overlapping real query windows')
        close(gap*1000,call['internal_gap_ms']);gaps+=gap
    for k in values[0]:close(sum(r[k] for r in values)/200,summary['measured_first200_metrics'][k])
    close(tokens,summary['common_Qwen_input_tokens200'])
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    a,b=float(calls[0]['input_start']),float(calls[-1]['cpu_result_return'])
    board=observed_energy([(a,b)],rows(folder/'telemetry.csv'));energy=338.2*(b-a)+board
    close(b-a,summary['continuous_wall_s']);close(board,summary['board_energy_j']);close(energy,summary['system_energy_j'])
    close(board/(b-a),summary['time_weighted_board_w'])
    bound=False
    for event in read(folder/'process_audit.json'):
        if set(event['clients'])-{identity['pid']}:raise ValueError('Non-owned private MPS client')
        if event['clients']:bound=True
        elif bound:raise ValueError('Own MPS client disappeared within window')
        if any(int(p[0]) not in {identity['pid'],*event['servers']} or p[1].strip()!=GPU for p in event['processes']):raise ValueError('Non-owned A100 compute process')
    if not bound:raise ValueError('Own private MPS client never observed')
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original measurement source differs from published Git')
    return dict(status='passed',repeat=identity['repeat'],calls=200,report_sha256=sha(folder/'summary.json'),
        output_sha256=sha(folder/'grid_outputs.npz'),cpu_contract_sha256=identity['cpu_contract_sha256'],
        formal_full1000_reference_metrics=summary['formal_full1000_reference_metrics'],
        measured_first200_diagnostics_only=summary['measured_first200_metrics'],
        common_Qwen_input_tokens200=tokens,continuous_wall_s=b-a,system_energy_j=energy,board_energy_j=board,
        time_weighted_board_w=board/(b-a),sample_ms=(b-a)*5,sample_j=energy/200,internal199_gaps_s=gaps)
