"""Independent CPU audit of original title NN stages, complete ranking and cost.

No Torch, CUDA, model forward, synthetic detector or resaved predictions. All
identities and the first200/two100 title order are replayed from original raw.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from audit_abo_text_qwen import original_csv_contract,validate_calls,close,read
from audit_abo_image_qwen import artifact_path
from measure_abo_title_student import validate_gallery,common_work,rows
from measure_abo_text_qwen import GPU,PROTOCOL,sha
from abo_student_graph import BODY_SHA
from abo_query_contract import timed_calls


def ledger_key(entry):
    return tuple(entry[k] for k in ('stage','family','charged','method','source_sha256','first_source_line','last_source_line'))


def audit(folder,data_root):
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    folder=Path(folder);identity=read(folder/'identity.json');summary=read(folder/'summary.json');state=read(folder/'status.json')
    titles,_,test=original_csv_contract(data_root)
    calls=rows(folder/'calls.csv');stages=rows(folder/'stages.csv')
    if (identity['protocol_id']!=PROTOCOL or identity['system']!='ours' or identity['body_sha256']!=BODY_SHA
            or identity['gpu_uuid']!=GPU or identity['mps_percent']!=100 or identity['batch']!=1
            or identity['visible_multiprocessors']!=108 or abs(identity['power_limit_w']-250)>.1
            or identity['score_device']!='cpu' or identity['score_dtype']!='FP32' or not identity['ranking_stable']
            or identity['descriptor_dimensions']!=64 or identity['explicit_warmup_forwards']!=0
            or not identity['first_sample_retained'] or identity['online_gallery_forwards']!=0
            or identity['online_visual_forwards']!=0 or state['status']!='complete' or state['samples']!=200
            or summary['status']!='complete' or summary['valid_samples']!=200 or summary['unique_titles']!=100):
        raise ValueError('Approved original title model/precision/workload differs')
    for record in read(folder/'files_manifest.json'):
        p=folder/record['file']
        if sha(p)!=record['sha256'] or p.stat().st_size!=record['bytes']:raise ValueError('Original raw artifact changed')
    elapsed,gaps,native_tokens=validate_calls(calls,titles)
    close(elapsed,summary['actual_software_wall_seconds'])
    close(native_tokens,summary['actual_native_input_tokens200'])
    bank_root=artifact_path(identity['gallery_root']);vectors,receipt=validate_gallery(bank_root,titles,test)
    if sha(bank_root/'receipt.json')!=identity['gallery_receipt_sha256'] or receipt['gallery_sha256']!=identity['gallery_sha256']:
        raise ValueError('Offline FP32 gallery receipt changed')
    contract_path=artifact_path(identity['title_contract']);contract=read(contract_path)
    if (sha(contract_path)!=identity['title_contract_sha256'] or contract['status']!='passed' or not contract['no_gpu']
            or contract['identity']['model_assets']!=identity['model_assets']
            or receipt['identity']['model_assets']!=identity['model_assets']
            or [r['title_index'] for r in contract['checks']]!=[0,99]
            or not all(r['bitwise_equal'] for r in contract['checks'])):
        raise ValueError('Original actual-title CPU neural contract is not bound')
    qwen=artifact_path(identity['common_Qwen_calls']);reference_calls=rows(qwen)
    common=common_work(reference_calls,timed_calls('text_to_image',titles,test))
    if sha(qwen)!=identity['common_Qwen_calls_sha256']:raise ValueError('Common original Qwen calls changed')
    close(sum(common),summary['common_input_tokens200'])
    for call,reference,reference_call in zip(calls,common,reference_calls):
        if (int(call['common_reference_tokens'])!=reference
                or json.loads(call['input_ids'])!=json.loads(reference_call['input_ids'])
                or json.loads(call['attention_mask'])!=json.loads(reference_call['attention_mask'])):
            raise ValueError('Different actual/common input token work')
        if json.loads(call['embedding_shape'])!=[1,64] or json.loads(call['detector_feature_shape'])!=[1,384]:
            raise ValueError('Original observed feature dimensions changed')
    output=np.load(folder/'retrieval_outputs.npz',allow_pickle=False)
    queries,scores,orders=(output[k] for k in ('vectors','scores','rankings'))
    if (queries.shape!=(200,64) or scores.shape!=(200,2400) or orders.shape!=(200,2400)
            or queries.dtype!=np.float32 or scores.dtype!=np.float32 or not np.isfinite(queries).all()
            or not np.isfinite(scores).all() or not np.allclose(np.linalg.norm(queries,axis=1),1,atol=2e-6)):
        raise ValueError('Original FP32 descriptor/score output shape differs')
    normalized=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),np.float32(1e-12))
    if not np.allclose(queries@normalized.T,scores,atol=3e-6,rtol=3e-5):raise ValueError('Original CPU cosine scores differ')
    if not np.array_equal(orders,np.argsort(-scores,axis=1,kind='stable')):raise ValueError('Original full CPU stable ranking differs')
    hits=[]
    for call,title,order in zip(calls,titles*2,orders):
        predicted=test[int(order[0])]['product_id'];hit=int(predicted==title['product_id'])
        if call['predicted_product_id']!=predicted or int(call['hit1'])!=hit:raise ValueError('Original exact-SKU prediction differs')
        hits.append(hit)
    close(sum(hits[:100])/100,summary['measured_unique100_hit1'])
    # Same two passes remain separate original calls, never cached predictions.
    if [r['pass_index'] for r in calls]!=['0']*100+['1']*100:raise ValueError('Not two complete100-title passes')
    declared=read(folder/'stage_ledger.json')
    expected={ledger_key(e) for e in contract['unique_stage_ledger']}
    if {ledger_key(e) for e in declared}!=expected:raise ValueError('Measured NN spans differ from paired original CPU forward')
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name.replace('\\','/')],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Runtime source not bound to published Git')
    counts=Counter();totals=defaultdict(float);intervals=defaultdict(list);charged=0.
    for stage in stages:
        i=int(stage['sample_index']);a=float(stage['start_monotonic']);b=float(stage['end_monotonic'])
        if (i not in range(200) or not math.isfinite(a+b) or a>b
                or a<float(calls[i]['input_start']) or b>float(calls[i]['cpu_result_return'])):
            raise ValueError('Stage outside its genuine request')
        ms=float(stage['synchronized_wall_ms']);close(ms,(b-a)*1000)
        intervals[i].append((a,b));counts[stage['stage']]+=1;totals[stage['stage']]+=ms
        if stage['family'] in ('simulation_only','parallel_residual','layout','fusion') and stage['charged']!='False':
            raise ValueError('Excluded layout, optics simulation or parallel residual charged')
        if stage['charged']=='True':charged+=ms
        if stage['family'] not in ('input','output') and stage['cuda_event_ms']=='':raise ValueError('Missing paired CUDA Event')
    for values in intervals.values():
        if any(a[1]>b[0]+1e-8 for a,b in zip(values,values[1:])):raise ValueError('Double-counted overlapping stage intervals')
    expected_counts={'input.utf8_tokenizer_h2d':200,'output.cpu_embedding':200,'output.cpu_normalize_cosine_stable_rank':200,
        'frontend.token_embedding':200,'head.retrieval_readout':200,'language.router_simulation':200,
        'language.expert_global_simulation':400,'language.parallel_residual1':200,'language.parallel_residual2':200}
    if any(counts[k]!=v for k,v in expected_counts.items()):raise ValueError('Original required NN/three-pass occurrence count differs')
    if any(v for k,v in counts.items() if k.startswith(('vision.','bridge.','frontend.patch'))):raise ValueError('Visual/bridge work appeared in title-only query')
    for index,call in enumerate(calls):
        gap=0. if not index else 1000*(float(call['input_start'])-float(calls[index-1]['cpu_result_return']))
        close(gap,float(call['internal_gap_ms']));charged+=gap
    optical=200*3*1.0447;seconds=(charged+optical)/1000;energy=(121.746*optical+104.230*charged)/1000
    close(charged,summary['charged_electronic_ms_total']);close(optical,summary['optical_ms_total'])
    close(seconds*5,summary['cost']['sample_ms']);close(energy/200,summary['cost']['sample_j'])
    raw=rows(folder/'telemetry.csv')
    board=observed_energy([(float(calls[0]['input_start']),float(calls[-1]['cpu_result_return']))],raw)
    events=read(folder/'process_audit.json');bound=False
    if len(events)<2:raise ValueError('Missing whole-window process evidence')
    for event in events:
        clients=set(event['clients'])
        if clients-{identity['pid']}:raise ValueError('Foreign private MPS client')
        if clients:bound=True
        elif bound:raise ValueError('Own MPS client disappeared')
        if any(int(p[0]) not in {identity['pid'],*event['servers']} or p[1].strip()!=GPU for p in event['processes']):
            raise ValueError('Foreign A100 compute process')
    if not bound:raise ValueError('Own process was never observed')
    return dict(status='passed',repeat=identity['repeat'],calls=200,report_sha256=sha(folder/'summary.json'),
        original_gallery_sha256=identity['gallery_sha256'],original_CPU_contract_sha256=identity['title_contract_sha256'],
        outputs_sha256=sha(folder/'retrieval_outputs.npz'),common_input_tokens200=sum(common),actual_input_tokens200=native_tokens,
        software_wall_s=elapsed,internal199_gaps_s=gaps,charged_electronic_ms_total=charged,optical_ms_total=optical,
        projected_seconds200=seconds,phase_estimated_j200=energy,sample_ms=seconds*5,sample_j=energy/200,
        measured_unique100_hit1=sum(hits[:100])/100,diagnostic_software_board_j200=board,
        observed_bridge_calls=0,bridge_zero_reason='Pure-title query: original native image path not invoked; declared and checked, not missing',
        module_call_counts=dict(counts),stage_wall_ms=dict(totals),not_a_device_measured_e2e=True)
