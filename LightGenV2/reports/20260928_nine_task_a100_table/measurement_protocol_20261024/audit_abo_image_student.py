"""CPU-only .8058 original input/stage/ranking/energy replay; no model load."""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from audit_abo_text_qwen import original_csv_contract,read,close
from audit_abo_image_qwen import artifact_path
from audit_abo_title_student import ledger_key
from measure_abo_image_student import validate_bank,common_work,rows,CPU,BODY_SHA,GPU,PROTOCOL,sha


def audit_bank(root,data_root):
    titles,_,_=original_csv_contract(data_root);vectors,receipt=validate_bank(root,titles)
    for name,digest in receipt['sources'].items():
        blob=subprocess.check_output(['git','show',receipt['source_commit']+':'+name],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original offline bank source not published')
    events=read(Path(root)/'process_audit.json')
    for event in events:
        clients=set(event['clients'])
        if len(clients)>1:raise ValueError('Non-independent bank MPS clients')
        if any(int(p[0]) not in {*clients,*event['servers']} or p[1].strip()!=GPU for p in event['processes']):raise ValueError('Foreign bank process')
    return dict(status='passed',titles=100,shape=list(vectors.shape),dtype=str(vectors.dtype),
        bank_sha256=receipt['bank_sha256'],cpu_contract_sha256=receipt['cpu_contract_sha256'],not_a_formal_cost_run=True)


def audit(folder,data_root):
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    folder=Path(folder);identity=read(folder/'identity.json');summary=read(folder/'summary.json');state=read(folder/'status.json')
    titles,_,test=original_csv_contract(data_root);calls=rows(folder/'calls.csv');stages=rows(folder/'stages.csv')
    if (identity['protocol_id']!=PROTOCOL or identity['body_sha256']!=BODY_SHA or identity['gpu_uuid']!=GPU
            or identity['mps_percent']!=100 or identity['batch']!=1 or identity['visible_multiprocessors']!=108
            or abs(identity['power_limit_w']-250)>.1 or identity['ranking_stable'] or identity['score_device']!='cpu'
            or identity['score_dtype']!='FP32' or identity['explicit_warmup_forwards']!=0
            or identity['online_title_bank_forwards']!=0 or state['status']!='complete' or state['samples']!=200
            or summary['valid_samples']!=200 or len(calls)!=200):raise ValueError('Original .8058 scope/model differs')
    for item in read(folder/'files_manifest.json'):
        p=folder/item['file']
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Sealed original raw changed')
    proof_path=artifact_path(identity['cpu_contract']);proof=read(proof_path)
    if (sha(proof_path)!=identity['cpu_contract_sha256'] or proof['status']!='passed' or not proof['no_gpu']
            or proof['identity']['model_assets']!=identity['model_assets']
            or not all(r['bitwise_equal'] for r in proof['checks'])):raise ValueError('Original full CPU neural contract absent')
    bank_path=artifact_path(identity['bank_root']);vectors,receipt=validate_bank(bank_path,titles)
    if sha(bank_path/'bank_receipt.json')!=identity['bank_receipt_sha256']:raise ValueError('Offline original bank changed')
    anchor=artifact_path(identity['common_Qwen_calls']);common=common_work(rows(anchor),test[:200])
    if sha(anchor)!=identity['common_Qwen_calls_sha256']:raise ValueError('Original common work changed')
    out=np.load(folder/'retrieval_outputs.npz',allow_pickle=False);queries,scores,orders=(out[k] for k in ('vectors','scores','rankings'))
    if (queries.shape!=(200,64) or scores.shape!=(200,100) or orders.shape!=(200,100)
            or queries.dtype!=np.float32 or scores.dtype!=np.float32 or not np.isfinite(queries).all()
            or not np.isfinite(scores).all()):raise ValueError('Original output dimensions/dtype differ')
    normalized=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),np.float32(1e-12))
    if not np.allclose(queries@normalized.T,scores,atol=3e-6,rtol=3e-5):raise ValueError('Original CPU cosine scores differ')
    # Native task argsort is explicitly NON-stable. Check complete permutations
    # and descending scores; never invent stable tie order in an independent ABI.
    if not np.array_equal(np.sort(orders,axis=1),np.tile(np.arange(100),(200,1))):raise ValueError('Incomplete original rankings')
    ordered=np.take_along_axis(scores,orders,axis=1)
    if np.any(ordered[:,1:]>ordered[:,:-1]):raise ValueError('Original descending rank order differs')
    raw=np.load(folder/'original_JPEG_bytes.npz',allow_pickle=False);manifest=read(folder/'original_input_manifest.json');hits=[]
    gaps=0.
    for i,(call,sample,record,order) in enumerate(zip(calls,test,manifest,orders)):
        if (int(call['index'])!=i or call['sample_id']!=sample['sample_id'] or call['product_id']!=sample['product_id']
                or int(call['label'])!=int(sample['label']) or record['sample_id']!=sample['sample_id']
                or int(call['common_reference_tokens'])!=common[i]):raise ValueError('Different original first200 samples/work')
        data=raw[f'JPEG_{i:03d}'].tobytes()
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Original JPEG SHA differs')
        if json.loads(call['embedding_shape'])!=[1,64] or json.loads(call['detector_feature_shape'])!=[1,384]:raise ValueError('Original feature dimensions differ')
        ids=json.loads(call['input_ids']);mask=json.loads(call['attention_mask'])
        if len(ids)!=1 or len(ids[0])!=len(mask[0]) or sum(mask[0])!=int(call['valid_input_tokens']):raise ValueError('Actual input token shape differs')
        a,b=float(call['input_start']),float(call['cpu_result_return']);gap=0. if not i else a-float(calls[i-1]['cpu_result_return'])
        if a>b or gap<0:raise ValueError('Overlapping actual requests')
        close(gap*1000,call['internal_gap_ms']);gaps+=gap
        predicted=titles[int(order[0])]['product_id'];hit=int(predicted==sample['product_id'])
        if predicted!=call['predicted_product_id'] or int(call['hit1'])!=hit:raise ValueError('Original prediction differs')
        hits.append(hit)
    declared=read(folder/'stage_ledger.json')
    if {ledger_key(e) for e in declared}!={ledger_key(e) for e in proof['stage_ledger']}:raise ValueError('Observed original NN source spans differ')
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original measured source differs')
    counts=Counter();charged=0.;intervals=defaultdict(list);totals=defaultdict(float)
    for row in stages:
        i=int(row['sample_index']);a=float(row['start_monotonic']);b=float(row['end_monotonic']);ms=float(row['synchronized_wall_ms'])
        if i not in range(200) or a>b or a<float(calls[i]['input_start']) or b>float(calls[i]['cpu_result_return']):raise ValueError('Stage outside real request')
        close(ms,(b-a)*1000);intervals[i].append((a,b));counts[row['stage']]+=1;totals[row['stage']]+=ms
        if row['family'] in ('simulation_only','parallel_residual','layout','fusion') and row['charged']!='False':raise ValueError('Excluded projection residual charged')
        if row['charged']=='True':charged+=ms
        if row['family'] not in ('input','output') and row['cuda_event_ms']=='':raise ValueError('Missing paired CUDA Event')
    for values in intervals.values():
        if any(a[1]>b[0]+1e-8 for a,b in zip(values,values[1:])):raise ValueError('Overlapping summed spans')
    for mode in ('vision','language'):
        if counts[mode+'.router_simulation']!=200 or counts[mode+'.expert_global_simulation']!=400:raise ValueError('Six light propagation counts differ')
    for name in ('frontend.patch_embedding','frontend.token_embedding','bridge.native_merger','head.retrieval_readout',
            'input.original_RGB_fit224_processor_h2d','output.cpu_embedding','output.cpu_original_cosine_nonstable_rank'):
        if counts[name]!=200:raise ValueError('Required original NN/frontend/merger missing: '+name)
    charged+=gaps*1000;optical=200*6*1.0447;seconds=(charged+optical)/1000;energy=(121.746*optical+104.230*charged)/1000
    close(charged,summary['charged_electronic_ms_total']);close(optical,summary['optical_ms_total'])
    close(seconds*5,summary['cost']['sample_ms']);close(energy/200,summary['cost']['sample_j'])
    close(sum(common),summary['common_input_tokens200']);close(sum(hits)/200,summary['first200_diagnostic_hit1'])
    elapsed=float(calls[-1]['cpu_result_return'])-float(calls[0]['input_start']);close(elapsed,summary['actual_software_wall_seconds'])
    board=observed_energy([(float(calls[0]['input_start']),float(calls[-1]['cpu_result_return']))],rows(folder/'telemetry.csv'))
    bound=False
    for event in read(folder/'process_audit.json'):
        clients=set(event['clients'])
        if clients-{identity['pid']}:raise ValueError('Foreign private MPS client')
        if clients:bound=True
        elif bound:raise ValueError('Own MPS client disappeared')
        if any(int(p[0]) not in {identity['pid'],*event['servers']} or p[1].strip()!=GPU for p in event['processes']):raise ValueError('Foreign A100 process')
    if not bound:raise ValueError('Own process never observed')
    return dict(status='passed',repeat=identity['repeat'],calls=200,report_sha256=sha(folder/'summary.json'),
        outputs_sha256=sha(folder/'retrieval_outputs.npz'),original_bank_sha256=receipt['bank_sha256'],
        original_CPU_contract_sha256=identity['cpu_contract_sha256'],projected_seconds200=seconds,phase_estimated_j200=energy,
        sample_ms=seconds*5,sample_j=energy/200,charged_electronic_ms_total=charged,optical_ms_total=optical,
        common_input_tokens200=sum(common),full2400_reference_hit1=.8058333333333333,
        first200_diagnostic_hit1=sum(hits)/200,software_wall_s=elapsed,internal199_gaps_s=gaps,
        diagnostic_software_board_j200=board,module_call_counts=dict(counts),stage_wall_ms=dict(totals),not_a_device_measured_e2e=True)
