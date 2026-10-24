"""Independent CPU replay of T08 title-query input-inclusive baseline raw."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_text_qwen import GPU,PROTOCOL,MODEL_CONFIG_SHA,MODEL_WEIGHTS_SHA,TEXT_BASELINE_BANK_SHA,sha


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-6):raise ValueError('T08 independent replay mismatch')


def validate_calls(calls,titles):
    if len(calls)!=200 or len(titles)!=100:raise ValueError('200 calls,100 unique titles required')
    last=-math.inf
    for i,(c,t) in enumerate(zip(calls,titles*2)):
        if (int(c['index'])!=i or int(c['query_index'])!=i%100 or int(c['pass_index'])!=i//100
                or c['product_id']!=t['product_id'] or c['title']!=t['title']):
            raise ValueError('Wrong fixed original title/pass order')
        a,p,m,b=map(float,(c['input_start'],c['prepare_end'],c['model_start'],c['cpu_result_return']))
        if not last<=a<=p<=m<b or not math.isfinite(a+p+m+b):raise ValueError('Invalid continuous input/output timestamps')
        last=b
        mask=json.loads(c['attention_mask']);ids=json.loads(c['input_ids'])
        if (len(ids)!=1 or len(mask)!=1 or len(ids[0])!=len(mask[0])
                or json.loads(c['input_ids_shape'])!=[1,len(ids[0])]
                or set(mask[0])-{0,1} or sum(mask[0])!=int(c['valid_input_tokens'])):
            raise ValueError('Invalid original B1 effective token evidence')
        if c['title_utf8_sha256']!=hashlib.sha256(t['title'].encode('utf8')).hexdigest():raise ValueError('UTF-8 input differs')
    elapsed=float(calls[-1]['cpu_result_return'])-float(calls[0]['input_start'])
    active=sum(float(c['cpu_result_return'])-float(c['input_start']) for c in calls)
    gaps=sum(float(b['input_start'])-float(a['cpu_result_return']) for a,b in zip(calls,calls[1:]))
    close(active+gaps,elapsed)
    return elapsed,gaps,sum(int(c['valid_input_tokens']) for c in calls)


def audit(folder,data):
    import numpy as np
    from abo_query_contract import load_contract
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    folder=Path(folder);report=read(folder/'report.json');state=read(folder/'status.json')
    titles,_,test=load_contract(data)
    def rows(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    calls=rows(folder/'calls.csv');raw=rows(folder/'telemetry.csv')
    if (report['status']!='complete' or state['status']!='complete' or state['samples']!=200
            or report['protocol_id']!=PROTOCOL or report['gpu_uuid']!=GPU
            or report['mps_percent']!=100 or report['batch']!=1 or report['visible_multiprocessors']!=108
            or abs(report['power_limit_w']-250)>.1 or report['explicit_warmup_forwards']!=0
            or not report['first_sample_retained'] or report['measured_calls']!=200
            or report['unique_queries']!=100 or report['full100_reference_hit1']!=.82
            or report['model_config_sha256']!=MODEL_CONFIG_SHA or report['model_weights_sha256']!=MODEL_WEIGHTS_SHA
            or report['bank_sha256']!=TEXT_BASELINE_BANK_SHA or report['score_device']!='cuda'
            or not report['ranking_stable'] or report['score_dtype']!='FP32' or report['descriptor_dimensions']!=2048):
        raise ValueError('Wrong original model, precision, device, workload or measurement contract')
    for item in read(folder/'files_manifest.json'):
        f=folder/item['file']
        if sha(f)!=item['sha256'] or f.stat().st_size!=item['bytes']:raise ValueError('Raw file SHA changed')
    elapsed,gaps,tokens=validate_calls(calls,titles)
    close(elapsed,report['continuous_wall_s']);close(tokens,report['common_reference_tokens'])
    outputs=np.load(folder/'retrieval_outputs.npz',allow_pickle=False)
    order,scores,vectors=(outputs[name] for name in ('rankings','scores','vectors'))
    if (order.shape!=(200,2400) or scores.shape!=(200,2400) or vectors.shape!=(200,2048)
            or scores.dtype!=np.float32 or vectors.dtype!=np.float32
            or not np.isfinite(scores).all() or not np.isfinite(vectors).all()
            or not np.array_equal(order,np.argsort(-scores,axis=1,kind='stable'))):
        raise ValueError('Not the complete original FP32 stable ranking')
    hits=[];first=[]
    for i,(call,ranking) in enumerate(zip(calls,order)):
        if set(ranking.tolist())!=set(range(2400)):raise ValueError('Missing/duplicated gallery IDs')
        predicted=test[int(ranking[0])]['product_id'];hit=int(predicted==titles[i%100]['product_id'])
        if predicted!=call['predicted_product_id'] or hit!=int(call['hit1']):raise ValueError('Prediction/label mismatch')
        hits.append(hit)
        first.append(next(j+1 for j,k in enumerate(ranking) if test[int(k)]['product_id']==titles[i%100]['product_id']))
    if not np.array_equal(order[:100],order[100:]):raise ValueError('Two identical title traversals predict differently; retain and investigate')
    quality=sum(hits[:100])/100;close(quality,report['measured_unique100_hit1'])
    board=observed_energy([(float(calls[0]['input_start']),float(calls[-1]['cpu_result_return']))],raw)
    energy=338.2*elapsed+board
    close(energy,report['total_system_energy_j']);close(board/elapsed,report['mean_board_power_w'])
    for name,digest in report['sources'].items():
        blob=subprocess.check_output(['git','show',report['source_commit']+':'+name.replace('\\','/')],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Source not at recorded published commit')
    events=read(folder/'process_audit.json')
    if len(events)<2:raise ValueError('Missing full independent process observations')
    bound=False
    for event in events:
        clients=set(event['clients'])
        if clients-{report['pid']}:raise ValueError('Foreign MPS client')
        if clients:bound=True
        elif bound:raise ValueError('Own MPS client disappeared')
        if any(int(r[0]) not in {report['pid'],*event['servers']} or r[1].strip()!=GPU for r in event['processes']):
            raise ValueError('Foreign compute in A100 observation')
    if not bound:raise ValueError('Own MPS client never observed')
    return dict(status='passed',repeat=report['repeat'],report_sha256=sha(folder/'report.json'),
        calls_sha256=sha(folder/'calls.csv'),outputs_sha256=sha(folder/'retrieval_outputs.npz'),
        continuous_wall_s=elapsed,internal199_gaps_s=gaps,board_energy_j=board,
        system_energy_j=energy,mean_board_power_w=board/elapsed,common_input_tokens200=tokens,
        measured_unique100_hit1=quality,measured_unique100_mrr=sum(1/r for r in first[:100])/100,
        full100_historical_reference_hit1=.82)
