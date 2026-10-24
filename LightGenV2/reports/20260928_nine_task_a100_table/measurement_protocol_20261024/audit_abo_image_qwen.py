"""Independent CPU replay of file-to-ranking T08 Qwen image query raw."""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from audit_abo_text_qwen import original_csv_contract,close,read
from measure_abo_image_qwen import (GPU,PROTOCOL,MODEL_CONFIG_SHA,MODEL_WEIGHTS_SHA,
    ORIGINAL_SHA,ORIGINAL_COMMIT,sha,validate_bank)


def value(raw):return ast.literal_eval(raw) if isinstance(raw,str) else raw


def artifact_path(recorded):
    prefix='/DATA/DATA1/guest3/2026OpticsMoE/'
    raw=str(recorded).replace('\\','/')
    if not raw.startswith(prefix):raise ValueError('Unknown original artifact root')
    relative=raw[len(prefix):]
    if not relative.startswith('LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/') or '..' in Path(relative).parts:
        raise ValueError('Artifact is outside original T08 runs')
    return REPO/relative


def validate_calls(calls,test):
    if len(calls)!=200:raise ValueError('All first200 distinct original TEST calls required')
    previous=-math.inf
    for index,(call,row) in enumerate(zip(calls,test[:200])):
        if (int(call['index'])!=index or call['sample_id']!=row['sample_id'] or call['product_id']!=row['product_id']
                or int(call['label'])!=int(row['label']) or call['image_path']!=row['image_path']):
            raise ValueError('Original first200 identity/order differs')
        a,p,m,b=map(float,(call['input_start'],call['prepare_end'],call['model_start'],call['cpu_result_return']))
        if not previous<=a<=p<=m<b or not math.isfinite(a+p+m+b):raise ValueError('Invalid continuous200 timestamps')
        previous=b;ids=value(call['input_ids']);mask=value(call['attention_mask']);shapes=value(call['input_shapes'])
        if (len(ids)!=1 or len(mask)!=1 or len(ids[0])!=len(mask[0]) or set(mask[0])-{0,1}
                or shapes['input_ids']!=[1,len(ids[0])] or shapes['attention_mask']!=shapes['input_ids']
                or sum(mask[0])!=int(call['valid_input_tokens']) or shapes['image_grid_thw']!=[1,3]
                or len(shapes['pixel_values'])!=2):
            raise ValueError('Missing original B1 image/token input shape evidence')
    elapsed=float(calls[-1]['cpu_result_return'])-float(calls[0]['input_start'])
    active=sum(float(c['cpu_result_return'])-float(c['input_start']) for c in calls)
    gaps=sum(float(b['input_start'])-float(a['cpu_result_return']) for a,b in zip(calls,calls[1:]))
    close(active+gaps,elapsed)
    return elapsed,gaps,sum(int(c['valid_input_tokens']) for c in calls)


def audit(folder,data):
    import numpy as np
    from PIL import Image,ImageOps
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    folder=Path(folder);report=read(folder/'report.json');state=read(folder/'status.json')
    titles,_,test=original_csv_contract(data)
    def rows(path):
        with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    calls=rows(folder/'calls.csv');raw=rows(folder/'telemetry.csv')
    if (report['status']!='complete' or state['status']!='complete' or state['samples']!=200
            or report['protocol_id']!=PROTOCOL or report['gpu_uuid']!=GPU or report['mps_percent']!=100
            or report['batch']!=1 or report['visible_multiprocessors']!=108 or abs(report['power_limit_w']-250)>.1
            or report['explicit_warmup_forwards']!=0 or not report['first_sample_retained']
            or report['measured_calls']!=200 or report['unique_queries']!=200
            or report['full2400_reference_hit1']!=.7358333333333333 or report['ranking_stable']
            or report['score_device']!='cuda' or report['score_dtype']!='FP32' or report['descriptor_dimensions']!=2048
            or report['model_config_sha256']!=MODEL_CONFIG_SHA or report['model_weights_sha256']!=MODEL_WEIGHTS_SHA
            or report['original_source_sha256']!=ORIGINAL_SHA or report['original_source_commit']!=ORIGINAL_COMMIT):
        raise ValueError('Original model/precision/device/workload contract mismatch')
    for item in read(folder/'files_manifest.json'):
        path=folder/item['file']
        if sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('Original raw SHA mismatch')
    elapsed,gaps,tokens=validate_calls(calls,test)
    close(elapsed,report['continuous_wall_s']);close(tokens,report['common_reference_tokens'])
    bank_root=artifact_path(report['offline_title_bank']);receipt=read(bank_root/'receipt.json')
    if sha(bank_root/'title_bank.npz')!=report['bank_sha256'] or sha(bank_root/'receipt.json')!=report['bank_receipt_sha256']:
        raise ValueError('Independent offline bank SHA differs')
    gallery=validate_bank(np.load(bank_root/'title_bank.npz',allow_pickle=False),receipt,titles)
    contract_path=artifact_path(report['cpu_contract']);contract=read(contract_path)
    if sha(contract_path)!=report['cpu_contract_sha256']:raise ValueError('Original CPU contract SHA changed')
    if (contract['status']!='passed' or contract['sources']!=report['sources']
            or contract['bank_sha256']!=report['bank_sha256'] or not contract['no_gpu_inference']
            or [c['test_index'] for c in contract['checks']]!=[200,2399]
            or not all(c['bitwise_equal'] for c in contract['checks'])):
        raise ValueError('Original actual-image CPU contract is not bound')
    outputs=np.load(folder/'retrieval_outputs.npz',allow_pickle=False)
    order,scores,vectors=(outputs[name] for name in ('rankings','scores','vectors'))
    if (order.shape!=(200,100) or scores.shape!=(200,100) or vectors.shape!=(200,2048)
            or scores.dtype!=np.float32 or vectors.dtype!=np.float32 or not np.isfinite(scores).all()
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6)
            or not np.allclose(vectors@gallery.T,scores,atol=3e-6,rtol=3e-5)):
        raise ValueError('Original FP32 normalized vectors/cosine scores differ')
    ordered=np.take_along_axis(scores,order,axis=1)
    # Original GPU argsort was non-stable. Validate all scores/permutations,
    # never silently force CPU-stable tie order onto a different original sort.
    if np.any(ordered[:,1:]>ordered[:,:-1]):raise ValueError('Ranking is not descending')
    originals=np.load(folder/'original_image_bytes.npz',allow_pickle=False)
    hits=[];ranks=[]
    for index,(call,ranking,row) in enumerate(zip(calls,order,test[:200])):
        if set(ranking.tolist())!=set(range(100)):raise ValueError('Full title permutation missing')
        content=originals['image'+str(index)].tobytes()
        if hashlib.sha256(content).hexdigest()!=call['image_file_sha256']:raise ValueError('Original input file SHA differs')
        original=data/row['image_path']
        if original.exists() and sha(original)!=call['image_file_sha256']:raise ValueError('Original data file changed')
        with Image.open(io.BytesIO(content)) as source:image=ImageOps.exif_transpose(source).convert('RGB')
        if list(image.size)!=value(call['decoded_image_size']) or hashlib.sha256(image.tobytes()).hexdigest()!=call['decoded_RGB_sha256']:
            raise ValueError('Original input decode/EXIF/RGB content differs')
        predicted=titles[int(ranking[0])]['product_id'];hit=int(predicted==row['product_id'])
        if predicted!=call['predicted_product_id'] or hit!=int(call['hit1']):raise ValueError('Prediction/SKU mismatch')
        hits.append(hit);ranks.append(ranking.tolist().index(int(row['label']))+1)
    quality=sum(hits)/200;close(quality,report['measured_first200_hit1'])
    board=observed_energy([(float(calls[0]['input_start']),float(calls[-1]['cpu_result_return']))],raw)
    energy=338.2*elapsed+board;close(energy,report['total_system_energy_j']);close(board/elapsed,report['mean_board_power_w'])
    for name,digest in report['sources'].items():
        blob=subprocess.check_output(['git','show',report['source_commit']+':'+name.replace('\\','/')],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Recorded source not in published Git')
    events=read(folder/'process_audit.json');bound=False
    if len(events)<2:raise ValueError('No full-window process observations')
    for event in events:
        clients=set(event['clients'])
        if clients-{report['pid']}:raise ValueError('Foreign MPS client')
        if clients:bound=True
        elif bound:raise ValueError('Own MPS client disappeared')
        if any(int(r[0]) not in {report['pid'],*event['servers']} or r[1].strip()!=GPU for r in event['processes']):
            raise ValueError('Foreign A100 compute process')
    if not bound:raise ValueError('Own MPS client never observed')
    return dict(status='passed',repeat=report['repeat'],report_sha256=sha(folder/'report.json'),
        calls_sha256=sha(folder/'calls.csv'),outputs_sha256=sha(folder/'retrieval_outputs.npz'),
        original_bytes_sha256=sha(folder/'original_image_bytes.npz'),offline_FP32_bank_sha256=report['bank_sha256'],
        continuous_wall_s=elapsed,internal199_gaps_s=gaps,board_energy_j=board,system_energy_j=energy,
        mean_board_power_w=board/elapsed,common_input_tokens200=tokens,measured_first200_hit1=quality,
        measured_first200_mrr=sum(1/r for r in ranks)/200,full2400_historical_reference_hit1=.7358333333333333)
