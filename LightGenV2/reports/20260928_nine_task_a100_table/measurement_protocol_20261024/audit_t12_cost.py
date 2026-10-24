"""Independent no-model T12 original RGB/UTF8, outputs, spans and power audit."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_t12_cost import GPU,PROTOCOL,FULL_TEST_PSNR,proof_contract,check_platform,sha,write
from t12_reference_files import load_bundle,verify_original_bytes,TEST_SHA,validate_records
from audit_abo_title_student import ledger_key


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-7):raise ValueError(('Recomputed value differs',a,b))
def mapped(recorded):
    prefix='/DATA/DATA1/guest3/2026OpticsMoE/'
    text=str(recorded).replace('\\','/')
    if not text.startswith(prefix):raise ValueError('Unknown original artifact root')
    relative=text[len(prefix):]
    if '..' in Path(relative).parts or not relative.startswith('LightGenV2/tasks/t12_text_to_image/runs/simulation/'):
        raise ValueError('Artifact outside original T12 runs')
    return REPO/relative


def source_blobs(identity):
    for name,digest in identity['sources'].items():
        blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original published source differs: '+name)


def audit_inputs(root,original_test=None):
    from PIL import Image
    root=Path(root);value=load_bundle(root);verify_original_bytes(root,value);source_blobs(value)
    if original_test is None:
        local=root/'original_test_manifest.jsonl'
        original_test=local if local.exists() else Path('/DATA/DATA1/guest3/t12_assets/datasets/abo_cleanrender_lamp_table_pillow_256_v1/test.jsonl')
    if sha(original_test)!=TEST_SHA:raise ValueError('Original full TEST source manifest differs')
    test=[json.loads(line) for line in Path(original_test).read_text(encoding='utf-8-sig').splitlines()]
    test=[r for r in test if r['category'] in ('lamp','table','pillow')]
    if len(test)!=192:raise ValueError('Original TEST192 product sources/2304 instruction pairs required')
    normalized=[]
    for i,r in enumerate(value['records']):
        offset=i%12;mode='background' if offset<4 else 'object' if offset<8 else 'joint'
        if r['sample_id']!=f"{test[i//12]['sample_id']}:{mode}:{offset}":raise ValueError('Original first200 order differs')
        with Image.open(root/r['image']) as image:
            if image.size!=(256,256) or image.mode!='RGB':raise ValueError('RGB256 input required')
            pixels=np.asarray(image,dtype=np.uint8)
            x=pixels.astype(np.float32)/np.float32(127.5)-np.float32(1)
        text=(root/r['text']).read_bytes().decode('utf8')
        if not text or not np.isfinite(x).all():raise ValueError('Invalid original input')
        normalized.append(hashlib.sha256(np.transpose(x,(2,0,1)).copy().tobytes()).hexdigest())
    return dict(status='passed',samples=200,reference_png_files=200,UTF8_files=200,
        input_manifest_sha256=sha(root/'input_manifest.json'),original_test_manifest_sha256=TEST_SHA,
        normalized_FP32_reference_sha256=normalized,no_model_or_GPU_forward=True,
        server_original_reference_bitwise_checks=all(r['reference_bitwise_equal'] for r in value['records']),
        not_an_independent_neural_replay=True)


def audit(root):
    from LightGenV2.tasks.t06_video_quality_assessment.reports.strict_tf_resource_audit_20261020 import observed_energy
    root=Path(root);identity=read(root/'identity.json');summary=read(root/'summary.json');state=read(root/'status.json')
    system=identity['system']
    if (system not in ('ours','qwen') or identity['protocol_id']!=PROTOCOL or state['status']!='complete'
            or state['samples']!=200 or identity['gpu_uuid']!=GPU or identity['mps_percent']!=100
            or identity['visible_multiprocessors']!=108 or abs(identity['power_limit_w']-250)>.1
            or identity['batch']!=1 or identity['explicit_warmup_forwards']!=0 or not identity['first_sample_retained']):
        raise ValueError('Original T12 platform/cost contract differs')
    if summary['status']!='complete' or summary['valid_samples']!=200:raise ValueError('Incomplete summary')
    for item in read(root/'files_manifest.json'):
        p=root/item['file']
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Sealed raw changed')
    source_blobs(identity)
    proofpath=mapped(identity['cpu_contract']);proof=proof_contract(proofpath,system,check_sources=False)
    if sha(proofpath)!=identity['cpu_contract_sha256']:raise ValueError('Original CPU proof changed')
    source_blobs(proof)
    if system=='ours' and (identity['floating_parameter_dtypes']!={'model':['torch.float32']}
            or identity['native_embedding_dtype']!='torch.bfloat16'):
        raise ValueError('Original FP32 small model/native BF16 embedding differs')
    inputs=mapped(identity['input_root']);bundle=load_bundle(inputs);verify_original_bytes(inputs,bundle)
    if sha(inputs/'input_manifest.json')!=identity['input_manifest_sha256']:raise ValueError('Original input manifest changed')
    source_blobs(bundle)
    calls=rows(root/'calls.csv');stages=rows(root/'stages.csv')
    out=np.load(root/'RGB_outputs.npz',allow_pickle=False)['predictions']
    if out.shape!=(200,3,256,256) or out.dtype!=np.float32 or not np.isfinite(out).all() or out.min()<0 or out.max()>1:
        raise ValueError('Full original float CPU RGB outputs missing')
    if len(calls)!=200:raise ValueError('Complete first200 required')
    previous=-math.inf;gaps=0.;tokens=0
    for i,(call,record) in enumerate(zip(calls,bundle['records'])):
        a,b=float(call['input_start']),float(call['cpu_result_return'])
        if int(call['index'])!=i or call['sample_id']!=record['sample_id'] or not previous<=a<b:
            raise ValueError('Original order/continuous window differs')
        gap=0. if i==0 else a-previous;close(gap*1000,call['internal_gap_ms']);gaps+=gap;previous=b
        ids=json.loads(call['input_ids']);mask=json.loads(call['attention_mask'])
        if (len(ids)!=1 or len(mask)!=1 or len(ids[0])!=len(mask[0]) or not 0<len(ids[0])<=64
                or set(mask[0])-{0,1} or sum(mask[0])!=int(call['valid_input_tokens'])
                or json.loads(call['output_shape'])!=[3,256,256]):raise ValueError('Actual B1 native token/RGB evidence differs')
        tokens+=sum(mask[0])
    elapsed=float(calls[-1]['cpu_result_return'])-float(calls[0]['input_start'])
    board=observed_energy([(float(calls[0]['input_start']),float(calls[-1]['cpu_result_return']))],rows(root/'telemetry.csv'))
    intervals=defaultdict(list);charged=0.;counts=Counter()
    for r in stages:
        index=int(r['sample_index']);a,b=float(r['start_monotonic']),float(r['end_monotonic'])
        if index not in range(200) or not float(calls[index]['input_start'])<=a<=b<=float(calls[index]['cpu_result_return']):
            raise ValueError('Observed original stage outside request')
        close((b-a)*1000,r['synchronized_wall_ms']);intervals[index].append((a,b));counts[r['stage']]+=1
        if r['charged']=='True':charged+=float(r['synchronized_wall_ms'])
        if r['family'] in ('parallel_residual','fusion','layout','simulation_only') and r['charged']!='False':
            raise ValueError('Excluded projection component charged')
        if r['family'] not in ('input','output') and r['cuda_event_ms']=='':raise ValueError('Paired original NN CUDA Event missing')
    for spans in intervals.values():
        if any(a[1]>b[0]+1e-8 for a,b in zip(spans,spans[1:])):raise ValueError('Overlapping summed neural spans')
    if counts['input.reference_RGB_UTF8_decode_H2D']!=200 or counts['output.CPU_RGB_clamp']!=200:
        raise ValueError('Inclusive original frontend/output missing')
    if system=='ours':
        for mode in ('language','vision'):
            if counts[mode+'.router_simulation']!=200 or counts[mode+'.expert_global_simulation']!=400:
                raise ValueError('Original six optical propagation counts differ')
        if {ledger_key(v) for v in read(root/'stage_ledger.json')}!={ledger_key(v) for v in proof['stage_ledger']}:
            raise ValueError('Original CPU-proven selected NN source spans differ')
        charged+=gaps*1000;optical=200*6*1.0447
        seconds=(charged+optical)/1000;joules=(121.746*optical+104.230*charged)/1000
        close(charged,summary['charged_electronic_ms_total']);close(optical,summary['optical_ms_total'])
        if summary['common_input_tokens200'] is not None:raise ValueError('Qwen common-token pairing not yet independently bound')
    else:
        if counts['baseline.full_original_Qwen28_UNet_VAE']!=200:raise ValueError('Full original baseline invocation missing')
        seconds=elapsed;joules=338.2*elapsed+board;optical=0.;charged=None
        close(tokens,summary['common_input_tokens200'])
    close(seconds,summary['seconds200']);close(joules,summary['joules200'])
    close(seconds*5,summary['sample_ms']);close(joules/200,summary['sample_j'])
    close(elapsed,summary['actual_software_wall_seconds']);close(board,summary['diagnostic_software_board_j200'])
    close(board/elapsed,summary['weighted_software_board_w']);close(FULL_TEST_PSNR[system],summary['formal_full2304_reference_PSNR'])
    events=read(root/'process_audit.json');bound=False
    if len(events)<2:raise ValueError('Missing admission/full-window process audit')
    anchor=identity['clock_anchor'];offset=anchor['unix']-anchor['monotonic']
    if (events[0]['unix']>float(calls[0]['input_start'])+offset
            or events[-1]['unix']<float(calls[-1]['cpu_result_return'])+offset):
        raise ValueError('Process/platform evidence does not cover full measured window')
    for event in events:
        platform=event['platform']
        check_platform(','.join(str(platform[k]) for k in ('uuid','name','power_limit_w','memory_total_mib')))
        clients=set(event['clients'])
        if clients-{identity['pid']}:raise ValueError('Foreign private MPS client')
        if clients:bound=True
        elif bound:raise ValueError('Own MPS client disappeared')
        if any(int(p[0]) not in {identity['pid'],*event['servers']} or p[1].strip()!=GPU for p in event['processes']):
            raise ValueError('Foreign A100 GPU process')
    if not bound:raise ValueError('Own MPS process never observed')
    shapes=read(root/'nn_module_shapes.json')
    if {r['index'] for r in shapes}!=set(range(200)):raise ValueError('Matching original actual NN shape records absent')
    return dict(status='passed',system=system,repeat=identity['repeat'],calls=200,sample_ms=seconds*5,sample_j=joules/200,
        seconds200=seconds,joules200=joules,software_wall_s=elapsed,internal199_gaps_s=gaps,
        diagnostic_software_board_j200=board,charged_electronic_ms_total=charged,optical_ms_total=optical,
        report_sha256=sha(root/'summary.json'),outputs_sha256=sha(root/'RGB_outputs.npz'),
        original_input_manifest_sha256=identity['input_manifest_sha256'],CPU_contract_sha256=identity['cpu_contract_sha256'],
        native_valid_input_tokens200=tokens,formal_full2304_reference_PSNR=FULL_TEST_PSNR[system],
        common_work_requires_matching_actual_Qwen_ID_mask_pairing=True,no_model_forward=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--inputs-only',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve independent audit receipt')
    write(args.output,audit_inputs(args.folder) if args.inputs_only else audit(args.folder))
