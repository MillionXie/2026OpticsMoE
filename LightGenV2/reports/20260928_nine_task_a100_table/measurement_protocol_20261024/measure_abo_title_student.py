"""Adopted .8600 title query: real UTF8 input, selected original NN and CPU rank.

No online image path, gallery inference, dummy forward or numerical-optics
substitution. The complete original simulated forward produces every vector;
only the approved selected-stage hardware projection replaces its three light
passes. Original per-query CPU FP32 normalization/stable ranking is retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_text_qwen import TASK,GPU,PROTOCOL,sha,write
from abo_student_graph import BODY_SHA,load,patch
from abo_query_contract import timed_calls
from measure_salicon import shared

QWEN=TASK/'runs/simulation/a100_title_input_inclusive_20261024_02'
TITLE_CONTRACT=TASK/'runs/simulation/a100_title_student_contract_20261024_01/original_title_cpu_contract/contract.json'
GALLERY=TASK/'runs/simulation/a100_title_student_gallery_20261024_01'


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def sources():
    from run_abo_student_contract import source_hashes
    from prepare_abo_student_gallery import imported_sources
    return {**source_hashes(),**imported_sources(),
        str(Path(__file__).relative_to(REPO)):sha(__file__)}


def validate_gallery(root,titles,test):
    import numpy as np
    root=Path(root);receipt=json.loads((root/'receipt.json').read_text())
    controller=json.loads((root/'controller_status.json').read_text())
    if (controller['status']!='complete' or not controller.get('mps_released')
            or receipt['status']!='complete' or receipt['images']!=2400 or receipt['batch']!=1
            or receipt['identity']['body_sha256']!=BODY_SHA or receipt['saved_dtype']!='float32'
            or receipt['title_or_dummy_inference_calls']!=0 or not receipt['not_a_formal_timing_or_energy_run']
            or receipt['gpu_uuid']!=GPU or receipt['mps_percent']!=100
            or sha(root/'gallery.npz')!=receipt['gallery_sha256']):
        raise ValueError('Original independent FP32 student gallery is not complete/bound')
    gallery=np.load(root/'gallery.npz',allow_pickle=False)
    expected_ids=[r['sample_id'] for r in test];expected_products=[r['product_id'] for r in test]
    if (gallery['vectors'].shape!=(2400,64) or gallery['vectors'].dtype!=np.float32
            or not np.isfinite(gallery['vectors']).all()
            or gallery['sample_ids'].tolist()!=expected_ids or gallery['product_ids'].tolist()!=expected_products
            or gallery['labels'].tolist()!=[int(r['label']) for r in test]):
        raise ValueError('Original FP32 gallery vectors, shape or sample identities differ')
    image_contract=root/'original_gallery_image_cpu_contract/contract.json'
    cpu=json.loads(image_contract.read_text())
    if (sha(image_contract)!=receipt['gallery_image_cpu_contract_sha256'] or cpu['status']!='passed'
            or cpu['no_gpu'] is not True or [c['test_index'] for c in cpu['checks']]!=[200,2399]
            or not all(c['bitwise_equal'] for c in cpu['checks'])):
        raise ValueError('Real original gallery-image CPU contract missing')
    return gallery['vectors'].copy(),receipt


def common_work(calls,work):
    if len(calls)!=200 or len(work)!=200:raise ValueError('Original complete Qwen200-call anchor required')
    values=[]
    for call,request in zip(calls,work):
        row=request['row']
        if (int(call['index'])!=request['index'] or int(call['query_index'])!=request['unique_query_index']
                or int(call['pass_index'])!=request['pass_index'] or call['product_id']!=row['product_id']
                or call['title']!=row['title']):raise ValueError('Common Qwen token work is not the same title/order')
        values.append(int(call['valid_input_tokens']))
    if sum(values)!=9576 or min(values)<1:raise ValueError('Original per-round common Qwen token count changed')
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','model','checkpoint','data'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gallery',type=Path,default=GALLERY)
    p.add_argument('--repeat',type=int,choices=(1,2,3),required=True)
    args=p.parse_args();output=args.output.resolve()
    if output.exists() or not output.is_relative_to(TASK/'runs/simulation'):raise ValueError('New immutable title trial required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100':
        raise ValueError('Physical A100 UUID and private MPS100 required')
    from audit_abo_text_qwen import original_csv_contract
    from run_abo_student_contract import check_result,source_hashes
    titles,_,test=original_csv_contract(args.data)
    work=timed_calls('text_to_image',titles,test)
    anchor=QWEN/f'qwen_repeat{args.repeat:02d}'/'calls.csv'
    reference_calls=rows(anchor);common=common_work(reference_calls,work)
    cpu=json.loads(TITLE_CONTRACT.read_text());check_result(cpu)
    cpu_queue=json.loads((TITLE_CONTRACT.parent.parent/'suite_status.json').read_text())
    if cpu_queue['status']!='complete' or cpu_queue['sources']!=source_hashes():raise ValueError('Original title CPU source binding differs')
    vectors,receipt=validate_gallery(args.gallery,titles,test)
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    engine.GPU=GPU;engine.PIPE=os.environ['CUDA_MPS_PIPE_DIRECTORY']
    guard=engine.MPSGuard(SimpleNamespace(deadline=time.time()+1800));guard.check()
    import numpy as np
    import torch
    import transformers
    from torch.nn import functional as F
    from LightGenV2.tasks.t08_abo_image_text_retrieval import reverse_runtime as original
    torch.set_num_threads(8);output.mkdir(parents=True)
    meter=shared.meter_lib.StageMeter(torch.cuda);replacement=None;sampler=None;telemetry=[]
    calls=[];saved_vectors=[];saved_scores=[];saved_orders=[]
    def status(value,**extra):write(output/'status.json',dict(status=value,pid=os.getpid(),samples=len(calls),**extra))
    status('loading')
    try:
        with torch.inference_mode():
            loaded,replacement,readout,settings,data,prompts,identity=load(args,'cuda:0')
            if identity['model_assets']!=cpu['identity']['model_assets'] or identity['model_assets']!=receipt['identity']['model_assets']:
                raise ValueError('CPU/gallery/formal native assets differ')
            if (identity['prompt_contract']!=cpu['identity']['prompt_contract'] or settings.dtype!='bfloat16'
                    or data.sha256!=receipt['dataset_sha256']):raise ValueError('Original prompt/dtype/dataset contract changed')
            props=torch.cuda.get_device_properties(0)
            cap=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
            if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:raise ValueError('Wrong physical A100/cap')
            # Original task scores on CPU; normalization of the fixed offline
            # gallery is one-time preparation, never repeated inside queries.
            gallery=F.normalize(torch.from_numpy(vectors),dim=1)
            patch(loaded,replacement,readout,meter);frozen=sources()
            identity.update(protocol_id=PROTOCOL,system='ours',task='ABO text-to-image retrieval',repeat=args.repeat,
                pid=os.getpid(),gpu_uuid=GPU,mps_percent=100,batch=1,power_limit_w=cap,visible_multiprocessors=108,
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),sources=frozen,
                command=sys.argv,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                cpu_threads=8,gallery_root=str(args.gallery),gallery_sha256=receipt['gallery_sha256'],
                title_contract=str(TITLE_CONTRACT),title_contract_sha256=sha(TITLE_CONTRACT),
                gallery_receipt_sha256=sha(args.gallery/'receipt.json'),common_Qwen_calls=str(anchor),common_Qwen_calls_sha256=sha(anchor),
                score_device='cpu',score_dtype='FP32',ranking_stable=True,descriptor_dimensions=64,
                first_sample_retained=True,explicit_warmup_forwards=0,online_gallery_forwards=0,online_visual_forwards=0,
                stage_ledger=meter.ledger,cost_scope='input-inclusive mixed hardware projection; selected NN plus parameter-derived optics, not device-measured E2E')
            write(output/'identity.json',identity)
            guard.bind();write(output/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=frozen))
            raw=output/'telemetry_stdout.jsonl';sampler=ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('continuous200_real_titles_zero_warmup');time.sleep(.15)
            for request in work:
                guard.fast();sampler.check();index=request['index'];row=request['row'];meter.sample_index=index
                start=time.perf_counter()
                with meter.scope('input.utf8_tokenizer_h2d','input',gpu=False):
                    prepared=original._text_inputs(loaded.processor,[row['title']],prompts.title_instruction)
                    original._validate_text_token_budget(prepared,settings)
                    inputs=original.move_inputs(prepared,loaded.device);torch.cuda.synchronize()
                ready=time.perf_counter()
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=settings.amp_enabled):
                    embedding,features=original.student_embeddings(loaded.model,replacement,readout,inputs)
                with meter.scope('output.cpu_embedding','output',gpu=False):vector=embedding[0].detach().float().cpu()
                with meter.scope('output.cpu_normalize_cosine_stable_rank','output',gpu=False):
                    query=F.normalize(vector[None],dim=-1);normalized=query[0];scores=(query@gallery.T)[0]
                    ranking=scores.argsort(descending=True,stable=True);host=ranking.tolist()
                    predicted=test[host[0]]['product_id']
                returned=time.perf_counter()
                gap=0. if not calls else 1000*(start-calls[-1]['cpu_result_return'])
                calls.append(dict(index=index,query_index=request['unique_query_index'],pass_index=request['pass_index'],
                    title=row['title'],product_id=row['product_id'],input_start=start,prepare_end=ready,model_start=ready,
                    cpu_result_return=returned,internal_gap_ms=gap,predicted_product_id=predicted,hit1=int(predicted==row['product_id']),
                    input_ids=prepared['input_ids'].tolist(),attention_mask=prepared['attention_mask'].tolist(),
                    input_ids_shape=list(prepared['input_ids'].shape),valid_input_tokens=int(prepared['attention_mask'].sum()),
                    embedding_shape=list(embedding.shape),detector_feature_shape=list(features.shape),
                    common_reference_tokens=common[index],title_utf8_sha256=hashlib.sha256(row['title'].encode('utf8')).hexdigest()))
                saved_vectors.append(normalized.numpy().copy());saved_scores.append(scores.numpy().copy());saved_orders.append(np.asarray(host,dtype=np.int64))
                if index==0 or (index+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
            charged=sum(r['synchronized_wall_ms'] for r in meter.rows if r['charged'])+sum(c['internal_gap_ms'] for c in calls)
            optical_ms=200*3*1.0447;seconds=(charged+optical_ms)/1000
            counts=Counter(r['stage'] for r in meter.rows)
            if (counts['language.router_simulation']!=200 or counts['language.expert_global_simulation']!=400
                    or counts['frontend.token_embedding']!=200 or counts['head.retrieval_readout']!=200
                    or any(v for k,v in counts.items() if k.startswith(('vision.','bridge.','frontend.patch')))):
                raise ValueError('Original title-only NN/three-pass contract differs')
            energy=shared.meter_lib.optical_energy_j(optical_ms/1000,charged/1000)
            elapsed=calls[-1]['cpu_result_return']-calls[0]['input_start']
            diagnostic=interval_energy([(calls[0]['input_start'],calls[-1]['cpu_result_return'])],telemetry,0)
            for call,reference in zip(calls,reference_calls):
                if (call['input_ids']!=json.loads(reference['input_ids'])
                        or call['attention_mask']!=json.loads(reference['attention_mask'])):
                    raise ValueError('Original title input IDs/masks differ from common Qwen workload')
            guard.check()
            if sources()!=frozen or sha(anchor)!=identity['common_Qwen_calls_sha256']:raise ValueError('Frozen source/common input work changed')
            np.savez_compressed(output/'retrieval_outputs.npz',vectors=np.asarray(saved_vectors,dtype=np.float32),
                scores=np.asarray(saved_scores,dtype=np.float32),rankings=np.asarray(saved_orders,dtype=np.int64))
            summary=dict(status='complete',repeat=args.repeat,valid_samples=200,unique_titles=100,
                measured_unique100_hit1=sum(c['hit1'] for c in calls[:100])/100,full100_historical_reference_hit1=.86,
                actual_software_wall_seconds=elapsed,charged_electronic_ms_total=charged,optical_ms_total=optical_ms,
                cost=shared.meter_lib.equivalent_cost(seconds,energy,200),common_input_tokens200=sum(common),
                actual_native_input_tokens200=sum(c['valid_input_tokens'] for c in calls),stage_counts=dict(counts),
                diagnostic_software_board_integral=diagnostic,diagnostic_GPU_energy_not_used_in_phase_estimate=True)
            write(output/'summary.json',summary);status('complete');print(json.dumps(summary),flush=True)
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(output/'telemetry.csv',telemetry)
        guard.close(output/'process_audit.json');meter.close()
        if replacement is not None:replacement.close()
        shared.write_csv(output/'calls.csv',calls);shared.write_csv(output/'stages.csv',meter.rows)
        write(output/'stage_ledger.json',meter.ledger)
        if saved_vectors and not (output/'retrieval_outputs.npz').exists():
            np.savez_compressed(output/'retrieval_outputs_partial.npz',vectors=np.asarray(saved_vectors),scores=np.asarray(saved_scores),rankings=np.asarray(saved_orders))
        write(output/'files_manifest.json',[dict(file=f.name,bytes=f.stat().st_size,sha256=sha(f))
            for f in sorted(output.iterdir()) if f.is_file() and f.name!='files_manifest.json'])


if __name__=='__main__':main()
