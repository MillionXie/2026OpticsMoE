"""Original .8058 image query; independent original100-title bank, no training.

The numerical graph/observer already passed held-out real images200/2399 and
titles0/99 on CPU. The bank is a separate fresh process, never query warmup.
Full TEST reference accuracy stays fixed; first200 predictions only audit cost.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
from abo_image_student_graph import BODY_SHA,load,patch,imported_sources
from measure_abo_text_qwen import TASK,GPU,PROTOCOL,sha,write
from measure_abo_title_student import rows
from measure_salicon import shared

CPU=TASK/'runs/simulation/a100_image_student_contract_20261025_01/original_image_title_CPU_contract/contract.json'
QWEN=TASK/'runs/simulation'


def cpu_contract():
    value=json.loads(CPU.read_text())
    if (value['status']!='passed' or not value['no_gpu'] or value['identity']['body_sha256']!=BODY_SHA
            or [r.get('test_index') for r in value['checks'][:2]]!=[200,2399]
            or [r.get('title_index') for r in value['checks'][2:]]!=[0,99]
            or not all(r['bitwise_equal'] for r in value['checks'])):raise ValueError('Original real-image/title CPU contract missing')
    for name,digest in value['sources'].items():
        if sha(REPO/name)!=digest:raise ValueError('Previously verified original CPU sources changed: '+name)
    return value


def validate_bank(root,titles):
    import numpy as np
    root=Path(root);receipt=json.loads((root/'bank_receipt.json').read_text());state=json.loads((root/'status.json').read_text())
    if (state['status']!='complete' or receipt['status']!='complete' or receipt['titles']!=100
            or receipt['batch']!=1 or receipt['body_sha256']!=BODY_SHA or receipt['cpu_contract_sha256']!=sha(CPU)
            or receipt['query_or_dummy_calls']!=0 or receipt['gpu_uuid']!=GPU or receipt['mps_percent']!=100
            or sha(root/'title_bank.npz')!=receipt['bank_sha256']):raise ValueError('Separate original title bank not bound')
    value=np.load(root/'title_bank.npz',allow_pickle=False)
    if (value['vectors'].shape!=(100,64) or value['vectors'].dtype!=np.float32
            or not np.isfinite(value['vectors']).all() or value['product_ids'].tolist()!=[r['product_id'] for r in titles]
            or value['labels'].tolist()!=list(range(100))):raise ValueError('Original100-title FP32 bank differs')
    return value['vectors'].copy(),receipt


def common_work(reference,test):
    if len(reference)!=200:raise ValueError('Original complete common Qwen input work required')
    for i,(a,b) in enumerate(zip(reference,test)):
        if int(a['index'])!=i or a['sample_id']!=b['sample_id'] or a['product_id']!=b['product_id']:
            raise ValueError('Different original first200 query order')
    values=[int(r['valid_input_tokens']) for r in reference]
    if sum(values)!=13728 or min(values)<1:raise ValueError('Actual native common token work differs')
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','model','data','checkpoint'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=('bank','query'),required=True)
    parser.add_argument('--bank',type=Path);parser.add_argument('--repeat',type=int,choices=(1,2,3))
    args=parser.parse_args();output=args.output.resolve()
    if output.exists() or not output.is_relative_to(TASK/'runs/simulation'):raise ValueError('Fresh immutable task run required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100':
        raise ValueError('Physical A100 UUID/private MPS100 required')
    proof=cpu_contract()
    from audit_abo_text_qwen import original_csv_contract
    titles,_,test=original_csv_contract(args.data)
    vectors=receipt=None;reference=[];common=[];anchor=None
    if args.mode=='query':
        if args.repeat is None or args.bank is None:raise ValueError('Original separate bank and repeat required')
        vectors,receipt=validate_bank(args.bank,titles)
        parent='a100_image_input_inclusive_20261024_02' if args.repeat<3 else 'a100_image_input_inclusive_20261024_03'
        anchor=QWEN/parent/f'qwen_repeat{args.repeat:02d}/calls.csv'
        reference=rows(anchor);common=common_work(reference,test[:200])
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    engine.GPU=GPU;engine.PIPE=os.environ['CUDA_MPS_PIPE_DIRECTORY']
    guard=engine.MPSGuard(SimpleNamespace(deadline=time.time()+1800));guard.check()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from LightGenV2.tasks.t08_abo_image_text_retrieval import optical_moe as original
    torch.set_num_threads(8);output.mkdir(parents=True)
    meter=shared.meter_lib.StageMeter(torch.cuda);replacement=None;sampler=None;telemetry=[];calls=[]
    saved_vectors=[];saved_scores=[];saved_orders=[]
    def status(value,**more):write(output/'status.json',dict(status=value,pid=os.getpid(),samples=len(calls),**more))
    status('loading')
    try:
        with torch.inference_mode():
            loaded,replacement,readout,settings,data,identity=load(args,'cuda:0')
            if identity['model_assets']!=proof['identity']['model_assets'] or settings.dtype!='bfloat16':
                raise ValueError('Original CPU-bound model/frontend changed')
            props=torch.cuda.get_device_properties(0)
            cap=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
            if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:raise ValueError('Wrong A100/cap')
            guard.bind()
            if args.mode=='bank':
                frozen=imported_sources();status('encoding_original100_titles')
                # Handoff the loading monitor to the already-bound independent
                # worker guard, exactly as query mode does before its window.
                # Avoid duplicate parent control queries throughout bank encoding.
                write(output/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=frozen,
                    mode='bank',independent_worker_guard_bound=True))
                value=original._encode_titles_student(loaded,replacement,readout,data.titles,settings)
                guard.fast()  # A prior transient/foreign guard failure stays fatal.
                guard.check()
                if value.shape!=(100,64) or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Original title output is not FP32[100,64]')
                if imported_sources()!=frozen:raise ValueError('Original source changed during bank')
                np.savez_compressed(output/'title_bank.npz',vectors=value.numpy(),product_ids=np.asarray([r['product_id'] for r in titles]),labels=np.arange(100))
                write(output/'bank_receipt.json',dict(status='complete',titles=100,batch=1,body_sha256=BODY_SHA,
                    cpu_contract_sha256=sha(CPU),bank_sha256=sha(output/'title_bank.npz'),query_or_dummy_calls=0,
                    gpu_uuid=GPU,mps_percent=100,identity=identity,sources=frozen,
                    not_a_formal_timing_or_energy_run=True,original_function='optical_moe._encode_titles_student, original DOCUMENT_INSTRUCTION',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
                status('complete',titles=100);return
            bank=F.normalize(torch.from_numpy(vectors),dim=-1)
            patch(loaded,replacement,readout,meter);frozen=imported_sources()
            identity.update(protocol_id=PROTOCOL,system='ours',repeat=args.repeat,pid=os.getpid(),gpu_uuid=GPU,mps_percent=100,
                batch=1,power_limit_w=cap,visible_multiprocessors=108,sources=frozen,command=sys.argv,
                cpu_contract=str(CPU),cpu_contract_sha256=sha(CPU),bank_root=str(args.bank),bank_sha256=receipt['bank_sha256'],
                bank_receipt_sha256=sha(args.bank/'bank_receipt.json'),common_Qwen_calls=str(anchor),common_Qwen_calls_sha256=sha(anchor),
                score_device='cpu',score_dtype='FP32',ranking_stable=False,descriptor_dimensions=64,
                first_sample_retained=True,explicit_warmup_forwards=0,online_title_bank_forwards=0,stage_ledger=meter.ledger,
                original_image_transform='RGB centered bicubic ImageOps.fit224; no EXIF transpose',
                cost_scope='mixed selected-NN/parameter-optics projection; not device measured E2E')
            write(output/'identity.json',identity);write(output/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=frozen))
            raw=output/'telemetry_stdout.jsonl';sampler=ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('continuous200_real_images_zero_warmup');time.sleep(.15)
            for index,sample in enumerate(data.test[:200]):
                guard.fast();sampler.check();meter.sample_index=index;start=time.perf_counter()
                # These are exactly image_forward's original frontend statements;
                # only synchronization/scopes are added, no arithmetic replaced.
                with meter.scope('input.original_RGB_fit224_processor_h2d','input',gpu=False):
                    image=original.GroceryRetrievalDataset([sample],settings.image_size,augment=False)[0]['image']
                    inputs=original.preprocess_images(loaded.processor,[image],settings.instruction)
                    original.validate_token_budgets(inputs,settings)
                    moved=original.move_inputs(inputs,loaded.device);torch.cuda.synchronize()
                ready=time.perf_counter()
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=settings.amp_enabled):
                    embedding,features=original.student_embeddings(loaded.model,replacement,readout,moved)
                with meter.scope('output.cpu_embedding','output',gpu=False):vector=embedding[0].detach().float().cpu()
                with meter.scope('output.cpu_original_cosine_nonstable_rank','output',gpu=False):
                    query=F.normalize(vector[None],dim=-1);scores=(query@bank.T)[0]
                    ranking=scores.argsort(descending=True);host=ranking.tolist();predicted=titles[host[0]]['product_id']
                returned=time.perf_counter();gap=0. if not calls else 1000*(start-calls[-1]['cpu_result_return'])
                calls.append(dict(index=index,sample_id=sample.sample_id,product_id=sample.sku_name,label=sample.sku_index,
                    input_start=start,prepare_end=ready,cpu_result_return=returned,internal_gap_ms=gap,
                    predicted_product_id=predicted,hit1=int(predicted==sample.sku_name),input_shapes={k:list(v.shape) for k,v in inputs.items()},
                    input_ids=inputs['input_ids'].tolist(),attention_mask=inputs['attention_mask'].tolist(),
                    valid_input_tokens=int(inputs['attention_mask'].sum()),embedding_shape=list(embedding.shape),
                    detector_feature_shape=list(features.shape),common_reference_tokens=common[index]))
                saved_vectors.append(query[0].numpy().copy());saved_scores.append(scores.numpy().copy());saved_orders.append(np.asarray(host,dtype=np.int64))
                if index==0 or (index+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
            charged=sum(r['synchronized_wall_ms'] for r in meter.rows if r['charged'])+sum(c['internal_gap_ms'] for c in calls)
            optical=200*6*1.0447;seconds=(charged+optical)/1000;energy=shared.meter_lib.optical_energy_j(optical/1000,charged/1000)
            counts=Counter(r['stage'] for r in meter.rows)
            for mode in ('vision','language'):
                if counts[mode+'.router_simulation']!=200 or counts[mode+'.expert_global_simulation']!=400:raise ValueError('Missing six original light passes')
            if counts['bridge.native_merger']!=200:raise ValueError('Actual original merger must be charged')
            guard.check()
            if imported_sources()!=frozen or sha(anchor)!=identity['common_Qwen_calls_sha256']:raise ValueError('Frozen source/common work changed')
            np.savez_compressed(output/'retrieval_outputs.npz',vectors=np.asarray(saved_vectors),scores=np.asarray(saved_scores),rankings=np.asarray(saved_orders))
            # Original file bytes/hashes are read only after the cost window.
            raw_images={};manifest=[]
            for i,sample in enumerate(data.test[:200]):
                value=sample.image_path.read_bytes();raw_images[f'JPEG_{i:03d}']=np.frombuffer(value,dtype=np.uint8)
                manifest.append(dict(index=i,sample_id=sample.sample_id,bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
            np.savez_compressed(output/'original_JPEG_bytes.npz',**raw_images);write(output/'original_input_manifest.json',manifest)
            diagnostic=interval_energy([(calls[0]['input_start'],calls[-1]['cpu_result_return'])],telemetry,0)
            write(output/'summary.json',dict(status='complete',repeat=args.repeat,valid_samples=200,
                formal_full2400_reference_hit1=.8058333333333333,first200_diagnostic_hit1=sum(c['hit1'] for c in calls)/200,
                actual_software_wall_seconds=calls[-1]['cpu_result_return']-calls[0]['input_start'],
                charged_electronic_ms_total=charged,optical_ms_total=optical,cost=shared.meter_lib.equivalent_cost(seconds,energy,200),
                common_input_tokens200=sum(common),stage_counts=dict(counts),diagnostic_software_board_integral=diagnostic))
            status('complete')
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(output/'telemetry.csv',telemetry)
        guard.close(output/'process_audit.json');meter.close()
        if replacement is not None:replacement.close()
        shared.write_csv(output/'calls.csv',calls);shared.write_csv(output/'stages.csv',meter.rows);write(output/'stage_ledger.json',meter.ledger)
        if saved_vectors and not (output/'retrieval_outputs.npz').exists():
            np.savez_compressed(output/'retrieval_outputs_partial.npz',vectors=np.asarray(saved_vectors),scores=np.asarray(saved_scores),rankings=np.asarray(saved_orders))
        write(output/'files_manifest.json',[dict(file=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(output.iterdir()) if f.is_file() and f.name!='files_manifest.json'])


if __name__=='__main__':main()
