"""Original .7358333 Qwen image query, file input to CPU full100 ranking.

An independent fresh process builds the original B1 FP32 title bank offline.
It never serves as a warmup of any formal query process. No FP16 saved bank is
silently substituted, and no original model, crop, prompt or sorting is changed.
"""
from __future__ import annotations
import argparse
import ast
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
from abo_query_contract import (BASELINE_QUERY,DOCUMENT_INSTRUCTION,MODEL_CONFIG_SHA,
    MODEL_WEIGHTS_SHA,load_contract,timed_calls)
from measure_abo_text_qwen import GPU,PROTOCOL,TASK,sha,write,model_assets

ORIGINAL_COMMIT='b9a4be689f198f55ee0caaf4e5bcddd4dfc80200'
ORIGINAL_PATH='LightGenV2/tasks/t08_abo_image_text_retrieval/baseline_5090d.py'
ORIGINAL_SHA='36adc669a58cea060719e48013e5c12b66a115aee928d88578652e2c49d2eb70'
FEATURE_PATH='experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/features.py'
FEATURE_SHA='6a8628c2ee1506cff2f25f33d93eb35d6485cc0dfce1bb12f809c48392c5b5da'
REFERENCE=REPO/'LightGenV2/reports/20260914_latest_qwen_baselines_a100_FIRSTBLOCK_FINAL/evidence/performance_sources/T08_abo_image_text_baseline_report.json'


def original_source():
    raw=subprocess.check_output(['git','show',ORIGINAL_COMMIT+':'+ORIGINAL_PATH],cwd=REPO)
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA:raise ValueError('Original baseline source differs')
    current=ast.parse((REPO/ORIGINAL_PATH).read_text(encoding='utf8'));old=ast.parse(raw)
    for name in ('_inputs','_template','_embed_title'):
        a=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False):
            raise ValueError('Original inference helper changed: '+name)
    if sha(REPO/FEATURE_PATH)!=FEATURE_SHA:raise ValueError('Original teacher feature math changed')
    return raw.decode('utf8')


def source_map():
    from measure_abo_text_qwen import source_map as common_sources
    return {**common_sources(),str(Path(__file__).relative_to(REPO)):sha(__file__)}


def prepare(processor,path):
    from PIL import Image,ImageOps
    from LightGenV2.tasks.t08_abo_image_text_retrieval.baseline_5090d import _inputs
    with Image.open(path) as source:image=ImageOps.exif_transpose(source).convert('RGB')
    cpu=_inputs(processor,image=image,instruction=BASELINE_QUERY['image_to_text'])
    return cpu,image


def descriptor(model,inputs):
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import teacher_embeddings
    return teacher_embeddings(model,inputs,2048)[0]


def validate_bank(arrays,receipt,titles):
    import numpy as np
    bank=arrays['vectors']
    if (receipt['status']!='complete' or receipt['purpose']!='offline_original_B1_FP32_title_bank'
            or receipt['model_weights_sha256']!=MODEL_WEIGHTS_SHA
            or receipt['model_config_sha256']!=MODEL_CONFIG_SHA
            or receipt['original_source_sha256']!=ORIGINAL_SHA
            or receipt['instruction']!=DOCUMENT_INSTRUCTION or receipt['batch']!=1
            or receipt['product_ids']!=[r['product_id'] for r in titles]
            or receipt['titles']!=[r['title'] for r in titles]
            or bank.shape!=(100,2048) or bank.dtype!=np.float32
            or not np.isfinite(bank).all() or not np.allclose(np.linalg.norm(bank,axis=1),1.,atol=2e-6)):
        raise ValueError('Not the original normalized FP32 title bank')
    return bank


def load(args,device,with_bank=True):
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    original_source()
    titles,_,test=load_contract(args.data)
    reference=json.loads(REFERENCE.read_text())
    if (reference['script_sha256']!=ORIGINAL_SHA or reference['git_commit']!=ORIGINAL_COMMIT
            or reference['model_safetensors_sha256']!=MODEL_WEIGHTS_SHA
            or reference['performance']['recall_at_1']!=.7358333333333333):
        raise ValueError('Wrong original full2400 reference')
    weights=list(args.model.glob('*.safetensors'))
    if sha(args.model/'config.json')!=MODEL_CONFIG_SHA or len(weights)!=1 or sha(weights[0])!=MODEL_WEIGHTS_SHA:
        raise ValueError('Original Embedding2B weights/config mismatch')
    bank=None
    if with_bank:
        import numpy as np
        receipt=json.loads((args.bank/'receipt.json').read_text())
        if sha(args.bank/'title_bank.npz')!=receipt['bank_sha256']:raise ValueError('Offline FP32 bank SHA mismatch')
        bank=validate_bank(np.load(args.bank/'title_bank.npz',allow_pickle=False),receipt,titles)
        if receipt['sources']!=source_map():raise ValueError('Offline bank source identity differs')
    processor=AutoProcessor.from_pretrained(str(args.model),local_files_only=True,min_pixels=50176,max_pixels=50176)
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(args.model),local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    return model,processor,bank,titles,test


def cpu_contract(args):
    """Use two real TEST images outside the timed200 to avoid selective cache warming."""
    import torch
    from PIL import Image,ImageOps
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import teacher_embeddings
    torch.set_num_threads(8)
    model,processor,bank,titles,test=load(args,'cpu')
    tree=ast.parse(original_source())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_template','_inputs')]
    scope=dict(Any=object,torch=torch,Image=Image,IGNORED_MODEL_INPUTS={'token_type_ids','mm_token_type_ids'})
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'original_image_helpers','exec'),scope)
    checks=[]
    with torch.inference_mode():
        for index in (200,2399):
            path=args.data/test[index]['image_path']
            with Image.open(path) as source:original_image=ImageOps.exif_transpose(source).convert('RGB')
            original_inputs=scope['_inputs'](processor,image=original_image,instruction=BASELINE_QUERY['image_to_text'])
            actual,image=prepare(processor,path)
            if set(actual)!=set(original_inputs) or any(not torch.equal(v,original_inputs[k]) for k,v in actual.items()):
                raise ValueError('Original file/EXIF/processor input differs')
            a=teacher_embeddings(model,original_inputs,2048)[0];b=descriptor(model,actual)
            if not torch.equal(a,b):raise ValueError('Original full image neural forward differs')
            sa=a.float()@torch.from_numpy(bank).T;sb=b.float()@torch.from_numpy(bank).T
            if not torch.equal(sa,sb) or not torch.equal(sa.argsort(descending=True),sb.argsort(descending=True)):
                raise ValueError('Original FP32 cosine or non-stable sorting differs')
            checks.append(dict(test_index=index,sample_id=test[index]['sample_id'],image_sha256=sha(path),
                image_size=list(image.size),input_shapes={k:list(v.shape) for k,v in actual.items()},bitwise_equal=True))
    if torch.cuda.is_initialized():raise ValueError('CPU contract initialized CUDA')
    args.output.mkdir(parents=True)
    write(args.output/'contract.json',dict(status='passed',mode='original_real_image_full_CPU_forward_bitwise',
        no_gpu_inference=True,checks=checks,sources=source_map(),model_assets=model_assets(args.model),
        bank_sha256=sha(args.bank/'title_bank.npz'),bank_receipt_sha256=sha(args.bank/'receipt.json'),
        original_source_sha256=ORIGINAL_SHA,reference_sha256=sha(REFERENCE)))
    print(json.dumps(dict(status='passed',real_images=2,no_gpu_inference=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bank',type=Path);p.add_argument('--contract',type=Path)
    p.add_argument('--offline-bank',action='store_true');p.add_argument('--cpu-contract',action='store_true')
    p.add_argument('--repeat',type=int,choices=(1,2,3));args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable original-task output required')
    if args.cpu_contract:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        return cpu_contract(args)
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100'
            or (not args.offline_bank and (args.repeat is None or args.contract is None))):
        raise ValueError('Original GPU/private MPS100, repeat and CPU contract required')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import write_csv
    engine.GPU,engine.PIPE=GPU,os.environ['CUDA_MPS_PIPE_DIRECTORY']
    bound=source_map()
    class Guard(engine.MPSGuard):
        def check(self):
            if source_map()!=bound:raise ValueError('Frozen original inference source changed')
            super().check()
    guard=Guard(SimpleNamespace(deadline=time.time()+1800));guard.check()
    import numpy as np
    import torch
    torch.set_num_threads(8)
    props=torch.cuda.get_device_properties(0)
    cap=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
    if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:
        raise ValueError('Physical GPU or cap differs')
    args.output.mkdir(parents=True);sampler=None;telemetry=[];calls=[];vectors=[];scores_rows=[];rankings=[]
    identity=dict(protocol_id=PROTOCOL,task='ABO image-to-title retrieval',system='qwen',repeat=args.repeat,
        pid=os.getpid(),gpu_uuid=GPU,mps_percent=100,batch=1,visible_multiprocessors=108,power_limit_w=cap,
        original_source_sha256=ORIGINAL_SHA,original_source_commit=ORIGINAL_COMMIT,reference_sha256=sha(REFERENCE),
        model_config_sha256=MODEL_CONFIG_SHA,model_weights_sha256=MODEL_WEIGHTS_SHA,sources=bound,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=sys.argv,
        query_instruction=BASELINE_QUERY['image_to_text'],document_instruction=DOCUMENT_INSTRUCTION,
        descriptor_dimensions=2048,ranking_stable=False,score_device='cuda',score_dtype='FP32',
        first_sample_retained=True,explicit_warmup_forwards=0,full2400_reference_hit1=.7358333333333333,
        timer='original file input to200th CPU complete100 ranking; all199 genuine gaps included')
    def status(state,**extra):write(args.output/'status.json',dict(status=state,pid=os.getpid(),samples=len(calls),**extra))
    status('loading');write(args.output/'identity.json',identity)
    try:
        with torch.inference_mode():
            model,processor,bank,titles,test=load(args,'cuda:0',with_bank=not args.offline_bank)
            identity['model_assets']=model_assets(args.model)
            if not args.offline_bank:
                contract=json.loads(args.contract.read_text())
                if (contract['status']!='passed' or contract['sources']!=bound or contract['model_assets']!=identity['model_assets']
                        or contract['bank_sha256']!=sha(args.bank/'title_bank.npz')):
                    raise ValueError('Actual original CPU contract is not bound')
                identity.update(bank_sha256=contract['bank_sha256'],bank_receipt_sha256=sha(args.bank/'receipt.json'),
                    cpu_contract_sha256=sha(args.contract),cpu_contract=str(args.contract),offline_title_bank=str(args.bank))
                gallery=torch.from_numpy(bank).to('cuda:0')
            write(args.output/'identity.json',identity)
            guard.bind();write(args.output/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=bound))
            if args.offline_bank:
                from LightGenV2.tasks.t08_abo_image_text_retrieval.baseline_5090d import _embed_title
                loaded=SimpleNamespace(model=model,processor=processor,device=torch.device('cuda:0'))
                for title in titles:
                    guard.fast();vectors.append(_embed_title(loaded,title['title']).detach().float())
                torch.cuda.synchronize();guard.check()
                np.savez_compressed(args.output/'title_bank.npz',vectors=torch.stack(vectors).cpu().numpy())
                write(args.output/'receipt.json',dict(status='complete',purpose='offline_original_B1_FP32_title_bank',
                    instruction=DOCUMENT_INSTRUCTION,batch=1,product_ids=[r['product_id'] for r in titles],
                    titles=[r['title'] for r in titles],bank_sha256=sha(args.output/'title_bank.npz'),
                    model_config_sha256=MODEL_CONFIG_SHA,model_weights_sha256=MODEL_WEIGHTS_SHA,
                    original_source_sha256=ORIGINAL_SHA,sources=bound,source_commit=identity['source_commit'],
                    not_a_formal_timing_trial=True,no_query_forwards_in_this_process=True))
                status('complete');print(json.dumps(dict(status='complete',offline_titles=100)));return
            raw=args.output/'telemetry_stdout.jsonl'
            sampler=ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw));sampler.start()
            sampler.set_phase('continuous200_file_to_CPU_ranking_no_warmup');time.sleep(.15)
            for call in timed_calls('image_to_text',titles,test):
                guard.fast();sampler.check();row=call['row'];path=args.data/row['image_path']
                start=time.perf_counter();cpu,image=prepare(processor,path);prepared=time.perf_counter()
                inputs={k:v.to('cuda:0',non_blocking=True) for k,v in cpu.items()};model_start=time.perf_counter()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();vector=descriptor(model,inputs);scores=vector.float()@gallery.float().T
                ranking=scores.argsort(descending=True);end.record();host=ranking.cpu().tolist()
                predicted=titles[host[0]]['product_id'];returned=time.perf_counter();end.synchronize()
                calls.append(dict(index=call['index'],sample_id=row['sample_id'],product_id=row['product_id'],
                    label=int(row['label']),image_path=row['image_path'],input_start=start,prepare_end=prepared,
                    model_start=model_start,cpu_result_return=returned,diagnostic_full_model_search_cuda_ms=begin.elapsed_time(end),
                    predicted_product_id=predicted,hit1=int(predicted==row['product_id']),
                    input_ids=cpu['input_ids'].tolist(),attention_mask=cpu['attention_mask'].tolist(),
                    valid_input_tokens=int(cpu['attention_mask'].sum()),input_shapes={k:list(v.shape) for k,v in cpu.items()},
                    image_grid_thw=cpu['image_grid_thw'].tolist(),
                    decoded_image_size=list(image.size),decoded_RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                vectors.append(vector.detach());scores_rows.append(scores.detach());rankings.append(host)
                if call['index']==0 or (call['index']+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
            a,b=calls[0]['input_start'],calls[-1]['cpu_result_return'];energy=interval_energy([(a,b)],telemetry,338.2)
            guard.check();load_contract(args.data)
            originals={}
            for i,row in enumerate(test[:200]):
                contents=(args.data/row['image_path']).read_bytes();calls[i]['image_file_sha256']=hashlib.sha256(contents).hexdigest()
                originals['image'+str(i)]=np.frombuffer(contents,dtype=np.uint8)
            np.savez_compressed(args.output/'original_image_bytes.npz',**originals)
            np.savez_compressed(args.output/'retrieval_outputs.npz',vectors=torch.stack(vectors).cpu().numpy(),
                scores=torch.stack(scores_rows).cpu().numpy(),rankings=np.asarray(rankings,dtype=np.int64))
            if source_map()!=bound or model_assets(args.model)!=identity['model_assets'] or sha(args.bank/'title_bank.npz')!=identity['bank_sha256']:
                raise ValueError('Source/model/offline bank changed')
            report=dict(status='complete',**identity,measured_calls=200,unique_queries=200,continuous_wall_s=b-a,
                power_integration=energy,total_system_energy_j=energy['energy_j'],
                mean_board_power_w=(energy['energy_j']-338.2*(b-a))/(b-a),
                measured_first200_hit1=sum(c['hit1'] for c in calls)/200,
                common_reference_tokens=sum(c['valid_input_tokens'] for c in calls))
            write(args.output/'report.json',report);status('complete')
            print(json.dumps({k:report[k] for k in ('status','measured_calls','measured_first200_hit1','continuous_wall_s','mean_board_power_w')}))
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(args.output/'telemetry.csv',telemetry)
        if calls:write_csv(args.output/'calls.csv',calls)
        if scores_rows and not (args.output/'retrieval_outputs.npz').exists():
            np.savez_compressed(args.output/'retrieval_outputs_partial.npz',vectors=torch.stack(vectors).cpu().numpy(),
                scores=torch.stack(scores_rows).cpu().numpy(),rankings=np.asarray(rankings,dtype=np.int64))
        guard.close(args.output/'process_audit.json')
        write(args.output/'files_manifest.json',[dict(file=f.name,sha256=sha(f),bytes=f.stat().st_size)
            for f in sorted(args.output.iterdir()) if f.is_file() and f.name!='files_manifest.json'])


if __name__=='__main__':main()
