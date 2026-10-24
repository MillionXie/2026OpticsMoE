"""Pinned .82 text-query baseline, UTF-8 input through CPU2400-ranking.

The historical online A100 profiler ranked FP32 cosine scores on CUDA (not the
bulk CPU performance producer). Keep that placement, original prompt, BF16
model.model forward, last valid hidden, and stable descending full ranking.
The original gallery is offline; no gallery or dummy inference in a trial.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from abo_query_contract import (BASELINE_QUERY, MODEL_CONFIG_SHA, MODEL_WEIGHTS_SHA,
    TEXT_BASELINE_BANK_SHA, TEXT_BASELINE_SOURCE, load_contract, timed_calls)

TASK = REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval'
GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
PROTOCOL = 'lightgen_qwen_input_inclusive_200x3_20261024_v1'
REFERENCE = TASK/'runs/simulation/text_to_image_frozen_20260920/report.json'
REFERENCE_SHA = 'e26ea454d46e9911e8355928be35bf589adae854445e0ac66ec047924a187e54'
ORIGINAL_SHA = 'ed141c360595ed677167060ef74e2853221a5f480db95097f00cdaed6225f9ea'
ONLINE_PROFILER_SHA = 'b97f14c9a8a7c17780887fbe2ab06e60177bb46b7af3d6bdede2cb8919de8a77'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def original_source():
    blob = subprocess.check_output(['git','show',TEXT_BASELINE_SOURCE[0]+':'+TEXT_BASELINE_SOURCE[1]], cwd=REPO)
    if hashlib.sha256(blob).hexdigest() != ORIGINAL_SHA:
        raise ValueError('Original text-to-image producer source differs')
    return blob.decode('utf8')


def source_map():
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
    files = [Path(__file__), HERE/'abo_query_contract.py',
        TASK/'baseline_5090d.py',
        REPO/'experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/features.py',
        REPO/'experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/modeling.py']
    return {**entry.source_map(), **{str(p.relative_to(REPO)):sha(p) for p in files}}


def prepare(processor, text):
    # Same exact template utility as the original frozen T08 producer. Neither
    # sample/product IDs nor labels are passed into the model.
    from LightGenV2.tasks.t08_abo_image_text_retrieval.baseline_5090d import _inputs
    cpu = _inputs(processor, text=text, instruction=BASELINE_QUERY['text_to_image'])
    return {k:v for k,v in cpu.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}


def raw_descriptor(model, inputs):
    import torch
    hidden = model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
    mask = inputs['attention_mask']
    pos = torch.arange(mask.shape[1],device=mask.device)[None].masked_fill(~mask.bool(),-1).amax(1)
    return hidden[0,pos[0]].float()


def validate_bank(cache, titles, test):
    import torch
    if (cache['galleries']['dynamic'].shape != (2400,2048)
            or cache['galleries']['dynamic'].dtype != torch.float32
            or cache['text_vectors'].shape != (100,2048)
            or cache['query_product_ids'] != [r['product_id'] for r in titles]
            or cache['gallery_ids'] != [r['sample_id'] for r in test]
            or cache['gallery_labels'].tolist() != [int(r['label']) for r in test]
            or cache['query_labels'].tolist() != [int(r['label']) for r in titles]
            or not torch.isfinite(cache['galleries']['dynamic']).all()):
        raise ValueError('Original FP32 gallery, dimension or identities differ')


def load(args, device):
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    titles,_,test = load_contract(args.data)
    if sha(args.bank) != TEXT_BASELINE_BANK_SHA or sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Wrong .82 original gallery or performance reference')
    reference = json.loads(REFERENCE.read_text())
    if abs(reference['metrics']['dynamic_2048']['hit_at_1']-.82)>1e-12:
        raise ValueError('Wrong full100 reference metric')
    if (sha(args.model/'config.json') != MODEL_CONFIG_SHA
            or len(list(args.model.glob('*.safetensors'))) != 1
            or sha(next(args.model.glob('*.safetensors'))) != MODEL_WEIGHTS_SHA):
        raise ValueError('Wrong native frozen Embedding2B snapshot')
    cache = torch.load(args.bank,map_location='cpu',weights_only=True)
    validate_bank(cache,titles,test)
    processor = AutoProcessor.from_pretrained(str(args.model),local_files_only=True,min_pixels=50176,max_pixels=50176)
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model),local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    return model,processor,cache,titles,test


def model_assets(model_path):
    return {p.name:sha(p) for p in sorted(Path(model_path).iterdir()) if p.is_file()}


def cpu_contract(args):
    """Actual original titles: paired original/new full frozen CPU forward.

    Extract original nested encode, change ONLY platform literals CUDA->CPU.
    This does not modify the original producer or the measured graph. Separate
    CPU process; cannot warm the fresh formal query processes.
    """
    if args.output.exists():raise FileExistsError(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(8)
    model,processor,cache,titles,test = load(args,'cpu')
    tree = ast.parse(original_source())
    function = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='encode')
    class CPU(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant('cpu'),node) if node.value=='cuda' else node
    unit = ast.fix_missing_locations(ast.Module(body=[CPU().visit(function)],type_ignores=[]))
    scope = dict(torch=torch,processor=processor,model=model,
        QUERY=BASELINE_QUERY['text_to_image'],DOCUMENT="Represent the user's input.")
    exec(compile(unit,'original_git_encode_cpu_platform_only','exec'),scope)
    results = []
    with torch.inference_mode():
        for i in (0,99):
            text = titles[i]['title']
            original = scope['encode'](text=text)
            cpu = prepare(processor,text)
            actual = raw_descriptor(model,cpu).cpu()
            if not torch.equal(original,actual):raise ValueError('Original/new real-title forward differs')
            a = F.normalize(original,dim=0) @ F.normalize(cache['galleries']['dynamic'],dim=1).T
            b = F.normalize(actual,dim=0) @ F.normalize(cache['galleries']['dynamic'],dim=1).T
            if not torch.equal(a,b) or not torch.equal(a.argsort(descending=True,stable=True),b.argsort(descending=True,stable=True)):
                raise ValueError('Original/new FP32 retrieval differs')
            results.append(dict(query_index=i,product_id=titles[i]['product_id'],
                input_ids=cpu['input_ids'].tolist(),attention_mask=cpu['attention_mask'].tolist(),
                descriptor_shape=list(actual.shape),descriptor_dtype=str(actual.dtype),bitwise_equal=True))
    args.output.mkdir(parents=True)
    result=dict(status='passed',mode='real_title_original_full_CPU_forward_bitwise_contract',
        no_gpu_inference=True,original_git_source=TEXT_BASELINE_SOURCE,original_source_sha256=ORIGINAL_SHA,
        historical_online_profiler_sha256=ONLINE_PROFILER_SHA,online_score_device='cuda FP32; original online profiler, not bulk CPU performance evaluation',
        original_reference_sha256=REFERENCE_SHA,bank_sha256=TEXT_BASELINE_BANK_SHA,
        model_config_sha256=MODEL_CONFIG_SHA,model_weights_sha256=MODEL_WEIGHTS_SHA,
        sources=source_map(),model_assets=model_assets(args.model),checks=results,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    write(args.output/'contract.json',result)
    print(json.dumps(dict(status='passed',real_titles=len(results),no_gpu_inference=True)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','bank','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3))
    p.add_argument('--cpu-contract',action='store_true')
    p.add_argument('--contract',type=Path)
    args = p.parse_args()
    if args.cpu_contract:return cpu_contract(args)
    if (args.output.exists() or not args.output.resolve().is_relative_to(TASK/'runs/simulation')
            or args.repeat is None or args.contract is None):
        raise ValueError('New original-task run, repeat and actual CPU contract required')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU
            or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE') != '100'):
        raise ValueError('Approved physical UUID/private MPS100 required')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import write_csv
    engine.GPU,engine.PIPE = GPU,os.environ['CUDA_MPS_PIPE_DIRECTORY']
    bound = source_map()
    contract = json.loads(args.contract.read_text())
    if contract['status']!='passed' or contract['sources']!=bound or contract['bank_sha256']!=TEXT_BASELINE_BANK_SHA:
        raise ValueError('CPU original-forward contract does not bind this adapter')
    class Guard(engine.MPSGuard):
        def check(self):
            if source_map()!=bound:raise ValueError('Bound T08 inference source changed')
            super().check()
    guard = Guard(SimpleNamespace(deadline=time.time()+1800))
    guard.check()
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(8)
    props = torch.cuda.get_device_properties(0)
    cap = float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
    if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:
        raise ValueError('Physical GPU or cap differs')
    args.output.mkdir(parents=True)
    calls,rankings,vectors,score_rows,telemetry = [],[],[],[],[]
    sampler = None
    identity = dict(protocol_id=PROTOCOL,task='ABO text-to-image retrieval',system='qwen',repeat=args.repeat,
        pid=os.getpid(),gpu_uuid=GPU,mps_percent=100,batch=1,visible_multiprocessors=108,power_limit_w=cap,
        model_config_sha256=MODEL_CONFIG_SHA,model_weights_sha256=MODEL_WEIGHTS_SHA,
        bank=str(args.bank),bank_sha256=TEXT_BASELINE_BANK_SHA,reference_sha256=REFERENCE_SHA,
        cpu_contract=str(args.contract),cpu_contract_sha256=sha(args.contract),
        full100_reference_hit1=.82,full_gallery_count=2400,query_instruction=BASELINE_QUERY['text_to_image'],
        query_modality='official UTF-8 title only, no image or product ID in model input',
        descriptor_dimensions=2048,ranking_stable=True,score_device='cuda',score_dtype='FP32',
        first_sample_retained=True,explicit_warmup_forwards=0,sources=bound,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),command=sys.argv,
        torch=torch.__version__,cuda=torch.version.cuda,cpu_threads=8,
        timer='first UTF-8 title input to200th CPU full ranking; all199 genuine gaps',
        energy_basis='338.2-W host proxy plus identical full-window NVML board integral, no idle subtraction')
    def status(state,**extra):write(args.output/'status.json',dict(status=state,pid=os.getpid(),samples=len(calls),**extra))
    status('loading');write(args.output/'identity.json',identity)
    try:
        with torch.inference_mode():
            model,processor,cache,titles,test = load(args,'cuda:0')
            work = timed_calls('text_to_image',titles,test)
            gallery = F.normalize(cache['galleries']['dynamic'].float(),dim=1).to('cuda:0')
            assets=model_assets(args.model)
            if assets!=contract['model_assets']:raise ValueError('Original tokenizer/processor/model assets changed')
            identity.update(model_assets=assets,
                floating_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
                processor_class=type(processor).__name__,language_layer_count=len(model.model.language_model.layers),
                online_vision_forward=False,original_full_language_forward=True,full_vocabulary_projection=False)
            write(args.output/'identity.json',identity)
            guard.bind()
            write(args.output/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=bound))
            raw = args.output/'telemetry_stdout.jsonl'
            sampler = ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('continuous200_UTF8_to_CPU_ranking_no_warmup')
            time.sleep(.15)  # power bracketing only, zero model forwards
            for call in work:
                guard.fast();sampler.check()
                row = call['row']
                start = time.perf_counter()
                cpu = prepare(processor,row['title'])
                prepared = time.perf_counter()
                inputs = {k:v.to('cuda:0') for k,v in cpu.items()}
                model_start = time.perf_counter()
                begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                vector = raw_descriptor(model,inputs)
                scores = F.normalize(vector,dim=0) @ gallery.T
                ranking = scores.argsort(descending=True,stable=True)
                end.record()
                host_ranking = ranking.cpu().tolist()
                predicted = test[host_ranking[0]]['product_id']
                returned = time.perf_counter()
                # All diagnostic bookkeeping between requests remains charged
                # to the continuous window, with no selected/minimum statistic.
                end.synchronize()
                calls.append(dict(index=call['index'],query_index=call['unique_query_index'],pass_index=call['pass_index'],
                    product_id=row['product_id'],title=row['title'],input_start=start,prepare_end=prepared,
                    model_start=model_start,cpu_result_return=returned,
                    diagnostic_full_model_search_cuda_ms=begin.elapsed_time(end),
                    predicted_product_id=predicted,hit1=int(predicted==row['product_id']),
                    input_ids=cpu['input_ids'].tolist(),attention_mask=cpu['attention_mask'].tolist(),
                    valid_input_tokens=int(cpu['attention_mask'].sum()),input_ids_shape=list(cpu['input_ids'].shape),
                    title_utf8_sha256=hashlib.sha256(row['title'].encode('utf8')).hexdigest()))
                rankings.append(host_ranking);vectors.append(vector.detach());score_rows.append(scores.detach())
                if call['index']==0 or (call['index']+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12)
            telemetry=sampler.stop();sampler=None
            a,b = calls[0]['input_start'],calls[-1]['cpu_result_return']
            energy = interval_energy([(a,b)],telemetry,338.2)
            guard.check()
            if source_map()!=bound or sha(args.bank)!=TEXT_BASELINE_BANK_SHA or model_assets(args.model)!=assets:
                raise ValueError('Frozen source or gallery changed')
            load_contract(args.data)  # recheck exact original CSVs after timing
            np.savez_compressed(args.output/'retrieval_outputs.npz',
                rankings=np.asarray(rankings,dtype=np.int64),vectors=torch.stack(vectors).cpu().numpy(),
                scores=torch.stack(score_rows).cpu().numpy())
            report=dict(status='complete',**identity,measured_calls=200,unique_queries=100,
                continuous_wall_s=b-a,total_system_energy_j=energy['energy_j'],power_integration=energy,
                mean_board_power_w=(energy['energy_j']-338.2*(b-a))/(b-a),
                measured_unique100_hit1=sum(r['hit1'] for r in calls[:100])/100,
                common_reference_tokens=sum(r['valid_input_tokens'] for r in calls))
            write(args.output/'report.json',report);status('complete')
            print(json.dumps({k:report[k] for k in ('status','measured_calls','measured_unique100_hit1','continuous_wall_s','mean_board_power_w')}))
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(args.output/'telemetry.csv',telemetry)
        write_csv(args.output/'calls.csv',calls)
        if vectors and not (args.output/'retrieval_outputs.npz').exists():
            np.savez_compressed(args.output/'retrieval_outputs_partial.npz',
                rankings=np.asarray(rankings,dtype=np.int64),vectors=torch.stack(vectors).cpu().numpy(),
                scores=torch.stack(score_rows).cpu().numpy())
        guard.close(args.output/'process_audit.json')
        write(args.output/'files_manifest.json',[dict(file=f.name,sha256=sha(f),bytes=f.stat().st_size)
            for f in sorted(args.output.iterdir()) if f.is_file() and f.name!='files_manifest.json'])


if __name__=='__main__':main()
