"""Real PNG/plain UTF8 through unchanged epoch45 NN to original CPU grid result.

Software FFT/fusion/residuals remain executed and retained; selected original
NN synchronized wall plus six parameter-derived passes is a mixed projection,
not apparatus-measured E2E. No features/cache/dummy input replace online input.
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
from openmoji_e45_graph import TASK,GPU,sha,write,load_runtime,live_input,patch_model,shape_observers,validate_stage_counts
from prepare_openmoji_e45 import source_files
from stage_meter import StageMeter,source_record,optical_energy_j
from measure_ours import write_csv
from openmoji_grid_audit import metrics
from protocol_identity import effective_protocol


def partition_live_input(meter,first,start,end):
    """Record disjoint real prefix/suffix around the independently timed lookup.

    No model/input math is rewritten. The already CPU-verified live_input is
    called unchanged. Prefix includes actual file/decode/tokenizer/H2D and its
    synchronization; suffix includes the original BF16->FP32 conversion.
    """
    new=meter.rows[first:]
    if len(new)!=1 or new[0]['stage']!='frontend.plain_token_embedding':
        raise ValueError('Live frontend must contain exactly its original embedding span')
    a,b=new[0]['start_monotonic'],new[0]['end_monotonic']
    if not start<=a<=b<=end:raise ValueError('Live frontend interval partition differs')
    def row(name,left,right):
        return dict(sample_index=meter.sample_index,valid_samples=1,stage=name,family='input',charged=True,
            start_monotonic=left,end_monotonic=right,synchronized_wall_ms=(right-left)*1000,cuda_event_ms=None)
    meter.rows.insert(first,row('input.PNG_plain_tokenizer_H2D',start,a))
    meter.rows.append(row('input.original_embedding_float_conversion',b,end))


def validate_contract(path):
    value=json.loads(Path(path).read_text())
    if (value.get('status')!='passed' or value.get('no_gpu') is not True
            or value.get('real_test_examples')!=2 or [r['test_index'] for r in value.get('checks',[])]!=[200,999]
            or not all(r['bitwise_equal'] for r in value['checks'])
            or value['identity']['selected_body_sha256']!='03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21'):
        raise ValueError('Actual original/instrumented epoch45 CPU contract required')
    if any(sha(REPO/k)!=v for k,v in value['sources'].items()):raise ValueError('CPU-bound original NN sources changed')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage-contract',type=Path,required=True)
    parser.add_argument('--repeat',type=int,choices=(1,2,3),required=True);args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(TASK/'runs/simulation'):raise ValueError('New immutable task trial required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100':
        raise ValueError('Exact A100 UUID/private MPS100 required')
    contract=validate_contract(args.stage_contract)
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
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import compose_prediction
    torch.set_num_threads(8);output.mkdir(parents=True)
    meter=StageMeter(torch.cuda);applied=None;handles=[];sampler=None;telemetry=[];calls=[];shapes=[]
    saved={k:[] for k in ('category_logits','edit_logits','task_logits','prediction','generated','edited')}
    def status(value,**extra):write(output/'status.json',dict(status=value,pid=os.getpid(),samples=len(calls),**extra))
    status('loading')
    try:
        with torch.inference_mode():
            model,embedding,tokenizer,settings,records,identity=load_runtime(output,'cuda:0')
            props=torch.cuda.get_device_properties(0)
            cap=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
            if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:
                raise ValueError('Wrong physical A100/cap')
            applied=patch_model(model,embedding,meter)
            for stage in ('input.PNG_plain_tokenizer_H2D','input.original_embedding_float_conversion'):
                meter.ledger.append(source_record(live_input,stage,'input',True))
            meter.ledger.append(source_record(compose_prediction,'output.original_grid_and_CPU_result','output',True))
            handles=shape_observers(model,embedding,shapes,lambda:meter.sample_index)
            frozen=source_files()
            identity.update(protocol_id=effective_protocol()['protocol_id'],system='ours',repeat=args.repeat,
                gpu_uuid=GPU,mps_percent=100,batch=1,pid=os.getpid(),power_limit_w=cap,visible_multiprocessors=108,
                stage_contract=str(args.stage_contract),stage_contract_sha256=sha(args.stage_contract),
                sources=frozen,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                command=sys.argv,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                explicit_warmup_forwards=0,first_sample_retained=True,input_count=200,input_order='original TEST first200',
                amp_enabled=settings.amp_enabled,autocast_dtype='bfloat16',cpu_threads=8,
                common_Qwen_token_work='pending actual same-input Qwen formal calls; not replaced by own plain token count',
                native_merger_calls=0,native_merger_zero_reason='Original frozen patch/position stem has no vision Transformer or native merger; actual prompt-to-vision bridge is charged',
                not_a_device_measured_e2e=True)
            write(output/'identity.json',identity)
            guard.bind();write(output/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=frozen))
            raw=output/'telemetry_stdout.jsonl';sampler=ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('continuous200_real_PNG_UTF8_zero_warmup');time.sleep(.15)
            for index,row in enumerate(records[:200]):
                guard.fast();sampler.check();meter.sample_index=index;first=len(meter.rows)
                start=time.perf_counter()
                source=torch.tensor(row['source_grid'],device='cuda:0')[None]
                pixels,hidden,ids=live_input(row,settings,tokenizer,embedding,'cuda:0')
                torch.cuda.synchronize();ready=time.perf_counter();partition_live_input(meter,first,start,ready)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=settings.amp_enabled):
                    result=model(pixels,hidden)
                if result['edit_logits'].dtype!=torch.bfloat16:raise ValueError('Original AMP decoder must return native BF16 logits')
                with meter.scope('output.original_grid_and_CPU_result','output'):
                    prediction,generated,edited=compose_prediction(result['category_logits'],result['edit_logits'],source)
                    cpu={k:result[k].detach().float().cpu().numpy()[0] for k in ('category_logits','edit_logits','task_logits')}
                    cpu.update(prediction=prediction.cpu().numpy()[0],generated=generated.cpu().numpy()[0],edited=edited.cpu().numpy()[0])
                returned=time.perf_counter()
                calls.append(dict(index=index,sample_id=row['sample_id'],task=row['task'],instruction=row['instruction'],
                    input_start=start,prepare_end=ready,cpu_result_return=returned,
                    internal_gap_ms=0. if not calls else (start-calls[-1]['cpu_result_return'])*1000,
                    plain_input_ids=ids,plain_input_tokens=len(ids),input_shapes=[list(pixels.shape),list(hidden[0].shape)],
                    category_logits_dtype=str(result['category_logits'].dtype),edit_logits_dtype=str(result['edit_logits'].dtype),
                    instruction_utf8_sha256=hashlib.sha256(row['instruction'].encode('utf8')).hexdigest()))
                for key in saved:saved[key].append(cpu[key])
                if index==0 or (index+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
            validate_stage_counts(Counter(r['stage'] for r in meter.rows),200)
            diagnostic=interval_energy([(calls[0]['input_start'],calls[-1]['cpu_result_return'])],telemetry,0)
            guard.check()
            if source_files()!=frozen:raise ValueError('Frozen inference sources changed')
            pngs={};input_manifest=[]
            for index,row in enumerate(records[:200]):
                path=settings.data_dir/row['relative_dir']/row['files']['source'];data=path.read_bytes()
                pngs[f'PNG_{index:03d}']=np.frombuffer(data,dtype=np.uint8)
                input_manifest.append(dict(index=index,sample_id=row['sample_id'],original_path=str(path),
                    bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            np.savez_compressed(output/'original_source_PNG_bytes.npz',**pngs)
            write(output/'original_input_manifest.json',input_manifest)
            (output/'original_TEST.jsonl').open('xb').write(settings.test_manifest.read_bytes())
            values=[metrics(saved['prediction'][i],saved['edited'][i],int(saved['task_logits'][i].argmax()),row)
                    for i,row in enumerate(records[:200])]
            charged=sum(r['synchronized_wall_ms'] for r in meter.rows if r['charged'])+sum(r['internal_gap_ms'] for r in calls)
            optical=200*6*1.0447;seconds=(charged+optical)/1000;energy=optical_energy_j(optical/1000,charged/1000)
            np.savez_compressed(output/'grid_outputs.npz',**{k:np.asarray(v) for k,v in saved.items()})
            write(output/'summary.json',dict(status='complete',repeat=args.repeat,valid_samples=200,
                measured_first200_metrics={k:sum(v[k] for v in values)/200 for k in values[0]},
                original_full1000_reference_CCA=.8765,original_laboratory_reference_CCA=.8655,
                actual_plain_input_tokens200=sum(r['plain_input_tokens'] for r in calls),common_Qwen_input_tokens200=None,
                charged_electronic_ms_total=charged,optical_ms_total=optical,projected_seconds200=seconds,
                phase_estimated_j200=energy,sample_ms=seconds*5,sample_j=energy/200,
                diagnostic_software_wall_s=calls[-1]['cpu_result_return']-calls[0]['input_start'],
                diagnostic_software_board_integral=diagnostic,diagnostic_GPU_energy_not_used_in_phase_estimate=True,
                not_a_device_measured_e2e=True))
            status('complete');print(json.dumps(dict(status='complete',repeat=args.repeat,sample_ms=seconds*5,sample_j=energy/200)),flush=True)
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(output/'telemetry.csv',telemetry)
        guard.close(output/'process_audit.json')
        for handle in handles:handle.remove()
        if applied is not None:applied.close()
        write_csv(output/'calls.csv',calls);write_csv(output/'stages.csv',meter.rows)
        write(output/'stage_ledger.json',meter.ledger);write(output/'nn_module_shapes.json',shapes)
        if saved['prediction'] and not (output/'grid_outputs.npz').exists():
            np.savez_compressed(output/'grid_outputs_partial.npz',**{k:np.asarray(v) for k,v in saved.items()})
        write(output/'files_manifest.json',[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(output.iterdir()) if p.is_file() and p.name!='files_manifest.json'])


if __name__=='__main__':main()
