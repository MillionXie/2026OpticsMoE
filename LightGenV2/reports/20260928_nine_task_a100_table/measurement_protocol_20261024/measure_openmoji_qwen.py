"""Original native Qwen PNG/UTF8 to CPU grid: continuous200, no dummy warmup.

Formal quality is the sealed full1000 report. First200 outputs are retained
only to audit measurement identity, never to replace full-TEST accuracy.
"""
from __future__ import annotations
import argparse
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
from openmoji_qwen_live import TASK,GPU,ORIGINAL,sha,write,load_runtime,native_and_head
from prepare_openmoji_e45 import source_files
from protocol_identity import effective_protocol


def validate_contract(path):
    value=json.loads(Path(path).read_text())
    row=next(r for r in effective_protocol()['rows'] if r['id']=='07')
    if (value.get('status')!='passed' or value.get('no_gpu') is not True
            or [r['test_index'] for r in value.get('checks',[])]!=[200,999]
            or not all(r['bitwise_equal'] for r in value['checks'])
            or value['identity']['selected_body_sha256']!=row['qwen_pt_sha256']):
        raise ValueError('Exact complete native CPU contract required')
    if any(sha(REPO/k)!=v for k,v in value['sources'].items()):raise ValueError('CPU-bound original source changed')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--repeat',type=int,choices=(1,2,3),required=True);args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(TASK/'runs/simulation'):raise ValueError('New immutable task trial required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100':
        raise ValueError('Exact A100 UUID/private MPS100 required')
    proof=validate_contract(args.contract)
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import write_csv
    engine.GPU=GPU;engine.PIPE=os.environ['CUDA_MPS_PIPE_DIRECTORY']
    guard=engine.MPSGuard(SimpleNamespace(deadline=time.time()+1800));guard.check()
    import numpy as np
    import torch
    import transformers
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import compose_prediction
    from openmoji_grid_audit import metrics
    torch.set_num_threads(8);output.mkdir(parents=True)
    calls=[];telemetry=[];sampler=None
    saved={k:[] for k in ('category_logits','edit_logits','task_logits','prediction','generated','edited')}
    def status(value,**extra):write(output/'status.json',dict(status=value,pid=os.getpid(),samples=len(calls),**extra))
    status('loading')
    try:
        with torch.inference_mode():
            device=torch.device('cuda:0');runtime,identity=load_runtime(output,device)
            settings,native,processor,head=runtime
            records=[json.loads(s) for s in settings.test_manifest.read_text().splitlines()]
            props=torch.cuda.get_device_properties(0)
            cap=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
            if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:
                raise ValueError('Wrong physical A100/cap')
            frozen=source_files()
            counts_expected=proof['checks'][0]['native_counts']
            identity.update(protocol_id=effective_protocol()['protocol_id'],system='qwen',repeat=args.repeat,
                gpu_uuid=GPU,mps_percent=100,batch=1,pid=os.getpid(),power_limit_w=cap,visible_multiprocessors=108,
                cpu_contract=str(args.contract),cpu_contract_sha256=sha(args.contract),sources=frozen,
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                command=sys.argv,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                explicit_warmup_forwards=0,first_sample_retained=True,input_count=200,input_order='original TEST first200',
                amp_enabled=settings.amp_enabled,autocast_dtype='bfloat16',cpu_threads=8,
                native_counts_per_query=counts_expected,original_native_CPU_feature_return_preserved=True,
                training_cache_used_as_online_input=False,full_vocabulary_projection=False,
                formal_accuracy_source='original confirmed full1000 TEST; measured subset is audit only')
            write(output/'identity.json',identity)
            guard.bind();write(output/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=frozen))
            raw=output/'telemetry_stdout.jsonl';sampler=ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('continuous200_native_PNG_UTF8_zero_warmup');time.sleep(.15)
            for index,row in enumerate(records[:200]):
                guard.fast();sampler.check()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter()
                begin.record()
                result,inputs,counts,features=native_and_head(runtime,row,device)
                source=torch.tensor(row['source_grid'],device=device)[None]
                prediction,generated,edited=compose_prediction(result['category_logits'],result['edit_logits'],source)
                cpu={k:result[k].detach().float().cpu().numpy()[0] for k in ('category_logits','edit_logits','task_logits')}
                cpu.update(prediction=prediction.cpu().numpy()[0],generated=generated.cpu().numpy()[0],edited=edited.cpu().numpy()[0])
                end.record()
                torch.cuda.synchronize();returned=time.perf_counter()
                if counts!=counts_expected or result['edit_logits'].dtype!=torch.bfloat16:
                    raise ValueError('Original full native layers/AMP precision changed')
                ids,mask=inputs['input_ids'],inputs['attention_mask']
                if len(ids)!=1 or len(mask)!=1 or len(ids[0])!=sum(mask[0]):raise ValueError('Original B1 no-padding input changed')
                calls.append(dict(index=index,sample_id=row['sample_id'],task=row['task'],instruction=row['instruction'],
                    input_start=start,cpu_result_return=returned,
                    diagnostic_request_cuda_ms=begin.elapsed_time(end),
                    internal_gap_ms=0. if not calls else (start-calls[-1]['cpu_result_return'])*1000,
                    native_inputs_json=json.dumps(inputs),native_layer_counts_json=json.dumps(counts),
                    valid_input_tokens=sum(mask[0]),visual_feature_shape_json=json.dumps(list(features[0].shape)),
                    language_feature_shape_json=json.dumps(list(features[1].shape)),
                    category_logits_dtype=str(result['category_logits'].dtype),edit_logits_dtype=str(result['edit_logits'].dtype),
                    instruction_utf8_sha256=hashlib.sha256(row['instruction'].encode('utf8')).hexdigest()))
                for key in saved:saved[key].append(cpu[key])
                if index==0 or (index+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
            a,b=calls[0]['input_start'],calls[-1]['cpu_result_return']
            energy=interval_energy([(a,b)],telemetry,338.2)
            guard.check()
            if source_files()!=frozen:raise ValueError('Frozen native inference source changed')
            # Input bytes/SHA are read after the complete measurement window.
            pngs={};input_manifest=[]
            for index,row in enumerate(records[:200]):
                path=settings.data_dir/row['relative_dir']/row['files']['source'];data=path.read_bytes()
                pngs[f'PNG_{index:03d}']=np.frombuffer(data,dtype=np.uint8)
                input_manifest.append(dict(index=index,sample_id=row['sample_id'],original_path=str(path),
                    bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            np.savez_compressed(output/'original_source_PNG_bytes.npz',**pngs)
            write(output/'original_input_manifest.json',input_manifest)
            (output/'original_TEST.jsonl').open('xb').write(settings.test_manifest.read_bytes())
            reference=(ORIGINAL/'selected_checkpoint_test_evaluation.json').read_bytes()
            (output/'original_full_TEST_reference.json').open('xb').write(reference)
            (output/'original_native_config.json').open('xb').write((settings.qwen_checkpoint/'config.json').read_bytes())
            np.savez_compressed(output/'grid_outputs.npz',**{k:np.asarray(v) for k,v in saved.items()})
            values=[metrics(saved['prediction'][i],saved['edited'][i],int(saved['task_logits'][i].argmax()),row)
                    for i,row in enumerate(records[:200])]
            write(output/'summary.json',dict(status='complete',repeat=args.repeat,valid_samples=200,
                formal_full1000_reference_metrics=json.loads(reference)['metrics']['overall'],
                measured_subset_diagnostics_not_formal_accuracy=True,
                measured_first200_metrics={k:sum(v[k] for v in values)/200 for k in values[0]},
                common_Qwen_input_tokens200=sum(r['valid_input_tokens'] for r in calls),
                continuous_wall_s=b-a,system_energy_j=energy['energy_j'],power_integration=energy,
                board_energy_j=energy['energy_j']-338.2*(b-a),
                time_weighted_board_w=(energy['energy_j']-338.2*(b-a))/(b-a),
                sample_ms=(b-a)*5,sample_j=energy['energy_j']/200))
            status('complete');print(json.dumps(dict(status='complete',repeat=args.repeat,wall_s=b-a)),flush=True)
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(output/'telemetry.csv',telemetry)
        guard.close(output/'process_audit.json');write_csv(output/'calls.csv',calls)
        if saved['prediction'] and not (output/'grid_outputs.npz').exists():
            np.savez_compressed(output/'grid_outputs_partial.npz',**{k:np.asarray(v) for k,v in saved.items()})
        write(output/'files_manifest.json',[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(output.iterdir()) if p.is_file() and p.name!='files_manifest.json'])


if __name__=='__main__':main()
