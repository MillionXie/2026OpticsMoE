"""Untimed actual-200-query MAC replay of the already measured T08 baseline.

No timings or power from this observer are used in the comparison. An exact
FP32 descriptor/score/ranking match to each qualified trial is required. Dense
MAC main terms use 1MAC=2OP; non-MAC and unsupported operators remain explicit.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
import measure_abo_text_qwen as measured
from operation_ledger import OperationStageMeter,make_dispatch_counter


def independent_main_mac(lengths,hidden,layers,intermediate,heads,kv_heads,head_dim,candidates=2400):
    """Native dense Q/K/V/O + SwiGLU3 + rectangular QK/AV + retrieval dot.

    Rotary-position construction and non-MAC arithmetic are not included in
    this analytical main-term cross-check; the dispatch ledger retains them.
    """
    values=list(lengths)
    if (not values or any(not isinstance(n,int) or n<1 for n in values)
            or min(hidden,layers,intermediate,heads,kv_heads,head_dim,candidates)<1
            or hidden!=heads*head_dim or heads%kv_heads):raise ValueError('Invalid original Qwen shapes')
    projection_per_token=hidden*(heads+2*kv_heads)*head_dim+heads*head_dim*hidden
    mlp_per_token=3*hidden*intermediate
    return layers*((projection_per_token+mlp_per_token)*sum(values)
        +2*heads*head_dim*sum(n*n for n in values))+len(values)*candidates*hidden


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('measured-root','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.measured_root.resolve();output=args.output.resolve()
    if output.exists() or not output.is_relative_to(measured.TASK/'runs/simulation'):
        raise ValueError('New immutable task operation-replay run required')
    state=measured.json.loads((root/'suite_status.json').read_text())
    if state['status']!='complete' or state['completed']!=[f'qwen_repeat{i:02d}' for i in (1,2,3)]:
        raise ValueError('Three qualified formal trials must already have completed')
    import csv
    import numpy as np
    from audit_abo_text_qwen import audit
    refs=[audit(root/f'qwen_repeat{i:02d}',REPO/'data/abo_easy100_dataset_20260906') for i in (1,2,3)]
    with (root/'qwen_repeat01/calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
    references=[np.load(root/f'qwen_repeat{i:02d}/retrieval_outputs.npz',allow_pickle=False) for i in (1,2,3)]
    for item in references[1:]:
        for key in ('vectors','scores','rankings'):
            if not np.array_equal(item[key],references[0][key]):raise ValueError('Qualified repeats differ; inspect before OP replay')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    import fcntl
    engine.GPU=measured.GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=output
    frozen=measured.source_map()
    for file in (Path(__file__),HERE/'operation_ledger.py',HERE/'stage_meter.py'):
        frozen[str(file.relative_to(REPO))]=measured.sha(file)
    def stable():
        if any(measured.sha(REPO/k)!=v for k,v in frozen.items()):raise ValueError('Frozen untimed replay source changed')
    lock_path=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if engine.base.guard_records(measured.GPU):raise ValueError('Foreign A100 process')
        validate_idle_memory(measured.GPU)
        output.mkdir(parents=True);active=False;guard=None;loading_audit=[];counter=None
        try:
            engine.daemon(True);active=True
            os.environ.update(CUDA_VISIBLE_DEVICES=measured.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100')
            # Loading has its own process watcher; this replay is never allowed
            # to run concurrently with a formal timing window.
            import torch
            from torch.nn import functional as F
            import threading
            stop=threading.Event();loading_errors=[]
            def watch():
                from run_temporal_remainder import process_snapshot
                while not stop.wait(1):
                    try:
                        snapshot=process_snapshot(measured.GPU,os.getpid(),engine.PIPE);loading_audit.append(snapshot)
                        if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                            loading_errors.append(snapshot);return
                    except BaseException as error:
                        loading_errors.append(dict(error=repr(error)));return
            watcher=threading.Thread(target=watch,daemon=True);watcher.start()
            try:
                torch.set_num_threads(8)
                props=torch.cuda.get_device_properties(0)
                cap=float(subprocess.check_output(['nvidia-smi','--id='+measured.GPU,
                    '--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
                if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:
                    raise ValueError('Wrong physical GPU/client cap')
                options=SimpleNamespace(model=Path('/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'),
                    data=REPO/'data/abo_easy100_dataset_20260906',bank=measured.TASK/'runs/simulation/text_to_image_frozen_20260920/embeddings.pt')
                model,processor,cache,titles,test=measured.load(options,'cuda:0')
            finally:stop.set();watcher.join(timeout=10)
            if watcher.is_alive() or loading_errors:raise ValueError('Foreign A100 use during untimed loading')
            guard=engine.MPSGuard(SimpleNamespace(deadline=time.time()+1800));guard.bind()
            config=model.config.text_config
            with torch.inference_mode():gallery=F.normalize(cache['galleries']['dynamic'].float(),dim=1).to('cuda:0')
            meter=OperationStageMeter();counter=make_dispatch_counter(meter);checks=[]
            work=measured.timed_calls('text_to_image',titles,test)
            with torch.inference_mode():
                for call,reference in zip(work,calls):
                    stable();guard.fast();meter.sample_index=call['index']
                    with counter,meter.scope('qwen.input_full_language_and_gallery','baseline',True):
                        cpu=measured.prepare(processor,call['row']['title'])
                        inputs={k:v.to('cuda:0') for k,v in cpu.items()}
                        vector=measured.raw_descriptor(model,inputs)
                        scores=F.normalize(vector,dim=0)@gallery.T
                        order=scores.argsort(descending=True,stable=True)
                        observed={'vectors':vector.cpu().numpy(),'scores':scores.cpu().numpy(),'rankings':order.cpu().numpy()}
                    if (cpu['input_ids'].tolist()!=json.loads(reference['input_ids'])
                            or cpu['attention_mask'].tolist()!=json.loads(reference['attention_mask'])):
                        raise ValueError('Original measured input IDs differ in replay')
                    for key,value in observed.items():
                        if not np.array_equal(value,references[0][key][call['index']]):
                            raise ValueError('Observer/native measured output differs: '+key)
                    checks.append(dict(index=call['index'],valid_input_tokens=int(cpu['attention_mask'].sum()),bitwise_equal=True))
            guard.check();stable();ledger=counter.report()
            if ledger['unscoped_review_required_mac']:raise ValueError('Actual MAC outside selected source scope')
            main_mac=independent_main_mac([c['valid_input_tokens'] for c in checks],config.hidden_size,
                config.num_hidden_layers,config.intermediate_size,config.num_attention_heads,
                config.num_key_value_heads,config.head_dim)
            if not main_mac<=ledger['actual_executed_real_mac']<=main_mac*1.001:
                raise ValueError('Actual dense MAC differs from independent native-network main terms')
            value=dict(ledger,status='UNTIMED_MAC_REPLAY_200_MATCHED_FORMAL_INPUTS_NOT_A_TIMING_RUN',
                gpu_uuid=measured.GPU,mps_percent=100,power_limit_w=cap,device_name=props.name,
                protocol_id=measured.PROTOCOL,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                sources=frozen,measured_root=str(root.relative_to(REPO)),qualified_trial_audits=refs,
                raw_reference_bitwise_matches=checks,input_calls=200,unique_queries=100,
                common_input_tokens=sum(c['valid_input_tokens'] for c in checks),
                independent_QKVO_SwiGLU_attention_search_main_mac=main_mac,
                additional_observed_dense_mac=ledger['actual_executed_real_mac']-main_mac,
                parameters_loaded_but_vocabulary_logits_not_executed=True,
                full_scalar_OP_including_nonlinearities_not_claimed=True)
            measured.write(output/'operation_counts.json',value)
            measured.write(output/'status.json',dict(status='complete',calls=200,mac=ledger['actual_executed_real_mac'],OP_MAC_convention=2*ledger['actual_executed_real_mac']))
            print(json.dumps({k:value[k] for k in ('status','input_calls','actual_executed_real_mac','independent_QKVO_SwiGLU_attention_search_main_mac','additional_observed_dense_mac','unsupported_operators')}),flush=True)
        except BaseException as error:
            measured.write(output/'failure.json',dict(error=repr(error)))
            if counter is not None:measured.write(output/'partial_operations.json',counter.report())
            raise
        finally:
            measured.write(output/'loading_process_audit.json',loading_audit)
            if guard is not None:guard.close(output/'process_audit.json')
            if active:engine.daemon(False)
            measured.write(output/'files_manifest.json',[dict(file=f.name,bytes=f.stat().st_size,sha256=measured.sha(f))
                for f in sorted(output.iterdir()) if f.is_file() and f.name!='files_manifest.json'])


if __name__=='__main__':main()
