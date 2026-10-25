"""LG-only two-buffer captured ORIGINAL prefix/tail pipeline, all128 fields."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import lgvq_graph_pipeline_20261026 as graphbase

base=graphbase.base
GRAPH_ENTRY=Path(graphbase.__file__)
PREVIOUS_SOURCES=base.source_map
graphbase.__file__=__file__
base.__file__=__file__
base.PIPE='/DATA/DATA1/guest3/.cache/lgvq_split_graph_pipeline_20261026'

def sources():
    result=PREVIOUS_SOURCES()
    for p in (GRAPH_ENTRY,Path(__file__)):result[p.relative_to(base.REPO).as_posix()]=base.sha(p)
    return result
base.source_map=sources

def buffer_for(field):
    assert 3<=field<128
    return (field-3)%2

def split_timed(model,front,position,raw,guard,folder):
    import numpy as np
    import torch
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native_front
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    from lgvq_live_frontend import equal_tree
    h=graphbase.hoisted;meter=h.minimal.NoInnerTimer()
    prefix_fn,tail_fn,partition=h.original_functions(model)
    base.write(folder/'AST_partition.json',partition)
    restore,constants=graphbase.prepare_graph_constants(model);base.write(folder/'graph_constants.json',constants)
    position=position.to(dtype=torch.bfloat16)
    reference=json.loads((folder.parent/'external_preparation.json').read_text())['rows'][0]
    slots=[dict(static={k:torch.empty_like(raw[0][k]) for k in ('pixels','quality','ids','mask')},
                front_graph=torch.cuda.CUDAGraph(),body_graph=torch.cuda.CUDAGraph()) for _ in range(2)]
    fs,bs=torch.cuda.Stream(),torch.cuda.Stream()
    origin=torch.cuda.Event(enable_timing=True);origin.record();origin.synchronize()
    outputs=[];diagnostic_outputs=[];events=[];tokens=[];pending=deque();capture=None
    count=dict(value=0);handles=[]
    for name in ('parallel_router','parallel_optics','serial_router','serial_optics'):
        def hook(*args):count['value']+=1
        handles.append(getattr(model,name).propagation.register_forward_pre_hook(hook))
    def diagnostic_tensor():return torch.stack([v for fusion in model.fusions for v in fusion.last_diagnostics.values()])
    host_start=time.perf_counter()
    try:
        with torch.inference_mode():
            for field,kind in enumerate(graphbase.execution_plan()):
                guard.fast();slot=slots[0 if field<3 else buffer_for(field)]
                start,end,front_end,body_start=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                with torch.cuda.stream(fs):
                    if 'released' in slot:fs.wait_event(slot['released'])
                    start.record()
                a=time.perf_counter()
                rendered=native_front.render_prompt(front.tokenizer,target_name='temporal',prompt=native_front.TARGET_PROMPTS['temporal'])
                tok=front.tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False)
                b=time.perf_counter();tokens.append(dict(field=field,start=a,end=b,input_ids=tok['input_ids'].tolist(),mask=tok['attention_mask'].tolist()))
                ids=tok['input_ids'].pin_memory();mask=tok['attention_mask'].bool().pin_memory()
                current=raw[field%32]
                with torch.cuda.stream(fs):
                    for key in ('pixels','quality'):slot['static'][key].copy_(current[key],non_blocking=True)
                    slot['static']['ids'].copy_(ids,non_blocking=True);slot['static']['mask'].copy_(mask,non_blocking=True)
                if kind=='eager_real':
                    with torch.cuda.stream(fs):
                        batch=base.native_lg_front(front,position,slot['static'],meter,fs)
                        prefix=prefix_fn(model,**batch,optical_enabled=True);front_end.record()
                    with torch.cuda.stream(bs):
                        bs.wait_event(front_end);body_start.record();previous=count['value']
                        result=tail_fn(model,prefix,optical_enabled=True)
                        assert count['value']-previous==6
                        prediction=result['prediction'].clone();diagnostic=diagnostic_tensor().clone();end.record()
                    end.synchronize()
                else:
                    if field==3:
                        # Four captures allocate static instructions, not extra
                        # discarded inference. All capture host time is retained.
                        ca=time.perf_counter();captured=[]
                        for number,target in enumerate(slots):
                            with torch.cuda.graph(target['front_graph'],stream=fs):
                                batch=base.native_lg_front(front,position,target['static'],meter,fs)
                                target['prefix']=prefix_fn(model,**batch,optical_enabled=True)
                            previous=count['value']
                            with torch.cuda.graph(target['body_graph'],stream=bs):
                                target['task']=tail_fn(model,target['prefix'],optical_enabled=True)
                                target['diagnostics']=diagnostic_tensor()
                            assert count['value']-previous==6;captured.append(dict(buffer=number,propagation_calls=6))
                        cb=time.perf_counter();capture=dict(first_replayed_field=3,host_start=ca,host_end=cb,host_ms=(cb-ca)*1000,
                            captured_original_full_field_propagation_calls=6,captured_buffers=captured,total_graphs=4,
                            first48_eager_video_results_retained=True)
                    with torch.cuda.stream(fs):slot['front_graph'].replay();front_end.record()
                    with torch.cuda.stream(bs):
                        bs.wait_event(front_end);body_start.record();slot['body_graph'].replay()
                        prediction=slot['task']['prediction'].clone();diagnostic=slot['diagnostics'].clone();end.record()
                slot['released']=end;outputs.append(prediction);diagnostic_outputs.append(diagnostic)
                events.append(dict(field=field,kind=kind,start=start,end=end,front_end=front_end,body_start=body_start))
                pending.append((end,ids,mask))
                if len(pending)>3:pending.popleft()[0].synchronize()
                if (field+1)%16==0:base.write(folder/'status.json',dict(status='two_buffer_graph_real_fields',completed_video_calls=(field+1)*16,total=2048))
            events[-1]['end'].synchronize();host_end=time.perf_counter()
        rows=[dict(field=r['field'],kind=r['kind'],**{k+'_ms':origin.elapsed_time(r[k]) for k in ('start','end','front_end','body_start')}) for r in events]
        elapsed=rows[-1]['end_ms']-rows[0]['start_ms']
        array=torch.cat(outputs).float().cpu().numpy();diagnostics=torch.stack(diagnostic_outputs).float().cpu().numpy()
        assert array.shape==(128,16) and diagnostics.shape==(128,16) and np.isfinite(array).all() and np.isfinite(diagnostics).all()
        for values in (array,diagnostics):assert all(np.array_equal(values[:32],values[k*32:(k+1)*32]) for k in (1,2,3))
        assert len(tokens)==128 and all(t['input_ids']==reference['input_ids'] and t['mask']==reference['mask'] for t in tokens)
        np.savez_compressed(folder/'graph_outputs.npz',prediction=array,diagnostics=diagnostics)
        base.write(folder/'graph_timing.json',dict(status='complete',video_calls=2048,unique_videos=512,fields=128,frames_per_field=64,
            continuous_GPU_timeline_ms=elapsed,dispatch_to_last_score_host_ms=(host_end-host_start)*1000,
            first_GPU_front_start_ms=rows[0]['start_ms'],last_GPU_score_ms=rows[-1]['end_ms'],time16_ms=elapsed/128,equivalent_frame_ms=elapsed/8192,
            rows=rows,capture=capture,eager_real_fields=3,eager_real_video_calls=48,graph_replays=125,graph_real_video_calls=2000,
            separate_front_and_body_graphs=True,double_buffer_with_previous_score_completion_dependency=True,
            no_dummy_or_discarded_inferences=True,no_fast_window_selection=True,first_eager_calls_and_capture_cost_in_continuous_window=True,
            raw_inputs_copied_on_every_field=True,all_neural_features_recomputed_on_every_graph_replay=True,
            actual_complete_software_includes_FFT=True,physical_optical_E2E_measured=False,original_dtype_and_precision_unchanged=True,
            fusion_diagnostic_GPU_arithmetic_and_copy_included=True,fusion_diagnostic_CPU_readback_batched_after_last_score=True,
            prefixes=[dict(video_calls=n,time16_ms=(rows[n//16-1]['end_ms']-rows[0]['start_ms'])*16/n) for n in base.prefixes()],
            steady=base.steady_report([r['end_ms']-rows[0]['start_ms'] for r in rows])))
        base.write(folder/'online_tokenizer.json',dict(records=tokens,CPU_work_sum_ms=sum((r['end']-r['start'])*1000 for r in tokens),all_actual_IDs_and_masks_equal_original=True))
        restore()
        with torch.inference_mode(),torch.cuda.stream(bs):
            batch=base.native_lg_front(front,position,raw[31],meter,bs);original=native.forward(model,batch)
            equal_tree(original,slots[buffer_for(127)]['task'])
            expected=np.asarray([v for fusion in model.fusions for v in fusion.last_diagnostics.values()],dtype=np.float32)
            assert np.array_equal(expected,diagnostics[-1])
        base.write(folder/'full_output_equivalence.json',dict(status='passed',original_last_full_tree_vs_graph_bitwise_equal=True,
            all_16_last_field_diagnostics_equal_original=True,all_fields_diagnostic_tensors_saved=True,validation_after_timing=True,no_forward_omitted=True))
        return array
    except BaseException as error:
        base.write(folder/'interruption.json',dict(status='failed_no_qualified_cost',error=repr(error),complete_field_events_enqueued=len(events),capture=capture,no_GPU_readback_attempted_after_failure=True));raise
    finally:
        restore()
        for handle in handles:handle.remove()

graphbase.graph_timed=split_timed

def worker(args):
    graphbase.worker(args)
    p=args.output/'identity.json';identity=json.loads(p.read_text());identity.update(profile='LG_original_AST_prefix_tail_double_buffer_graph_20261026_v1',original_tail_AST_unchanged=True,two_buffers_not_neural_cache=True);base.write(p,identity)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['suite','worker'],required=True)
    parser.add_argument('--system',choices=['lg'],default='lg');parser.add_argument('--repeat',type=int);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();args.output=args.output.resolve();(graphbase.suite if args.phase=='suite' else worker)(args)
