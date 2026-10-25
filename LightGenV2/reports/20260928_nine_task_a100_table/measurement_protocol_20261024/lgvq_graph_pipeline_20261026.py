"""LG original complete forward, less Python dispatch via CUDA Graph replay.

First THREE REAL 16-video fields execute eagerly, stay in outputs and clocks.
Capture occurs between measured fields 3 and 4 and its cost stays in the full
continuous window. There is no extra/dummy/discarded inference or best-window
selection. Every subsequent field copies new raw inputs and reexecutes the
captured ORIGINAL full forward; no neural-feature/prediction cache.
"""
import argparse
import ast
from collections import deque
import inspect
import json
import os
from pathlib import Path
import time
import textwrap
from types import MethodType

import lgvq_hoisted_pipeline_20261026 as hoisted

base=hoisted.base
PREVIOUS_SOURCES=base.source_map
PREVIOUS_ENTRY=Path(base.__file__)
base.__file__=__file__
base.PIPE='/DATA/DATA1/guest3/.cache/lgvq_graph_pipeline_20261026'
EAGER_REAL_FIELDS=3


def execution_plan(fields=128):
    assert fields>EAGER_REAL_FIELDS
    return ['eager_real' if i<EAGER_REAL_FIELDS else 'graph_real' for i in range(fields)]


def sources():
    result=PREVIOUS_SOURCES()
    for p in (PREVIOUS_ENTRY,Path(__file__)):
        result[p.relative_to(base.REPO).as_posix()]=base.sha(p)
    return result


base.source_map=sources


def constant_allocation_AST(source):
    """Replace ONLY allocation of immutable scalars/vectors, not phase math."""
    tree=ast.parse(textwrap.dedent(source));count=0
    class Replace(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='new_tensor':
                assert len(node.args)==1 and not node.keywords
                count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_graph_constant',ctx=ast.Load()),
                    args=[node.func.value,node.args[0]],keywords=[]),node)
            return node
    tree=Replace().visit(tree);assert count==1
    return ast.fix_missing_locations(tree)


def prepare_graph_constants(model):
    import torch
    module=inspect.getmodule(type(model).forward)
    originals={name:getattr(module,name) for name in ('_phase_modulation','_slotwise_routing_statistics')}
    parameter=next(model.parameters());cache={}
    def key(tensor,value):
        data=tuple(value) if isinstance(value,(tuple,list)) else float(value)
        return str(tensor.device),tensor.dtype,data
    for value in (model.settings.unmodulated_power_fraction_eval,(1.0,2.0,4.0,8.0)):
        tensor=parameter.new_tensor(value,dtype=torch.float32)
        cache[key(tensor,value)]=tensor
        assert torch.equal(tensor,parameter.new_tensor(value,dtype=torch.float32))
    def constant(tensor,value):
        return cache[key(tensor,value)]  # Unknown dtype/value fails, no fallback.
    for name,original in originals.items():
        namespace=dict(original.__globals__,_graph_constant=constant)
        exec(compile(constant_allocation_AST(inspect.getsource(original)),inspect.getsourcefile(original)+'::constant_only','exec'),namespace)
        setattr(module,name,namespace[name])
    fusion_originals=[]
    for fusion in model.fusions:
        original=fusion.forward;namespace=dict(original.__func__.__globals__)
        exec(compile(deferred_diagnostics_AST(inspect.getsource(original)),inspect.getsourcefile(original)+'::deferred_diagnostics','exec'),namespace)
        fusion_originals.append((fusion,original));fusion.forward=MethodType(namespace['forward'],fusion)
    def restore():
        for name,original in originals.items():setattr(module,name,original)
        for fusion,original in fusion_originals:fusion.forward=original
    return restore,dict(only_immutable_new_tensor_allocations_precreated=True,
        phase_fraction=float(model.settings.unmodulated_power_fraction_eval),routing_bits=[1.0,2.0,4.0,8.0],
        dtype='torch.float32',no_phase_or_route_result_cache=True,
        diagnostic_CPU_scalar_reads_deferred=True,all_diagnostic_GPU_arithmetic_retained=True,
        original_functions_restored_for_post_timing_reference=True)


def deferred_diagnostics_AST(source):
    """Keep every GPU statistic; defer ONLY float(Tensor) host synchronization."""
    tree=ast.parse(textwrap.dedent(source));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='last_diagnostics' for t in node.targets):
            assert isinstance(node.value,ast.Dict) and len(node.value.values)==4
            for i,value in enumerate(node.value.values):
                assert isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='float' and len(value.args)==1
                node.value.values[i]=value.args[0];changed+=1
    assert changed==4
    return ast.fix_missing_locations(tree)


def graph_timed(model,front,position,raw,guard,folder):
    import numpy as np
    import torch
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native_front
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    from lgvq_live_frontend import equal_tree
    reference=json.loads((folder.parent/'external_preparation.json').read_text())['rows'][0]
    position=position.to(dtype=torch.bfloat16)
    restore_constants,constant_receipt=prepare_graph_constants(model)
    base.write(folder/'graph_constants.json',constant_receipt)
    static={k:torch.empty_like(raw[0][k]) for k in ('pixels','quality','ids','mask')}
    stream=torch.cuda.Stream();graph=torch.cuda.CUDAGraph();captured_output=None
    origin=torch.cuda.Event(enable_timing=True);origin.record();origin.synchronize()
    events=[];outputs=[];diagnostic_outputs=[];tokens=[];pending=deque();capture=None
    counter=dict(passes=0);handles=[]
    for name in ('parallel_router','parallel_optics','serial_router','serial_optics'):
        def count(*args):counter['passes']+=1
        handles.append(getattr(model,name).propagation.register_forward_pre_hook(count))
    def full_forward():
        batch=base.native_lg_front(front,position,static,hoisted.minimal.NoInnerTimer(),stream)
        task=native.forward(model,batch)
        diagnostics=torch.stack([value for fusion in model.fusions for value in fusion.last_diagnostics.values()])
        return dict(task=task,diagnostics=diagnostics)
    host_start=time.perf_counter()
    try:
        with torch.inference_mode(),torch.cuda.stream(stream):
            for field,kind in enumerate(execution_plan()):
                guard.fast();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                a=time.perf_counter()
                rendered=native_front.render_prompt(front.tokenizer,target_name='temporal',prompt=native_front.TARGET_PROMPTS['temporal'])
                tok=front.tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False)
                b=time.perf_counter()
                tokens.append(dict(field=field,start=a,end=b,input_ids=tok['input_ids'].tolist(),mask=tok['attention_mask'].tolist()))
                ids=tok['input_ids'].pin_memory();mask=tok['attention_mask'].bool().pin_memory()
                current=raw[field%32]
                static['pixels'].copy_(current['pixels'],non_blocking=True)
                static['quality'].copy_(current['quality'],non_blocking=True)
                static['ids'].copy_(ids,non_blocking=True);static['mask'].copy_(mask,non_blocking=True)
                if kind=='eager_real':
                    previous=counter['passes'];result=full_forward()
                    assert counter['passes']-previous==6
                    prediction=result['task']['prediction'].clone();diagnostic=result['diagnostics'].clone()
                else:
                    if field==EAGER_REAL_FIELDS:
                        # Prior eager calls are actual first48 measured videos.
                        # No make_graphed_callables (which would run dummy work).
                        capture_start=time.perf_counter();previous=counter['passes']
                        with torch.cuda.graph(graph,stream=stream):captured_output=full_forward()
                        capture_end=time.perf_counter()
                        assert counter['passes']-previous==6
                        capture=dict(first_replayed_field=field,host_start=capture_start,host_end=capture_end,
                            host_ms=(capture_end-capture_start)*1000,captured_original_full_field_propagation_calls=6,
                            capture_records_operations_without_executing_a_discarded_prediction=True,
                            first48_eager_video_results_retained=True)
                    graph.replay();prediction=captured_output['task']['prediction'].clone();diagnostic=captured_output['diagnostics'].clone()
                end.record();events.append(dict(field=field,kind=kind,start=start,end=end));outputs.append(prediction);diagnostic_outputs.append(diagnostic)
                pending.append((end,ids,mask))
                # Bound queued graph replays and pinned-input lifetime.
                if kind=='eager_real':end.synchronize()
                elif len(pending)>3:pending.popleft()[0].synchronize()
                if (field+1)%16==0:base.write(folder/'status.json',dict(status='graph_replay_real_fields',completed_video_calls=(field+1)*16,total=2048))
            events[-1]['end'].synchronize();host_end=time.perf_counter()
        assert capture is not None and len(tokens)==128
        assert all(r['input_ids']==reference['input_ids'] and r['mask']==reference['mask'] for r in tokens)
        rows=[dict(field=r['field'],kind=r['kind'],start_ms=origin.elapsed_time(r['start']),end_ms=origin.elapsed_time(r['end'])) for r in events]
        begin=rows[0]['start_ms'];elapsed=rows[-1]['end_ms']-begin
        array=torch.cat(outputs).float().cpu().numpy()
        assert array.shape==(128,16) and np.isfinite(array).all()
        assert all(np.array_equal(array[:32],array[32*c:32*(c+1)]) for c in (1,2,3))
        diagnostics=torch.stack(diagnostic_outputs).float().cpu().numpy()
        assert diagnostics.shape==(128,16) and np.isfinite(diagnostics).all()
        assert all(np.array_equal(diagnostics[:32],diagnostics[32*c:32*(c+1)]) for c in (1,2,3))
        np.savez_compressed(folder/'graph_outputs.npz',prediction=array,diagnostics=diagnostics)
        base.write(folder/'graph_timing.json',dict(status='complete',video_calls=2048,unique_videos=512,fields=128,frames_per_field=64,
            continuous_GPU_timeline_ms=elapsed,first_GPU_front_start_ms=begin,last_GPU_score_ms=rows[-1]['end_ms'],
            dispatch_to_last_score_host_ms=(host_end-host_start)*1000,time16_ms=elapsed/128,equivalent_frame_ms=elapsed/8192,
            rows=rows,capture=capture,eager_real_fields=3,eager_real_video_calls=48,graph_replays=125,graph_real_video_calls=2000,
            no_dummy_or_discarded_inferences=True,no_fast_window_selection=True,first_eager_calls_and_capture_cost_in_continuous_window=True,
            all_original_model_operations_captured=True,captured_propagation_count_is_capture_contract_not_per_replay_Python_hooks=True,
            actual_complete_software_includes_FFT=True,physical_optical_E2E_measured=False,
            original_function='lab_runtime.forward plus original native frontend',all_input_adapters_inside_captured_original_graph=True,
            raw_inputs_copied_on_every_field=True,all_neural_features_recomputed_on_every_graph_replay=True,
            online_CPU_tokenizer=True,original_dtype_and_precision_unchanged=True,
            fusion_diagnostic_GPU_arithmetic_and_copy_included=True,
            fusion_diagnostic_CPU_readback_batched_after_last_score=True,
            prefixes=[dict(video_calls=n,time16_ms=(rows[n//16-1]['end_ms']-begin)*16/n) for n in base.prefixes()],
            steady=base.steady_report([r['end_ms']-begin for r in rows])))
        base.write(folder/'online_tokenizer.json',dict(records=tokens,CPU_work_sum_ms=sum((r['end']-r['start'])*1000 for r in tokens),
            all_actual_IDs_and_masks_equal_original=True,not_to_be_added_twice_to_continuous_window=True))
        # Last captured full output tree must equal original on its current raw
        # field, not only a pooled/argmax/rounded score. Validation AFTER timing.
        restore_constants()
        with torch.inference_mode(),torch.cuda.stream(stream):
            batch=base.native_lg_front(front,position,raw[31],hoisted.minimal.NoInnerTimer(),stream)
            original=native.forward(model,batch)
            equal_tree(original,captured_output['task'])
            expected_diagnostics=np.asarray([v for fusion in model.fusions for v in fusion.last_diagnostics.values()],dtype=np.float32)
            assert np.array_equal(expected_diagnostics,diagnostics[-1])
        base.write(folder/'full_output_equivalence.json',dict(status='passed',original_last_full_tree_vs_graph_bitwise_equal=True,
            all_16_last_field_diagnostics_equal_original=True,all_fields_diagnostic_tensors_saved=True,
            validation_after_timing=True,no_forward_omitted=True))
        return array
    except BaseException as error:
        # A failed capture may invalidate its CUDA stream: preserve CPU-side
        # evidence without attempting another GPU call or numerical fallback.
        base.write(folder/'interruption.json',dict(status='failed_no_qualified_cost',error=repr(error),
            complete_field_events_enqueued=len(events),online_tokenizer_records=tokens,
            capture_receipt=capture,no_GPU_readback_attempted_after_failure=True))
        raise
    finally:
        restore_constants()
        for handle in handles:handle.remove()


def paired_timed(model,front,position,raw,guard,mode,folder):
    name='control' if mode=='serial' else 'graph'
    output=folder/name;output.mkdir()
    base.write(folder/'status.json',dict(status='paired_LG_GPU_graph_experiment',strategy=name))
    if mode=='serial':return hoisted.minimal.timed(model,front,position,raw,guard,'pipeline',output)
    return graph_timed(model,front,position,raw,guard,output)


base.lg_timed=paired_timed


def worker(args):
    assert args.system=='lg'
    base.worker(args)
    path=args.output/'identity.json';identity=json.loads(path.read_text())
    identity.update(profile='LG_paired_original_pipeline_vs_original_full_CUDA_Graph_20261026_v1',
        nonexclusive_authorized=True,foreign_GPU_processes_observed_not_rejected=True,
        not_uncontended_baseline_performance=True,private_mps_pipe=base.PIPE,
        paired_order=['control','graph'] if args.repeat%2 else ['graph','control'],
        no_model_math_or_precision_change=True,no_dummy_or_discarded_inferences=True,
        initial48_real_video_calls_and_graph_capture_included=True,
        CPU_tokenizer_scope=dict(control='before primary',graph='online per16-video field; leading work retained in host elapsed'),
        Qwen_not_executed=True)
    identity.pop('tokenizer_is_CPU_separate_not_added_to_GPU_timeline',None)
    base.write(path,identity)


def suite(args):
    # Reuse the identical bounded, serialized, LG-only controller. Override
    # ONLY its entry-file global so its children invoke this graph profile.
    namespace=dict(hoisted.suite.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(hoisted.suite),hoisted.__file__+'::graph_controller','exec'),namespace)
    namespace['suite'](args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['suite','worker'],required=True)
    parser.add_argument('--system',choices=['lg'],default='lg');parser.add_argument('--repeat',type=int)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path)
    parser.add_argument('--deadline',type=float,required=True)
    options=parser.parse_args();options.output=options.output.resolve()
    (suite if options.phase=='suite' else worker)(options)
