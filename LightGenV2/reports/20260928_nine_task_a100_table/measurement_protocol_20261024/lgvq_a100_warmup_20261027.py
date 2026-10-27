"""Exactly ONE explicitly authorized extra real field, separate warm profile.

The original measured 128 fields/2048 calls, three eager fields and two real
captures remain. Warming is not hidden: save its scores, six-pass events,
elapsed time, and the inclusive software wall charged to the measured workload.
No original cold result is relabelled or replaced.
"""
import ast
import hashlib
import time
import lgvq_async_raw_adapter_20261026 as original_adapter

PROFILE='ONE_EXTRA_REAL_WARMUP_FIELD_DIAGNOSTIC'


def prewarm(torch,model,raw,prefix,tail,tokenize,fs,bs,tape_factory,install,reference,guard,sampler):
    import numpy as np
    guard.fast();sampler.check()
    mono_before=time.monotonic();started=time.perf_counter();mono_after=time.monotonic()
    torch.cuda.synchronize()  # input/model readiness, explicitly charged to the extra field
    tape=tape_factory(torch,bs);tape.eager=[];tape.eager_host=[]
    restore,ledger=install(model,tape)
    try:
        with torch.inference_mode():
            ids,mask,token=tokenize(-1)
            ids=ids.pin_memory();mask=mask.bool().pin_memory()
            with torch.cuda.stream(fs):
                origin=torch.cuda.Event(enable_timing=True);origin.record()
                current=dict(raw[0],ids=ids.to('cuda',non_blocking=True),mask=mask.to('cuda',non_blocking=True))
                prepared=prefix(current)
                ready=torch.cuda.Event(enable_timing=True);ready.record()
            with torch.cuda.stream(bs):
                bs.wait_event(ready)
                result=tail(prepared);prediction=result['prediction'].clone()
                finished=torch.cuda.Event(enable_timing=True);finished.record()
            finished.synchronize()
            array=prediction.float().cpu().numpy().copy()
            events=[dict(kind=kind,label=label,cuda_start_ms=origin.elapsed_time(a),cuda_end_ms=origin.elapsed_time(b),
                         host_submit_start=h[0],host_submit_end=h[1])
                    for (kind,label,a,b),h in zip(tape.eager,tape.eager_host)]
            gpu_ms=origin.elapsed_time(finished)
        with np.load(reference,allow_pickle=False) as sealed:expected=sealed['prediction'][0:1]
        if array.dtype!=np.float32 or array.shape!=(1,16) or array.tobytes()!=expected.tobytes():
            raise ValueError('Extra real warm field differs from original A100 reference')
        if sum(e['kind']=='optical_simulation' for e in events)!=6:
            raise ValueError('Warmup must execute all six complete coherent passes')
        # Never reuse a warm feature/prediction as a measured input or output.
        del current,prepared,result,prediction,ids,mask,origin,ready,finished
    finally:restore()
    ended=time.perf_counter();guard.fast();sampler.check()
    return dict(profile=PROFILE,extra_real_fields=1,extra_video_calls=16,extra_frames=64,
        actual_full_coherent_optical_passes=6,source_real_field=0,tokenizer=token,
        host_start=started,host_end=ended,host_elapsed_ms=(ended-started)*1000,
        perf_monotonic_alignment_bracket=[mono_before,mono_after],GPU_envelope_ms=gpu_ms,
        six_pass_events=events,boundary_ledger=ledger,warm_scores_bitwise_equal_reference=True,
        neural_features_or_predictions_reused=False,original_cold_scope_qualified=False),array


def save(output,warm,array):
    import json
    import numpy as np
    with (output/'extra_warmup_identity.json').open('x',encoding='utf8') as stream:
        json.dump(warm,stream,ensure_ascii=True,indent=2)
    if (output/'extra_warmup_outputs.npz').exists():raise FileExistsError('Preserve warm scores')
    np.savez_compressed(output/'extra_warmup_outputs.npz',prediction=array)


def accounting(raw):
    warm=raw['extra_warmup'];start=raw['host_start'];end=raw['host_end']
    if (raw.get('timing_profile')!=PROFILE or raw.get('explicit_warmup_forwards')!=1
            or warm['profile']!=PROFILE or warm['extra_real_fields']!=1 or warm['extra_video_calls']!=16
            or warm['actual_full_coherent_optical_passes']!=6 or warm['warm_scores_bitwise_equal_reference'] is not True
            or warm['neural_features_or_predictions_reused'] is not False
            or not warm['host_start']<warm['host_end']<=start<end):
        raise ValueError('Exactly one separately charged real pre-window warm field required')
    return dict(warmup_software_ms=(warm['host_end']-warm['host_start'])*1000,
        warm_to_main_gap_ms=(start-warm['host_end'])*1000,measured_main_software_ms=(end-start)*1000,
        inclusive_software_ms=(end-warm['host_start'])*1000,
        inclusive_software_ms_charged_per_measured_16=(end-warm['host_start'])*1000/128,
        measured_video_calls=2048,total_executed_video_calls=2064,
        inclusive_wall_is_actual_software_NOT_optical_projection=True)


def adapt(raw):
    info=accounting(raw)
    trace,report=original_adapter.adapt(raw)
    trace['timing_profile']=PROFILE
    report.update(system='LG_warmed_diagnostic_fixed_optics_counterfactual',timing_profile=PROFILE,
        qualified_for_final_hybrid_table=False,original_cold_scope_qualified=False,
        extra_pre_window_real_fields=1,warmup_accounting=info)
    return trace,report


def worker_AST(source):
    tree=ast.parse(source);counts=dict(imports=0,warmup=0,save=0,raw=0,identity=0)
    class Warm(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module=='lgvq_async_raw_adapter_20261026' and [(a.name,a.asname) for a in node.names]==[('adapt',None)]:
                counts['imports']+=1;node.module='lgvq_a100_warmup_20261027'
            return node
        def visit_Expr(self,node):
            text=ast.unparse(node)
            if text=='torch.cuda.synchronize()':
                counts['warmup']+=1
                return ast.parse('_warm_info,_warm_array=_prewarm(torch,model,raw,prefix,tail,tokenize,fs,bs,ObservedEagerTape,install,reference,guard,sampler)').body+[node]
            if (isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='base.write'
                    and ast.unparse(node.value.args[0])=="output / 'async_event_records.json'"):
                counts['save']+=1
                return ast.parse('_save_warm(output,_warm_info,_warm_array)').body+[node]
            if (isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='base.write'
                    and ast.unparse(node.value.args[0])=="output / 'async_identity.json'"):
                counts['identity']+=1
                node.value.args[1].keywords.extend([
                    ast.keyword(arg='timing_profile',value=ast.Constant(PROFILE)),
                    ast.keyword(arg='additional_pre_window_real_fields',value=ast.Constant(1))])
            return node
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='explicit_warmup_forwards' for k in node.keywords):
                counts['raw']+=1
                for key in node.keywords:
                    if key.arg=='explicit_warmup_forwards':key.value=ast.Constant(1)
                node.keywords.extend([ast.keyword(arg='timing_profile',value=ast.Constant(PROFILE)),
                    ast.keyword(arg='extra_warmup',value=ast.Name(id='_warm_info',ctx=ast.Load()))])
            return self.generic_visit(node)
    # Expr overrides do not recurse, so visit the raw dict independently.
    tree=Warm().visit(tree)
    if counts!=dict(imports=1,warmup=1,save=1,raw=1,identity=1):
        raise ValueError('Pinned original one-warmup insertion boundaries required: '+str(counts))
    return ast.fix_missing_locations(tree)
