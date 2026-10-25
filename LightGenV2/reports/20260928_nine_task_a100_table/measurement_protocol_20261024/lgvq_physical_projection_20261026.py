"""Full unchanged LG math, explicit optical-simulation replacement ledger.

Six original amplitude-to-detector simulation prefixes are observed intact.
Their replay GPU spans and their own capture-construction CPU spans are removed
from the projection, replaced by six1.4kHz SLM+exposure+CCD cycles per16videos.
ROI extraction/arrangement and electronic residual branches execute unchanged
but are excluded per the latest user scope. Learned readout, fusion and head
remain charged. No claim of physical hardware acquisition or unproved overlap.
"""
import ast
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time
from types import MethodType

def simulation_prefix_AST(source):
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    ends=[i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
          and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='_detector']
    assert len(ends)==1;cut=ends[0]+1;original=ast.dump(ast.Module(body=fn.body[:cut],type_ignores=[]))
    context=ast.Call(func=ast.Attribute(value=ast.Name(id='_projection_tape',ctx=ast.Load()),attr='simulation',ctx=ast.Load()),args=[ast.Name(id='_projection_label',ctx=ast.Load())],keywords=[])
    fn.body=[ast.With(items=[ast.withitem(context_expr=context)],body=fn.body[:cut])]+fn.body[cut:]
    assert ast.dump(ast.Module(body=fn.body[0].body,type_ignores=[]))==original
    fn.decorator_list=[]
    return ast.fix_missing_locations(tree),dict(prefix_statements=cut,prefix_original_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),original_prefix_math_unchanged=True,
        boundary='from original field/canvas/phase simulation start through original detector intensity and guard; post-detector ROI, router decisions, learned readout retained')

def roi_AST(source,router=False):
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    if router:
        start=next(i+1 for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='_detector')
        end=next(i+1 for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='energy' for t in n.targets))
    else:
        start=0;end=next(i+1 for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Attribute) and t.value.attr=='last_ccd' for t in n.targets))
    original=ast.dump(ast.Module(body=fn.body[start:end],type_ignores=[]))
    context=ast.Call(func=ast.Attribute(value=ast.Name(id='_projection_tape',ctx=ast.Load()),attr='excluded',ctx=ast.Load()),args=[ast.Name(id='_projection_label',ctx=ast.Load()),ast.Constant('detector_ROI')],keywords=[])
    fn.body=fn.body[:start]+[ast.With(items=[ast.withitem(context_expr=context)],body=fn.body[start:end])]+fn.body[end:]
    fn.decorator_list=[]
    return ast.fix_missing_locations(tree),dict(ROI_original_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),ROI_math_executes_unchanged=True,learned_readout_and_router_decisions_retained=True)

class Tape:
    def __init__(self,torch,stream):
        self.torch=torch;self.stream=stream;self.capturing=False;self.segments=[];self.eager=[];self.context=None
    def start(self,kind,label):
        g=self.torch.cuda.CUDAGraph();a=time.perf_counter();ctx=self.torch.cuda.graph(g,stream=self.stream);ctx.__enter__()
        self.context=(ctx,g,kind,label,a)
    def close(self):
        ctx,g,kind,label,a=self.context;ctx.__exit__(None,None,None);b=time.perf_counter()
        self.segments.append(dict(graph=g,kind=kind,label=label,capture_host_start=a,capture_host_end=b,capture_host_ms=(b-a)*1000));self.context=None
    @contextmanager
    def simulation(self,label):
        with self.excluded(label,'optical_simulation'):yield
    @contextmanager
    def excluded(self,label,kind):
        if self.capturing:
            self.close();self.start(kind,label)
            try:yield
            finally:self.close();self.start('electronics','after_'+label)
        else:
            a=self.torch.cuda.Event(enable_timing=True);b=self.torch.cuda.Event(enable_timing=True);a.record()
            try:yield
            finally:b.record();self.eager.append((kind,label,a,b))

def install_simulation_boundaries(model,tape):
    restore=[];ledger=[]
    for name,method in [('parallel_router','forward'),('parallel_optics','expert'),('parallel_optics','global_path'),('serial_router','forward'),('serial_optics','expert'),('serial_optics','global_path')]:
        owner=getattr(model,name);original=getattr(owner,method);label=name+'.'+method
        source=inspect.getsource(original)
        if name.endswith('router'):
            roi_tree,roi_receipt=roi_AST(source,True);source=ast.unparse(roi_tree);ledger.append(dict(stage=label,**roi_receipt))
        tree,receipt=simulation_prefix_AST(source);namespace=dict(original.__func__.__globals__,_projection_tape=tape,_projection_label=label)
        exec(compile(tree,inspect.getsourcefile(original)+'::projection_prefix','exec'),namespace)
        fn=namespace[method];fn.__defaults__=original.__func__.__defaults__;fn.__kwdefaults__=original.__func__.__kwdefaults__;functools.update_wrapper(fn,original)
        setattr(owner,method,MethodType(fn,owner));restore.append((owner,method,original));ledger.append(dict(stage=label,**receipt))
    for name,method in [('parallel_optics','_read_frames'),('serial_optics','_read_videos')]:
        owner=getattr(model,name);original=getattr(owner,method);label=name+'.'+method
        tree,receipt=roi_AST(inspect.getsource(original));namespace=dict(original.__func__.__globals__,_projection_tape=tape,_projection_label=label)
        exec(compile(tree,inspect.getsourcefile(original)+'::ROI_boundary','exec'),namespace)
        fn=namespace[method];fn.__defaults__=original.__func__.__defaults__;fn.__kwdefaults__=original.__func__.__kwdefaults__
        setattr(owner,method,MethodType(fn,owner));restore.append((owner,method,original));ledger.append(dict(stage=label,**receipt))
    def residual_wrapper(original,label):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            with tape.excluded(label,'electronic_residual'):return original(*args,**kwargs)
        return wrapped
    for group in ('vision_routes','language_routes'):
        for i,owner in enumerate(getattr(model,group)):
            original=owner.forward;label=group+'.'+str(i)
            owner.forward=residual_wrapper(original,label);restore.append((owner,'forward',original));ledger.append(dict(stage=label,executes_unchanged=True,timing_excluded=True,fusion_still_charged=True))
    def close():
        for owner,name,original in reversed(restore):setattr(owner,name,original)
    return close,ledger

def worker(args,guard,N):
    import os
    import numpy as np
    import torch
    import lgvq_gpu_pipeline_20261026 as base
    # Import approved capture helpers without changing the active suite's pipe,
    # source function, entry identity or guard through their legacy module globals.
    saved={k:getattr(base,k) for k in ('__file__','PIPE','source_map','model_guard')}
    import lgvq_graph_pipeline_20261026 as approved
    for k,v in saved.items():setattr(base,k,v)
    import lgvq_live_frontend as live
    from lgvq_pipeline_demo_20261025 import PreparedFront
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    base.UNIQUE=N if N<=512 else 512;base.CYCLES=N//base.UNIQUE;base.TOTAL=N
    front=live.LiveFront('cuda:0');model,_=native.load_model('temporal',live.TASK/'runs/simulation/multivideo16x4_rank_s163/best_checkpoint.pt','cuda:0');model.requires_grad_(False)
    guard.bind();assert torch.cuda.get_device_properties(0).multi_processor_count==108
    fixed=PreparedFront(front);position=fixed.position4.repeat(16,1).to(dtype=torch.bfloat16)
    raw,files=base.prep_lg(front,[r for r in read_manifest(live.MANIFEST) if r.split=='test'],args.output,guard)
    base.write(args.output/'input_files.json',[dict(**r,bytes=Path(r['video_path']).stat().st_size,sha256=base.sha(Path(r['video_path']))) for r in files])
    paths=[Path(__file__),Path(approved.__spec__.origin),Path(approved.hoisted.__file__),Path(approved.hoisted.minimal.__file__)]
    frozen={str(p.relative_to(base.REPO)):base.sha(p) for p in paths}
    deployment=json.loads((args.output.parent/'LG_module_deployment.json').read_text());assert deployment['sha256']==base.sha(Path(__file__))
    base.write(args.output/'projection_identity.json',dict(sources=frozen,module_deployment=deployment,original_model_math_unchanged=True,electronic_residual_and_ROI_excluded=True,fusion_and_learned_readout_charged=True,N=N,unique_videos=base.UNIQUE))
    restore_constants,constants=approved.prepare_graph_constants(model);base.write(args.output/'approved_optimization.json',constants)
    stream=torch.cuda.Stream();tape=Tape(torch,stream);restore_boundaries,ledger=install_simulation_boundaries(model,tape)
    base.write(args.output/'optical_replacement_ledger.json',ledger)
    static={k:torch.empty_like(raw[0][k]) for k in ('pixels','quality','ids','mask')};meter=approved.hoisted.minimal.NoInnerTimer()
    def full_forward():
        batch=base.native_lg_front(front,position,static,meter,stream);result=native.forward(model,batch)
        diagnostic=torch.stack([v for f in model.fusions for v in f.last_diagnostics.values()])
        return dict(task=result,diagnostic=diagnostic)
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as text
    records=[];outputs=[];diagnostics=[];capture=None;host_start=time.perf_counter()
    try:
        with torch.inference_mode(),torch.cuda.stream(stream):
            for field in range(N//16):
                guard.fast();field_start=time.perf_counter();a=time.perf_counter()
                rendered=text.render_prompt(front.tokenizer,target_name='temporal',prompt=text.TARGET_PROMPTS['temporal'])
                token=front.tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False);b=time.perf_counter()
                # No prior GPU field is pending here. Exclude exactly this
                # input H2D, not electronics or a hidden inference warmup.
                h2d_start=time.perf_counter();static['ids'].copy_(token['input_ids'].pin_memory(),non_blocking=True);static['mask'].copy_(token['attention_mask'].bool().pin_memory(),non_blocking=True);torch.cuda.synchronize();h2d_end=time.perf_counter()
                current=raw[field%len(raw)]
                for k in ('pixels','quality'):static[k].copy_(current[k],non_blocking=True)
                spans=[]
                if field<3:
                    tape.eager=[];result=full_forward();events=list(tape.eager)
                else:
                    if field==3:
                        ca=time.perf_counter();tape.capturing=True;tape.start('electronics','frontend')
                        try:captured=full_forward();tape.close()
                        finally:tape.capturing=False
                        cb=time.perf_counter();assert len(tape.segments)==33 and sum(s['kind']=='optical_simulation' for s in tape.segments)==6
                        assert sum(s['kind']=='detector_ROI' for s in tape.segments)==6 and sum(s['kind']=='electronic_residual' for s in tape.segments)==4
                        capture=dict(host_start=ca,host_end=cb,total_ms=(cb-ca)*1000,segments=[{k:v for k,v in s.items() if k!='graph'} for s in tape.segments],no_extra_or_discarded_inference=True)
                    events=[]
                    for segment in tape.segments:
                        if segment['kind']!='electronics':
                            x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record();segment['graph'].replay();y.record();events.append((segment['kind'],segment['label'],x,y))
                        else:segment['graph'].replay()
                    result=captured
                outputs.append(result['task']['prediction'].clone());diagnostics.append(result['diagnostic'].clone())
                end=torch.cuda.Event();end.record();end.synchronize();score_ready=time.perf_counter();book_start=time.perf_counter()
                assert len(events)==16
                for kind,label,x,y in events:spans.append(dict(kind=kind,stage=label,cuda_ms=x.elapsed_time(y)))
                row=dict(field=field,field_start=field_start,score_ready=score_ready,tokenizer_start=a,tokenizer_end=b,
                    input_ids=token['input_ids'].tolist(),mask=token['attention_mask'].tolist(),H2D_start=h2d_start,H2D_end=h2d_end,
                    optical_simulation_spans=spans,bookkeeping_start=book_start)
                records.append(row)
                if (field+1)%16==0:base.write(args.output/'status.json',dict(status='LG_actual_electronics_plus_fixed_optics',fields=field+1,total=N//16))
                row['bookkeeping_end']=time.perf_counter()
        full=(records[-1]['score_ready']-host_start)*1000
        removed_h2d=sum((r['H2D_end']-r['H2D_start'])*1000 for r in records)
        removed_book=sum((r['bookkeeping_end']-r['bookkeeping_start'])*1000 for r in records[:-1])
        removed_by_kind={kind:sum(s['cuda_ms'] for r in records for s in r['optical_simulation_spans'] if s['kind']==kind) for kind in ('optical_simulation','detector_ROI','electronic_residual')}
        capture_by_kind={kind:sum(s['capture_host_ms'] for s in capture['segments'] if s['kind']==kind) for kind in removed_by_kind}
        sim=removed_by_kind['optical_simulation']
        sim_capture=sum(s['capture_host_ms'] for s in capture['segments'] if s['kind']=='optical_simulation')
        electronic=full-removed_h2d-removed_book-sum(removed_by_kind.values())-sum(capture_by_kind.values());assert electronic>0
        fixed_pass=1000/1400+.300+.0307;optical=N//16*6*fixed_pass;projected=electronic+optical
        array=torch.cat(outputs).float().cpu().numpy();diag=torch.stack(diagnostics).float().cpu().numpy()
        assert array.shape==(N//16,16) and np.isfinite(array).all()
        unique_fields=base.UNIQUE//16
        assert all(np.array_equal(array[:unique_fields],array[k:k+unique_fields]) for k in range(unique_fields,N//16,unique_fields))
        # Validation-only sealed reference, never used by online forward.
        reference=live.TASK/'runs/simulation/a100_lgvq_graph_pipeline_20261026_03/lg_repeat01/graph/graph_outputs.npz'
        with np.load(reference,allow_pickle=False) as f:expected=f['prediction']
        assert np.array_equal(array,expected[:N//16])
        np.savez_compressed(args.output/'outputs.npz',prediction=array,diagnostics=diag)
        base.write(args.output/'timing.json',dict(status='complete',system='LG_projected',video_calls=N,unique_videos=base.UNIQUE,fields=N//16,frames_per_field=64,
            full_software_host_ms=full,removed_original_H2D_ms=removed_h2d,removed_measurement_only_bookkeeping_ms=removed_book,
            removed_optical_simulation_replay_ms=sim,removed_optical_simulation_capture_construction_ms=sim_capture,
            excluded_GPU_ms_by_kind=removed_by_kind,excluded_capture_CPU_ms_by_kind=capture_by_kind,
            ROI_and_electronic_residual_execute_but_not_charged=True,fusion_and_learned_readout_charged=True,
            charged_actual_electronic_ms=electronic,fixed_optical_ms=optical,fixed_pass_ms=fixed_pass,
            projection_ms=projected,time16_ms=projected*16/N,equivalent_frame_ms=projected/(4*N),records=records,capture=capture,
            first48_actual_video_results_and_all_electronic_capture_cost_retained=True,physical_hardware_E2E_measured=False,
            native_math_and_all_electronic_routes_retained=True,simulation_replaced_not_added_twice=True,
            source_reference_for_validation_only=dict(path=str(reference),sha256=base.sha(reference),all_predictions_bitwise_equal=True),
            no_unproved_overlap_deducted=True,no_fast_window_or_round_selection=True))
        assert frozen=={str(p.relative_to(base.REPO)):base.sha(p) for p in paths}
    finally:
        restore_boundaries();restore_constants()
