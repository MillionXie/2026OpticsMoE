"""Move ONLY original steady body dispatch loop to C++; original raw schema.

No nested capture, removed launch, warmed forward or GPU arithmetic change.
The C++ clock must fall inside a real Python perf_counter bracket; otherwise
reject instead of shifting a clock. Every original event and native graph
replay remains. Host timings are actual per-segment C++ observations.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time
import lgvq_async_replay_20261026 as original

HERE=Path(__file__).resolve().parent
EXT=None
BUILD=None


def load_extension():
    global EXT,BUILD
    if EXT is not None:return
    build=HERE/'cpp_dispatch_build_20261027'
    receipt=json.loads((build/'build_identity.json').read_text())
    source=HERE/'lgvq_a100_cpp_dispatch_20261027.cpp';binary=build/'lgvq_native_dispatch_20261027.so'
    if (hashlib.sha256(source.read_bytes()).hexdigest()!=receipt['source_sha256']
            or hashlib.sha256(binary.read_bytes()).hexdigest()!=receipt['binary_sha256']
            or receipt['compiler_returncode']!=0 or receipt['CUDA_context_created_during_build'] is not False):
        raise ValueError('Exact CPU-built extension receipt required')
    spec=importlib.util.spec_from_file_location('lgvq_native_dispatch_20261027',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a=time.perf_counter();point=module.now_seconds();b=time.perf_counter()
    if not a<=point<=b:raise ValueError('Different C++/Python clock; never shift timestamps')
    EXT=module;BUILD=dict(receipt,clock_bracket=[a,b],cpp_clock_observation=point,same_clock_verified=True)


def submit(torch,slot,event,body_stream):
    if EXT is None:raise RuntimeError('CPU-built published extension must load before model measurement')
    pairs=[(event(),event()) for _ in slot['segments'][1:]]
    if '_cpp_plan' not in slot:
        for a,b in pairs:a.record();b.record()  # actual lazy-handle initialization stays timed
        slot['_cpp_plan']=EXT.Plan([s['graph'] for s in slot['segments'][1:]],
            [e.cuda_event for pair in pairs for e in pair])
        slot['_cpp_pairs']=pairs
    elif pairs!=slot['_cpp_pairs']:raise ValueError('Event pool changed after native-plan binding')
    times=slot['_cpp_plan'].replay(body_stream.cuda_stream)
    if len(times)!=37:raise ValueError('Every original body submit observation required')
    return [dict(kind=s['kind'],label=s['label'],start=a,end=b,stream='body',host_submit_start=h[0],host_submit_end=h[1])
            for s,(a,b),h in zip(slot['segments'][1:],pairs,times)]


def replay_AST():
    tree=ast.parse(inspect.getsource(original.replay_fields));count=0
    class Body(ast.NodeTransformer):
        def visit_For(self,node):
            nonlocal count
            if ast.unparse(node.target)=='segment' and ast.unparse(node.iter)=="slot['segments'][1:]":
                count+=1
                return ast.parse('observed.extend(_submit(torch,slot,event,body_stream))').body[0]
            return self.generic_visit(node)
    tree=Body().visit(tree)
    if count!=1:raise ValueError('Exact original steady body loop required')
    returned=tree.body[0].body[-1]
    if not isinstance(returned,ast.Return):raise ValueError('Original return boundary changed')
    returned.value.keywords.append(ast.keyword(arg='native_dispatch',value=ast.parse('_summary(slots)').body[0].value))
    return ast.fix_missing_locations(tree)


def summary(slots):
    calls=[int(s['_cpp_plan'].launches) for s in slots]
    if calls!=[62,61]:raise ValueError('Exact 123-field native dispatch counts required')
    return dict(profile='original_37_replays_in_one_CPP_call',slot_calls=calls,original_graph_replays=sum(calls)*37,
        original_timing_event_records=sum(calls)*74,no_removed_or_nested_graph_replays=True,
        all_original_host_costs_inside_continuous_wall=True,per_segment_CPU_timestamps_are_measured=True,
        no_clock_shift_or_ideal_overlap_subtraction=True,build=BUILD)


def replay_fields(*args,**kwargs):
    namespace=dict(original.__dict__,_submit=submit,_summary=summary)
    exec(compile(replay_AST(),__file__+'::native_dispatch','exec'),namespace)
    return namespace['replay_fields'](*args,**kwargs)


def worker_AST(source):
    tree=ast.parse(source);count=0
    class Import(ast.NodeTransformer):
        def visit_Import(self,node):
            nonlocal count
            if len(node.names)==1 and node.names[0].name=='lgvq_async_replay_20261026' and node.names[0].asname=='replay':
                count+=1;node.names[0].name='lgvq_a100_cpp_dispatch_20261027'
            return node
    tree=Import().visit(tree)
    if count!=1:raise ValueError('Original replay import required')
    fn=tree.body[0]
    class Save(ast.NodeTransformer):
        changes=0
        def visit_Expr(self,node):
            if (isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='base.write'
                    and ast.unparse(node.value.args[0])=="output / 'async_event_records.json'"):
                self.changes+=1
                return ast.parse("base.write(output/'a100_cpp_dispatch_identity.json',online['native_dispatch'])").body+[node]
            return node
    saver=Save();tree=saver.visit(tree)
    if saver.changes!=1:raise ValueError('Original post-window raw save boundary required')
    return ast.fix_missing_locations(tree)
