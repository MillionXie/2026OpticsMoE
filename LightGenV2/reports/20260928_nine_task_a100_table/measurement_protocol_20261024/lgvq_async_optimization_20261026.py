"""Optional fresh-run runtime policy and explicit electronic phase boundaries.

No inference, warmup, cached features, dtype changes or GPU actions on import.
EAGER module loading is selected before worker startup, never by a dummy forward.
The capture policy keeps synchronization and every capture but does not force a
full Python GC / allocator-cache purge. All remaining preparation stays timed.
"""
import ast
import copy
import hashlib
from contextlib import contextmanager
import time


def separate_prompt_adapter(module):
    """Move four unchanged independent statements, never simplify arithmetic."""
    module=copy.deepcopy(module);prefix=next(n for n in module.body if n.name=='prepare_native_prefix')
    def assigns(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    start=next(i for i,n in enumerate(prefix.body) if assigns(n,'prompt_tokens'))
    end=next(i for i,n in enumerate(prefix.body) if assigns(n,'qwen_vision'))
    statements=prefix.body[start:end]
    if len(statements)!=4:raise ValueError('Original four prompt-adapter statements required')
    names=('prompt_tokens','prompt_valid','prompt_summary','prompt_scale','prompt_shift')
    fn=ast.parse('def prepare_text_adapter(self,language_tokens,language_mask):\n    pass').body[0]
    fn.body=copy.deepcopy(statements)+[ast.Return(ast.Dict(
        keys=[ast.Constant(n) for n in names],values=[ast.Name(id=n,ctx=ast.Load()) for n in names]))]
    if ast.dump(ast.Module(body=fn.body[:-1],type_ignores=[]))!=ast.dump(ast.Module(body=statements,type_ignores=[])):
        raise ValueError('Prompt adapter arithmetic changed')
    prefix.args.kwonlyargs.append(ast.arg(arg='_async_text_prefix'));prefix.args.kw_defaults.append(ast.Constant(None))
    bindings=ast.parse('\n'.join(f'{n}=_async_text_prefix[{n!r}]' for n in names)).body
    prefix.body=prefix.body[:start]+bindings+prefix.body[end:];module.body.append(fn)
    digest=hashlib.sha256(ast.dump(ast.Module(body=statements,type_ignores=[])).encode()).hexdigest()
    return ast.fix_missing_locations(module),dict(original_prompt_adapter_AST_sha256=digest,
        original_prompt_adapter_arithmetic_unchanged=True,executed_once_per_real_field=True,
        text_adapters_join_before_prompt_conditioned_vision=True)


def parallel_front(torch,front,position,raw,model,text_fn,text_stream):
    """Real same-field text branch, fork after input-copy and join before use."""
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
    text_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(text_stream):
        language=front.embedding(raw['ids']).half()
        prepared=text_fn(model,language,raw['mask'])
        ready=torch.cuda.Event();ready.record()
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        hidden=front.visual.patch_embed(raw['pixels'])
        hidden=hidden+position.to(dtype=hidden.dtype)
        pooled=native.pool_qwen_front_tokens(hidden,image_count=64,output_grid=7)
    torch.cuda.current_stream().wait_event(ready)
    return dict(vision_tokens=pooled.reshape(1,16,4,49,1024).half(),quality_tokens=raw['quality'],
                language_tokens=language,language_mask=raw['mask'],_async_text_prefix=prepared)


def prepare_without_cache_purge(tape):
    if tape.preparation is not None:
        return
    started = time.perf_counter()
    tape.torch.cuda.synchronize()
    tape.preparation_calls += 1
    tape.preparation = dict(host_start=started, host_end=time.perf_counter(),
        host_ms=(time.perf_counter()-started)*1000,
        charged_in_first_electronics_segment=True, calls=tape.preparation_calls,
        extra_forward_calls=0, forced_gc_collect=False, allocator_cache_purge=False)
    tape.capture_audit['preparation'] = tape.preparation


def detailed_boundaries(model, tape, original_install):
    restore, ledger = original_install(model, tape)
    saved = []
    try:
        for name in ('frame_merger', 'readout'):
            owner = getattr(model, name)
            original = owner.forward
            def wrapped(*args, _original=original, _name=name, **kwargs):
                with tape.excluded(_name, 'electronics'):
                    return _original(*args, **kwargs)
            owner.forward = wrapped
            saved.append((owner, original))
            ledger.append(dict(stage=name, original_math_unchanged=True,
                               detailed_electronics_boundary=True))
    except BaseException:
        for owner, original in reversed(saved):
            owner.forward = original
        restore()
        raise
    def close():
        for owner, original in reversed(saved):
            owner.forward = original
        restore()
    return close, ledger


@contextmanager
def runtime_gc_policy(enabled):
    import gc
    was_enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        elif gc.isenabled():
            gc.disable()
