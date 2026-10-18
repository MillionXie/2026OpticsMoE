"""Independent format failure preflight plus unchanged dual-view math proof."""
import types
import caltech20_rich_dualview_hosthash_repair as m
import check_caltech20_rich_targeted_dualview as previous
import audit_caltech20_rich_dualview_hosthash_repair as audit
def run():
    failures={}
    for key in m.config()['cases']:
        f=m.original.OUT/key;p=f/'failure.json';q=m.s.read(p);ex=m.s.read(m.original.OUT/(key+'.producer_exit.json'))
        assert q['error'].startswith('TypeError(') and 'tensor to numpy' in q['error'] and ex['returncode']==1 and ex['actual_exit'] and not m.Path('/proc',str(ex['pid']),'cmdline').exists()
        assert not any((f/n).exists() for n in ('numeric_gate.json','history.json','best.pt','last.pt','fit_result.json','metadata.json','optimizer_boundary.pt'));failures[key]=m.s.sha(p)
    namespace=dict(vars(previous));namespace.update(m=m,audit=audit);proof=types.FunctionType(previous.run.__code__,namespace,'unchanged_CPU_math',previous.run.__defaults__,previous.run.__closure__);proof()
    q=m.s.read(m.CPU/'checks.json')
    for key,case in m.config()['cases'].items():assert q['parents'][key]['gradient_policy']==case['full_rms_gradient']
    for key,sha in failures.items():assert m.s.sha(m.original.OUT/key/'failure.json')==sha
    q.update(original_gate_failures_preserved=failures,old_optimizer_and_TEST_calls=0,host_hash_path_explicit=True,derivative_policies_independently_bound=True);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
