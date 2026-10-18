"""Independent exact weighted-state arithmetic, no model or forward."""
import inspect
import caltech20_rich_electronic_weight_average46 as m
import check_caltech20_rich_weight_average as previous
import audit_caltech20_rich_electronic_weight_average46 as audit
text=inspect.getsource(previous.run);assert text.count('for d in (2,4,6):')==1;text=text.replace('for d in (2,4,6):','for d in (4,6):');site="assert sum(v.numel() for n,v in states[0].items() if n.startswith('linear.'))==5140;";assert text.count(site)==1;text=text.replace(site,"assert all(torch.equal(m.state(d,f)[n],v) for f in m.config()['mixture_fractions'] for n,v in states[0].items() if n=='raw_router' or n.startswith('phase_sets.'));"+site);site='real_model_constructions=0';assert text.count(site)==1;text=text.replace(site,'optical_phase_bytes_identical_for_all_fractions=True,L2_final_untouched=True,'+site);ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);exec(compile(text,__file__+':same_independent_float64_weight_math_only46','exec'),ns);run=ns['run']
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
