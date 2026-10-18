"""Original strict saved average/VALID selection and one-TEST audit for L2."""
import inspect,types
import caltech20_rich_L2_local_weight_average as m
import audit_caltech20_rich_weight_average as previous
ns=dict(previous.verify.__globals__);ns['m']=m;verify=types.FunctionType(previous.verify.__code__,ns,'independent_L2_average_and_predictions',previous.verify.__defaults__,previous.verify.__closure__)
text=inspect.getsource(previous.run);assert text.count('for d in (2,4,6)')==1;text=text.replace('for d in (2,4,6)','for d in (2,)');ns=dict(previous.run.__globals__);ns.update(m=m,verify=verify);exec(compile(text,__file__+':same_independent_science_L2','exec'),ns);run=ns['run']
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:run(a.phase)
    except BaseException as e:
        p=m.OUT/(a.phase+'_audit_failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
