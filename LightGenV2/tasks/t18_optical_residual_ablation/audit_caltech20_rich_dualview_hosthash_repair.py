"""Same independent saved science in complete, correctly rebound namespace."""
import types
import caltech20_rich_dualview_hosthash_repair as m
import audit_caltech20_rich_targeted_dualview as previous
ns=dict(vars(previous));ns['m']=m;ks=dict(previous.kernel.__globals__);ks['m']=m;ns['kernel']=types.FunctionType(previous.kernel.__code__,ks,'saved_kernel',previous.kernel.__defaults__,previous.kernel.__closure__)
kernel=ns['kernel']
run=types.FunctionType(previous.run.__code__,ns,'run',previous.run.__defaults__,previous.run.__closure__)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
