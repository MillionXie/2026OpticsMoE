"""Full explicit saved-audit namespace; no omitted imported math helper."""
import types
import caltech20_rich_alpha40_fresh246 as m
import audit_caltech20_rich_alpha40_rebalance as original
namespace=dict(vars(original));namespace['m']=m
run=types.FunctionType(original.run.__code__,namespace,'run')
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
