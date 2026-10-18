"""Identical independent numerical audit, bound to repaired run identity."""
import types
import audit_caltech20_rich_error_target_steps46 as previous
import caltech20_rich_error_target_jsonhash_repair46 as m
ns=dict(vars(previous));ns['m']=m
run=types.FunctionType(previous.run.__code__,ns,'same_independent_short_step_science',previous.run.__defaults__,previous.run.__closure__)
if __name__=='__main__':
 a=previous.argparse.ArgumentParser();a.add_argument('--depth',type=int,required=True);z=a.parse_args()
 try:run(z.depth)
 except BaseException as e:
  p=m.OUT/f'L{z.depth}/audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
