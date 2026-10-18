"""Explicit repair: JSON sampling plans must not use tensor-state digest."""
import types,json,hashlib
import caltech20_rich_error_target_steps46 as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_error_target_jsonhash_repair46.py','check_caltech20_rich_error_target_jsonhash_repair46.py','audit_caltech20_rich_error_target_jsonhash_repair46.py','CALTECH20_RICH_ERROR_TARGET_JSONHASH_REPAIR46_PROTOCOL_20261018.md')
OUT=previous.OUT.with_name(previous.OUT.name+'_jsonhash_repaired')
CPU=previous.CPU.with_name(previous.CPU.name+'_jsonhash_repaired')

def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z

def plan_digest(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
for name in ('identity','claim','plan','model_for','inputs','gradients','gate','ready','fit','campaign'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['fit','campaign'],required=True);a.add_argument('--depth',type=int);a.add_argument('--gpu');z=a.parse_args()
 try:campaign() if z.phase=='campaign' else fit(z.depth,z.gpu)
 except BaseException as e:
  folder=OUT/f'L{z.depth}' if z.depth else OUT;mark=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
