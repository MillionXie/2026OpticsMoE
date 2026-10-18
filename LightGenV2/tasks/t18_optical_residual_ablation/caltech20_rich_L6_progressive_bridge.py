"""Continue the genuinely improving L6 Newton/NLL bridge from verified348 state."""
import types
import caltech20_rich_L6_accuracy_NLL_bridge as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_L6_progressive_bridge.py','caltech20_rich_L6_progressive_bridge_profile.json','audit_caltech20_rich_L6_progressive_bridge.py','CALTECH20_RICH_L6_PROGRESSIVE_BRIDGE_PROTOCOL_20261018.md')
PROFILE=s.read(HERE/FILES[1]);CASE=int(os.environ.get('T18_L6_COMPETITION_CASE','0'));ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'case{CASE:02d}';CPU=HERE/'runs/smoke/caltech20_rich_L6_progressive_bridge_CPU_20261018'
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def initial_parent():
 completed=s.read(previous.ROOT_OUT/'completed.json');ex=s.read(previous.ROOT_OUT/'campaign_exit.json');assert completed['passed'] and ex['actual_exit'] and ex['returncode']==0;q=dict(completed['best']);assert q['sha256']==PROFILE['initial_parent_SHA256'];return q
for name in ('identity','claim','specification','config','frozen','load_parent','ready','plan','prepared','raw_gradient','original_update','update_kernel','update','gradient','improves','fit','campaign'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:fit() if a.phase=='fit' else campaign()
 except BaseException as e:
  folder=OUT if a.phase=='fit' else ROOT_OUT;p=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
