"""Continue verified335/352 heads toward337/354 without editing any retained version."""
import types,copy
import caltech20_rich_head_targets26 as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_head_targets_next26.py','caltech20_rich_head_targets_next26_profile.json','audit_caltech20_rich_head_targets_next26.py','CALTECH20_RICH_HEAD_TARGETS_NEXT26_PROTOCOL_20261018.md')
DEPTH=int(os.environ.get('T18_HEAD_DEPTH','2'));assert DEPTH in (2,6)
WHOLE_PROFILE=s.read(HERE/FILES[1]);PROFILE={k:copy.deepcopy(v) for k,v in WHOLE_PROFILE.items() if k!='cases'};PROFILE.update(WHOLE_PROFILE['cases'][str(DEPTH)])
CASE=int(os.environ.get('T18_L6_COMPETITION_CASE','0'));ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'case{CASE:02d}';CPU=HERE/'runs/smoke'/f'caltech20_rich_head_targets_next26_L{DEPTH}_CPU_20261018'
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def initial_parent():
 root=HERE/'runs/simulation'/previous.WHOLE_PROFILE['cases'][str(DEPTH)]['run'];done=s.read(root/'completed.json');ex=s.read(root/'campaign_exit.json');assert done['passed'] and done['state']=='target_achieved' and ex['actual_exit'] and ex['returncode']==0;q=dict(done['best']);assert q['sha256']==PROFILE['initial_parent_SHA256'];return q
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
