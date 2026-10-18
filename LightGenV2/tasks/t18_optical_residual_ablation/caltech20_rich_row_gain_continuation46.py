"""Foreground gain-conditioned continuation of verified343/338 same-head states."""
import types,argparse
import caltech20_rich_local_row_coverage46 as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_row_gain_continuation46.py','caltech20_rich_row_gain_continuation46_profile.json','audit_caltech20_rich_row_gain_continuation46.py','CALTECH20_RICH_ROW_GAIN_CONTINUATION46_PROTOCOL_20261018.md')
PROFILE=s.read(HERE/FILES[1]);ROUND=int(os.environ.get('T18_ROW_GAIN_ROUND','1'));assert 1<=ROUND<=PROFILE['maximum_rounds']
ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'round{ROUND:02d}';CPU=HERE/'runs/smoke/caltech20_rich_row_gain_continuation46_CPU_20261018'
def config():
 c=copy.deepcopy(PROFILE);c.update(run=PROFILE['run']+f'_round{ROUND:02d}',round=ROUND,seed=PROFILE['seed']+ROUND*10000,frozen_depths=[])
 if ROUND>1:
  f=ROOT_OUT/f'round{ROUND-1:02d}';c['parents']={str(d):s.sha(f/f'L{d}/best.pt') for d in (4,6)}
  for d in (4,6):
   q=s.read(f/f'L{d}/fit_result.json')
   if not q['strictly_improved'] or q['target_achieved']:c['frozen_depths'].append(d)
 return c
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def incumbent(d):
 root=previous.OUT if ROUND==1 else ROOT_OUT/f'round{ROUND-1:02d}';rec=s.read(root/'audit_completed.json');assert rec['passed'] and rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(root/'science/audit.json')
 f=root/f'L{d}';p=f/'best.pt';q=dict(path=str(p),sha256=s.sha(p),source_science=rec['audit_sha256'],parent_round=ROUND-1);assert q['sha256']==config()['parents'][str(d)]
 pt=torch.load(p,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(f/('selected_'+r+'.npz')) for r in ('train','val','test')};return q,pt,pred
for name in ('identity','claim','prepared','gradient','update','ready','fit'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=previous.stage_plan;_stage_plan=types.FunctionType(f.__code__,globals(),'same_TRAIN_sampler',f.__defaults__,f.__closure__)
def stage_plan(d,stage,pred,w,b,used):
 if d in config()['frozen_depths']:return None
 return _stage_plan(d,stage,pred,w,b,used)
def campaign():
 assert ROUND==1 and os.environ['CUDA_VISIBLE_DEVICES']=='' and not ROOT_OUT.exists();ready();claim(ROOT_OUT.with_name(ROOT_OUT.name+'.claim.json'),'foreground_gain_conditioned_two_depth_head_continuation');ROOT_OUT.mkdir();s.save(ROOT_OUT/'metadata.json',dict(identity(),profile=PROFILE,pid=os.getpid(),no_automation=True),exclusive=True);history=[];reason='bounded_gain_rounds_complete'
 for r in range(1,PROFILE['maximum_rounds']+1):
  f=ROOT_OUT/f'round{r:02d}';assert not f.exists();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',T18_ROW_GAIN_ROUND=str(r))
  with (ROOT_OUT/f'round{r:02d}.fit.log').open('x') as log:p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
  assert f.exists();s.save(f/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'preserve failure, no retry'
  with (f/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[2]),'--phase','audit'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  s.save(f/'audit_exit.json',dict(actual_audit_returncode=arc),exclusive=True);assert arc==0,'preserve audit failure, no retry';assert s.read(f/'science/audit.json')['passed'];s.save(f/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(f/'science/audit.json')),exclusive=True)
  arms={str(d):s.read(f/f'L{d}/fit_result.json') for d in (4,6)};history.append(dict(round=r,science=s.sha(f/'science/audit.json'),arms={d:dict(correct=q['selected_correct'],improved=q['strictly_improved'],target=q['target_achieved'],steps=q['actual_updates'],PT=str(f/f'L{d}/best.pt'),PT_SHA256=q['hashes']['best.pt'],metrics=q['metrics']) for d,q in arms.items()}));s.save(ROOT_OUT/'status.json',dict(state='gain_conditioned_running',round=r,history=history))
  if all(q['target_achieved'] for q in arms.values()):reason='both_targets_achieved';break
  if not any(q['strictly_improved'] and not q['target_achieved'] for q in arms.values()):reason='no_unfinished_depth_with_strict_gain_review';break
 s.save(ROOT_OUT/'completed.json',dict(passed=True,state=reason,history=history,TEST_informed_development=True,independent_generalization=False),exclusive=True);identity();parent.old.controls()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:campaign() if a.phase=='campaign' else fit()
 except BaseException as e:
  f=ROOT_OUT if a.phase=='campaign' else OUT;p=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
