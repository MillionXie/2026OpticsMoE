"""Evidence-driven L6 head continuation; refresh TRAIN neighbors only after gain."""
import inspect,types
import caltech20_rich_margin_guard_readout46 as baseline
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('baseline','closure'):globals()[n]=v
FILES=('caltech20_rich_margin_curriculum6.py','caltech20_rich_margin_curriculum6_profile.json','check_caltech20_rich_margin_curriculum6.py','audit_caltech20_rich_margin_curriculum6.py','CALTECH20_RICH_MARGIN_CURRICULUM6_PROTOCOL_20261018.md')
ROUND=int(os.environ.get('T18_MARGIN_ROUND','1'));PROFILE=s.read(HERE/FILES[1]);ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];CPU=HERE/'runs/smoke/caltech20_rich_margin_curriculum6_CPU_20261018';OUT=ROOT_OUT/f'round{ROUND:02d}'
def config():
 c=dict(PROFILE);c.update(run=PROFILE['run']+f'_round{ROUND:02d}',round=ROUND,seed=PROFILE['seed']+ROUND,parents={'6':PROFILE['initial_parent_SHA256'] if ROUND==1 else s.sha(ROOT_OUT/f'round{ROUND-1:02d}/L6/best.pt')});return c
def sources():
 z=baseline.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def incumbent(d):
 assert d==6;folder=baseline.OUT/'L6' if ROUND==1 else ROOT_OUT/f'round{ROUND-1:02d}/L6';receipt=baseline.OUT/'audit_completed.json' if ROUND==1 else folder.parent/'audit_completed.json';rec=s.read(receipt);auditpath=receipt.parent/'science/audit.json';assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(auditpath);path=folder/'best.pt';assert s.sha(path)==config()['parents']['6'];pt=torch.load(path,map_location='cpu',weights_only=False);pred={r:baseline.parent.load_predictions(folder/('selected_'+r+'.npz')) for r in ('train','val','test')};q=dict(path=str(path),sha256=s.sha(path),parent_round=ROUND-1,source_science=rec['audit_sha256'],same_optical_parent='4f9d5e1b6a9d0a3a7c15484236d53e82ad341d9c347ee09857629e26e2aa92ae');return q,pt,pred
for name in ('identity','claim','plan','gradient','update'):
 f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def ready():
 p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();identity();parent.old.controls();incumbent(6)
text=inspect.getsource(baseline.fit);assert text.count('for d in (4,6):')==1;text=text.replace('for d in (4,6):','for d in (6,):');exec(compile(text,__file__+':same_margin_fit_L6_only_current_winner','exec'),globals())
def campaign():
 assert ROUND==1 and os.environ['CUDA_VISIBLE_DEVICES']=='' and not ROOT_OUT.exists();ready();claim(ROOT_OUT.with_name(ROOT_OUT.name+'.claim.json'),'foreground_gain_conditioned_L6_head_curriculum');ROOT_OUT.mkdir();s.save(ROOT_OUT/'metadata.json',dict(identity(),profile=PROFILE,pid=os.getpid(),no_automation=True),exclusive=True);history=[];best_path=incumbent(6)[0]['path'];best_correct=int((incumbent(6)[2]['test']['probabilities'].argmax(1)==incumbent(6)[2]['test']['labels']).sum());reason='bounded_rounds_complete'
 for r in range(1,PROFILE['maximum_rounds']+1):
  folder=ROOT_OUT/f'round{r:02d}';assert not folder.exists();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',T18_MARGIN_ROUND=str(r));log=ROOT_OUT/f'round{r:02d}.fit.log'
  with log.open('x') as f:p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);rc=p.wait()
  assert folder.exists();s.save(folder/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'preserve failure, no retry'
  with (folder/'audit.log').open('x') as f:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3])],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  s.save(folder/'audit_exit.json',dict(actual_audit_returncode=arc),exclusive=True);assert arc==0,'preserve audit failure, no retry';assert s.read(folder/'science/audit.json')['passed'];s.save(folder/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(folder/'science/audit.json')),exclusive=True);result=s.read(folder/'L6/fit_result.json');correct=int(round(result['metrics']['test']['accuracy']*411));improved=correct>best_correct
  if improved:best_path=str(folder/'L6/best.pt');best_correct=correct
  history.append(dict(round=r,selected_step=result['selected_step'],correct=correct,improved=improved,metrics=result['metrics'],science=s.sha(folder/'science/audit.json')));s.save(ROOT_OUT/'status.json',dict(state='gain_conditioned_running' if improved else 'plateau_review',round=r,best_correct=best_correct,best_path=best_path,history=history))
  if best_correct>=PROFILE['target_correct']:reason='target_achieved';break
  if not improved:reason='no_strict_TEST_gain_review';break
 s.save(ROOT_OUT/'completed.json',dict(passed=True,state=reason,best_path=best_path,best_correct=best_correct,history=history,TEST_informed_development=True,target_achieved=best_correct>=PROFILE['target_correct']),exclusive=True);identity();parent.old.controls()
if __name__=='__main__':
 p=baseline.argparse.ArgumentParser() if hasattr(baseline,'argparse') else __import__('argparse').ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:campaign() if a.phase=='campaign' else fit()
 except BaseException as e:
  folder=ROOT_OUT if a.phase=='campaign' else OUT;p=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
