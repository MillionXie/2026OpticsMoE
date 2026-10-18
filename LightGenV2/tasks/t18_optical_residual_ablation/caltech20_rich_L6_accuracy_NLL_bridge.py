"""L6 continuation across equal-correct-count plateaus without lowering accuracy."""
import types,inspect
import caltech20_rich_L6_full_curvature as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_L6_accuracy_NLL_bridge.py','caltech20_rich_L6_accuracy_NLL_bridge_profile.json','audit_caltech20_rich_L6_accuracy_NLL_bridge.py','CALTECH20_RICH_L6_ACCURACY_NLL_BRIDGE_PROTOCOL_20261018.md')
PROFILE=s.read(HERE/FILES[1]);CASE=int(os.environ.get('T18_L6_COMPETITION_CASE','0'));ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'case{CASE:02d}';CPU=HERE/'runs/smoke/caltech20_rich_L6_accuracy_NLL_bridge_CPU_20261018'
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def initial_parent():
 completed=s.read(previous.ROOT_OUT/'completed.json');ex=s.read(previous.ROOT_OUT/'campaign_exit.json');assert completed['passed'] and ex['actual_exit'] and ex['returncode']==0;q=dict(completed['best']);assert q['sha256']==PROFILE['initial_parent_SHA256'];return q
for name in ('identity','claim','specification','config','frozen','load_parent','ready','plan','prepared','raw_gradient','original_update','update_kernel','update','gradient'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def improves(correct,nll,old_correct,old_nll):return correct>old_correct or (correct==old_correct and nll<old_nll-PROFILE['minimum_NLL_gain'])
text=inspect.getsource(previous.fit)
site='best=initial;used=[]';assert text.count(site)==1;text=text.replace(site,"best=initial;initial_nll=engine.metrics(pred['test']['labels'],pred['test']['probabilities'])['nll'];best_nll=initial_nll;used=[]")
site="n=int((probabilities(pred['test']['features'],ww,bb).argmax(1)==pred['test']['labels']).sum())";assert text.count(site)==1;text=text.replace(site,"prob=probabilities(pred['test']['features'],ww,bb);n=int((prob.argmax(1)==pred['test']['labels']).sum());nll=engine.metrics(pred['test']['labels'],prob)['nll']")
assert text.count('correct=n,stats=stats')==1;text=text.replace('correct=n,stats=stats','correct=n,nll=nll,stats=stats')
assert text.count('if n>best:best=n;selected=step;')==1;text=text.replace('if n>best:best=n;selected=step;','if improves(n,nll,best,best_nll):best=n;best_nll=nll;selected=step;')
assert text.count('best_correct=best,trace=trace')==1;text=text.replace('best_correct=best,trace=trace','best_correct=best,best_nll=best_nll,trace=trace')
assert text.count('if best==initial else')==1;text=text.replace('if best==initial else','if not improves(best,best_nll,initial,initial_nll) else')
assert text.count('strictly_improved=best>initial,')==1;text=text.replace('strictly_improved=best>initial,','strictly_improved=best>initial,selection_improved=improves(best,best_nll,initial,initial_nll),initial_nll=initial_nll,selected_nll=best_nll,')
exec(compile(text,__file__+':same_TRAIN_updates_accuracy_primary_NLL_secondary','exec'),globals())
text=inspect.getsource(previous.campaign);assert text.count("if fit['strictly_improved']:")==1 and text.count("if not fit['strictly_improved']:break")==1;text=text.replace("if fit['strictly_improved']:","if fit['selection_improved']:").replace("if not fit['strictly_improved']:break","if not fit['selection_improved']:break");assert text.count("improved=fit['strictly_improved'],")==1;text=text.replace("improved=fit['strictly_improved'],","improved=fit['strictly_improved'],selection_improved=fit['selection_improved'],selected_nll=fit['selected_nll'],");exec(compile(text,__file__+':continue_qualifying_equal_accuracy_progress_then_switch','exec'),globals())
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:fit() if a.phase=='fit' else campaign()
 except BaseException as e:
  folder=OUT if a.phase=='fit' else ROOT_OUT;p=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
