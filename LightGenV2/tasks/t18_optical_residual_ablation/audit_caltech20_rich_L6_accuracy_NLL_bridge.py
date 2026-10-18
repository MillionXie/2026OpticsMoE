"""Independent primary accuracy/secondary NLL selection and unchanged Newton math."""
import inspect,types
import audit_caltech20_rich_L6_full_curvature as previous
import audit_caltech20_rich_L6_binary_rest as kernel
import caltech20_rich_L6_accuracy_NLL_bridge as m
for n,v in vars(previous).items():
 if not n.startswith('__') and n not in ('m','previous','kernel'):globals()[n]=v
for name in ('independent_plan','raw_independent','independent_gradient','independent_update'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=previous.cpu;cpu_kernel=types.FunctionType(f.__code__,globals(),'unchanged_independent_full_Hessian_CPU',f.__defaults__,f.__closure__)
def better(n,loss,best,old):return n>best or (n==best and loss<old-m.PROFILE['minimum_NLL_gain'])
def cpu():
 tests=[(346,.79,346,.8,True),(345,.1,346,.8,False),(347,2.,346,.8,True),(346,.79999,346,.8,False),(346,.8,346,.8,False),(346,.7999,346,.8,False)]
 for n,loss,best,old,want in tests:assert m.improves(n,loss,best,old)==want and better(n,loss,best,old)==want
 cpu_kernel();q=m.s.read(m.CPU/'checks.json');q.update(selector_accuracy_never_decreases=True,independent_selector_cases=len(tests),equal_accuracy_minimum_NLL_gain=m.PROFILE['minimum_NLL_gain']);m.s.save(m.CPU/'checks.json',q)
text=inspect.getsource(kernel.audit)
start=text.index('scale=min(1.,1/(');end=text.index('ww,bb=w.copy(),b.copy()',start);text=text[:start]+"u,v,mu,mv=independent_update(u,v,mu,mv,g,h,u0,v0,spec['family']);"+text[end:]
site='best=initial;used=[]';assert text.count(site)==1;text=text.replace(site,"best=initial;initial_nll=float(-np.log(np.maximum(pred['test']['probabilities'][np.arange(411),pred['test']['labels']],1e-12)).mean());best_nll=initial_nll;used=[]")
site="n=int((m.probabilities(pred['test']['features'],ww,bb).argmax(1)==pred['test']['labels']).sum())";assert text.count(site)==1;text=text.replace(site,"prob=m.probabilities(pred['test']['features'],ww,bb);n=int((prob.argmax(1)==pred['test']['labels']).sum());nll=float(-np.log(np.maximum(prob[np.arange(411),pred['test']['labels']],1e-12)).mean());assert abs(nll-row['nll'])<1e-12")
assert text.count('if n>best:best=n;selected=step;')==1;text=text.replace('if n>best:best=n;selected=step;','if better(n,nll,best,best_nll):best=n;best_nll=nll;selected=step;')
assert text.count("record['best_correct']==best;")==1;text=text.replace("record['best_correct']==best;","record['best_correct']==best;assert abs(record['best_nll']-best_nll)<1e-12;")
site="assert fit['strictly_improved']==(best>initial)";assert text.count(site)==1;text=text.replace(site,"assert fit['selection_improved']==better(best,best_nll,initial,initial_nll);assert abs(fit['initial_nll']-initial_nll)<1e-12 and abs(fit['selected_nll']-best_nll)<1e-12;"+site)
exec(compile(text,__file__+':same_independent_curvature_full_trajectory_and_secondary_NLL','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
