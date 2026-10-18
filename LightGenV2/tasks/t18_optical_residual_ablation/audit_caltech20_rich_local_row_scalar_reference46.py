"""Fix independent scalar dtype, keeping the original1e-10 criterion."""
import inspect,types
import audit_caltech20_rich_local_row_repair46 as previous
import caltech20_rich_local_row_scalar_reference46 as m
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='m':globals()[n]=v
f=previous.independent_plan;independent_plan=types.FunctionType(f.__code__,globals(),'same_independent_class_local_sampler',f.__defaults__,f.__closure__)
text=inspect.getsource(previous.cpu);assert text.count('vv=torch.tensor(v,requires_grad=True)')==1;text=text.replace('def cpu():','def cpu_kernel():',1).replace('vv=torch.tensor(v,requires_grad=True)','vv=torch.tensor(v,dtype=torch.float64,requires_grad=True)');exec(compile(text,__file__+':explicit_F64_reference_bias_matches_F64_weights','exec'),globals())
f=previous.audit;audit=types.FunctionType(f.__code__,globals(),'unchanged_independent_one_row_science',f.__defaults__,f.__closure__)
def cpu():
 failure=m.previous.CPU.with_name(m.previous.CPU.name+'.failure.json');before=m.s.sha(failure);assert before=='6b7bccd08d2350e07b2116ce6b3b21a8707a9d105475f4342095c39c99cbd0bb' and not m.previous.OUT.exists();cpu_kernel();assert m.s.sha(failure)==before;proof=m.s.read(m.CPU/'checks.json');proof.update(original_CPU_failure_path=str(failure),original_CPU_failure_SHA256=before,exact_engineering_repair='Independent reference scalar bias explicitly float64, same as weight reference; producer gradient unchanged',threshold_unchanged=1e-10);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
