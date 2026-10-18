"""Repair the exact failed checksum call, preserving independent numerical checks."""
import inspect,hashlib,json
import caltech20_rich_error_target_jsonhash_repair46 as m
import check_caltech20_rich_error_target_steps46 as previous
import numpy as np
import torch
independent_plan=previous.independent_plan
text=inspect.getsource(previous.run);assert text.count('m.s.digest(pl)')==1;text=text.replace('m.s.digest(pl)','m.plan_digest(pl)');exec(compile(text,__file__+':explicit_JSON_not_tensor_checksum_repair','exec'),globals());checked_run=run

def run():
 failed=m.previous.CPU.with_name(m.previous.CPU.name+'.failure.json');before=m.s.sha(failed);assert m.s.read(failed)['error']=="AttributeError(\"'list' object has no attribute 'detach'\")" and not m.previous.OUT.exists() and not m.previous.CPU.exists()
 toy=dict(b=[3,2,1],a=['TRAIN',17]);raw=json.dumps(toy,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode();assert m.plan_digest(toy)==hashlib.sha256(raw).hexdigest();assert m.plan_digest(toy)==m.plan_digest(dict(a=['TRAIN',17],b=[3,2,1]));assert m.plan_digest(toy)!=m.plan_digest(dict(a=['TRAIN',17],b=[3,2,4]))
 checked_run();assert m.s.sha(failed)==before;proof=m.s.read(m.CPU/'checks.json');proof.update(original_CPU_failure_SHA256=before,exact_repair='JSON plan hash replaces tensor-only digest; sampler and gradients unchanged',independent_JSON_checksum=True,original_failed_source_unchanged=True);m.s.save(m.CPU/'checks.json',proof)

if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
