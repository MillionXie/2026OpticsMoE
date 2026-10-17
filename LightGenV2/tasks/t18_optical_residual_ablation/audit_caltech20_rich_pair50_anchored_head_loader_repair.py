"""Unchanged independent factor10 NumPy verifier in own explicit namespace."""
import argparse
import audit_caltech20_rich_pair50_anchored_head as frozen
import caltech20_rich_pair50_anchored_head_loader_repair as m
ns=dict(vars(frozen));ns.update(m=m,__file__=__file__);exec(compile(frozen.text,__file__+':independent','exec'),ns);exec(compile(frozen.previous.t,__file__+':audit','exec'),ns)
independent=ns['independent'];audit=ns['audit']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
 try:audit(a.case,a.phase)
 except BaseException as e:
  mark=m.folder(a.case)/(a.phase+'_audit_failure.json')
  if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
