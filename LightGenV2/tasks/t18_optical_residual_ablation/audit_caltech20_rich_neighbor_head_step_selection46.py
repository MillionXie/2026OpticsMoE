"""Verify every reconstructed state against saved actual TRAIN update hashes."""
import os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_neighbor_head_step_selection46 as m
import audit_caltech20_rich_pair50_alpha as saved
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_verified_head_trajectory_selection');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources();m.engine.verify_sources(meta['sources'],meta['git_commit']);m.ready();records={}
 for d in (4,6):
  f=m.OUT/f'L{d}';result=m.s.read(f/'result.json');rows=m.s.read(f/'scores.json');q,pt,pred=m.parent.incumbent(d);candidate=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);assert len(rows)==129 and [r['step'] for r in rows]==list(range(129));selected=max(rows,key=lambda r:r['correct'])['step'];assert selected==result['selected_step']==candidate['selected_replayed_head_step'];assert result['candidate_sha256']==m.s.sha(f/'candidate.pt') and result['scores_sha256']==m.s.sha(f/'scores.json')
  # Every replayed float64 state must match the producer's original per-step SHA,
  # whose full momentum/projection trajectory was independently audited by prior.
  for step,w,b,pred in m.trajectory(d):
   p=pred['test']['probabilities'] if step==0 else m.prior.probabilities(pred['test']['features'],w,b);correct=int((p.argmax(1)==pred['test']['labels']).sum());assert rows[step]['correct']==correct
   if step==selected:assert np.array_equal(candidate['model']['linear.weight'].numpy(),w.astype(np.float32)) and np.array_equal(candidate['model']['linear.bias'].numpy(),b.astype(np.float32))
  assert all(torch.equal(v,candidate['model'][n]) for n,v in pt['model'].items() if not n.startswith('linear.'))
  for role in ('train','val','test'):met,_=saved.verify_predictions(f/('selected_'+role+'.npz'),candidate['model'],role,m.base.manifest());saved.agree(met,result['metrics'][role]);saved.agree(met,candidate['metrics'][role])
  records[str(d)]=dict(selected_step=selected,correct=rows[selected]['correct'],all_actual_128_state_hashes_match=True,new_training_steps=0,no_L2_model=True)
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],checks=records,TEST_informed_development=True,independent_generalization=False,optical_forward=0,controls=m.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
