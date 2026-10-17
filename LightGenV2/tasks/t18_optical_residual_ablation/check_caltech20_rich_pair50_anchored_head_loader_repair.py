"""Real cached TRAIN/VAL path preflight plus unchanged analytic gradient proof."""
import os,numpy as np,torch
import check_caltech20_rich_pair50_anchored_head as frozen
import caltech20_rich_pair50_anchored_head_loader_repair as m
from audit_caltech20_rich_pair50_anchored_head_loader_repair import independent

def preflight():
 assert 'load_np' not in vars(m.failed)
 for name in ('load_np','probability','logits','core_digest'):assert callable(getattr(m,name)) and getattr(m,name) is getattr(m.failed.previous,name)
 proof={}
 for key in m.config()['cases']:
  arm,_=m.admit(key);state=torch.load(arm/'best.pt',map_location='cpu',weights_only=False)['model'];a,q,valid=m.training_data(arm,state);assert a['train']['features'].shape==(3141,256) and a['val']['features'].shape==(383,256) and q.shape==(3141,20) and np.allclose(q.sum(1),1,atol=2e-7);proof[key]=dict(TRAIN_features_sha256=m.s.sha(arm/'selected_train.npz'),VAL_features_sha256=m.s.sha(arm/'selected_val.npz'),valid_teacher=int(valid.sum()),preserved_failure_sha256=m.s.sha(m.failed.folder(key)/'failure.json'))
 return proof
text=frozen.previous.text
assert '    r=np.random.default_rng(17);' in text;text=text.replace('    r=np.random.default_rng(17);','    cached=preflight()\n    r=np.random.default_rng(17);').replace('passed=True,gradient_torch_numpy','passed=True,cached_feature_preflight=cached,gradient_torch_numpy');exec(compile(text,__file__+':main','exec'),globals())
if __name__=='__main__':
 try:main()
 except BaseException as e:
  mark=m.CPU/'failure.json' if m.CPU.exists() else m.CPU.with_name(m.CPU.name+'.failure.json')
  if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
