"""Full saved CPU review of disclosed microbatch continuation and sealed prefix."""
import argparse,json,os
from pathlib import Path
import torch
import resisc20 as s
import resisc20_microbatch as speed
import audit_resisc20 as audit
from audit_caltech_repair234 import exited

def run(a):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and not (a.base/'cpu_report_failure.json').exists()
 meta=s.read(a.base/'metadata.json');state=s.read(a.base/'status.json');assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures'] and len(state['completed'])==2
 assert meta['resume_parent']==str(speed.PARENT) and meta['stop_receipt_sha256']==s.sha(speed.STOP)
 stop=speed.stopped();micro=speed.prerequisites();parents={j['case']:j for j in stop['children']};checks=[]
 for j in state['completed']:
  assert exited(j['pid']) and j['exit_code']==0;key=j['case'];parent=speed.PARENT/key;arm=a.base/key;prior=parents[key]
  assert s.sha(parent/'last_checkpoint.pt')==prior['last_checkpoint_sha256'] and s.sha(parent/'best_checkpoint.pt')==prior['best_checkpoint_sha256']
  old=torch.load(parent/'last_checkpoint.pt',map_location='cpu',weights_only=False);f=s.read(arm/'fit_result.json');r=s.read(arm/'resume_receipt.json');history=s.read(arm/'history.json');prefix=s.read(parent/'history.json');grads=s.read(arm/'gradients.json');oldgrad=s.read(parent/'gradients.json');epoch=old['epoch']
  assert old['sources']==s.read(speed.PARENT/'metadata.json')['sources'] and epoch==prior['epoch']==len(prefix)==r['parent_epoch']==f['execution_migration']['prefix_epochs']
  assert r['first_new_epoch']==epoch+1 and not r['optimizer_reset'] and not r['ema_reset'] and not r['scheduler_reset'] and r['model_source_unchanged']
  assert r['old_microbatch']==1 and r['new_microbatch']==f['config']['microbatch']==micro
  assert r['last_checkpoint_sha256']==prior['last_checkpoint_sha256'] and r['best_checkpoint_sha256']==prior['best_checkpoint_sha256'] and r['prefix_history_sha256']==s.sha(parent/'history.json')
  assert r['restored_state_sha256']=={k:s.digest(old[k]) for k in ('model','ema')}
  assert history[:epoch]==prefix and grads[:epoch]==oldgrad and f['orders'][:epoch]==old['orders'] and f['transforms'][:epoch]==old['transforms']
  assert r['conservative_prior_seconds']>=prefix[-1]['seconds'] and r['conservative_prior_seconds']==stop['conservative_prior_fit_seconds']
  current=dict(f['config']);previous=dict(old['config'])
  for cfg in (current,previous):cfg.pop('microbatch')
  assert current==previous
  assert s.read(arm/'smoke.json')['passed'] and f['config']['maximum_epochs']==30 and f['config']['maximum_seconds']==14400 and f['config']['batch_size']==16
  checks.append(dict(case=key,prefix_epoch=epoch,first_continued_epoch=epoch+1,prefix_checkpoint_sha256=prior['last_checkpoint_sha256'],raw_ema_restored_exact=True,optimizer_ema_scheduler_reset=False,all_prefix_orders_transforms_gradients_history_preserved=True,effective_batch16_full_train11198_steps700_tail14_preserved=True,old_microbatch=1,new_microbatch=micro,conservative_prior_seconds=r['conservative_prior_seconds']))
 # This independently checks every actual epoch/order/affine/MixUp/Adam700E/
 # scheduler/phase gradient, seed/config/PT/data and saved ID/CM/NLL, no forward.
 audit.audit(a)
 s.save(a.out/'execution_migration_cpu_checks.json',dict(passed=True,checks=checks,old_training_source_commit='513f0e884a16be5f37846974deaeea55fac95f94',continuation_source_commit=meta['git_commit'],audit_sha256=s.sha(a.out/'audit.json'),model_forward_calls=0,test_arrays_read=0,test_evaluations=0,formal_authorized_by_this_receipt=False),exclusive=True)
 return s.read(a.out/'audit.json')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('base','out','data'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args()
 try:print(json.dumps(run(a)))
 except BaseException as e:
  marker=a.base/'cpu_report_failure.json'
  if not marker.exists():s.save(marker,dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
  raise
