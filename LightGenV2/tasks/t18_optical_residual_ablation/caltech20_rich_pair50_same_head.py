"""One full-batch fit of the SAME linear head on verified70 cached features."""
import argparse,ast,inspect,os,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import torch.nn.functional as F
import caltech20_same_linear_convex as math_source
import caltech20_rich_pair50_depthwise_continue as driver
import audit_caltech20_rich_pair50_alpha as metric_source
parent=driver.parent;HERE,ROOT,s,engine=parent.HERE,parent.ROOT,parent.s,parent.engine
FILES=('caltech20_rich_pair50_same_head.py', 'caltech20_rich_pair50_same_head_profile.json', 'check_caltech20_rich_pair50_same_head.py', 'audit_caltech20_rich_pair50_same_head.py', 'CALTECH20_RICH_PAIR50_SAME_HEAD_PROTOCOL_20261017.md')
CPU=HERE/'runs/smoke/caltech20_rich_pair50_same_head_cpu_20261017'
load_np=math_source.old.load_np
probability=math_source.old.probability
logits=math_source.old.logits
core_digest=math_source.old.core_digest
old=SimpleNamespace(load_np=load_np,probability=probability,logits=logits,core_digest=core_digest,old=SimpleNamespace(base=SimpleNamespace(metrics=metric_source.metrics,save_np=parent.base.save_np,manifest=parent.base.manifest),teacher=parent.teacher))

def config():
 c=s.read(HERE/FILES[1]);assert c['teacher_temperature']==4 and c['teacher_target_temperature']==16 and c['CE_weight']==.98 and c['teacher_weight']==.02 and c['new_inference_parameters']==0 and c['class_weight_exponent']==1;return c

def folder(key):assert key in config()['cases'];return HERE/'runs/simulation'/config()['run']/key

def sources():
 z={**driver.sources(),**math_source.sources()}
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z

def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def gone(pid):return not Path('/proc',str(pid),'cmdline').exists()

def admit(key):
 assert key in config()['cases'];parent.admit();meta=s.read(parent.OUT/'metadata.json');assert meta['sources']==parent.sources() and meta['config']==parent.config();engine.verify_sources(meta['sources'],meta['git_commit'])
 assert s.read(parent.OUT/'status.json')['state']=='scaledTeacherKD20_science_and_reviewed_tests_complete'
 for receipt,where in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
  z=s.read(parent.OUT/receipt);assert z['actual_audit_returncode']==0 and z['audit_sha256']==s.sha(parent.OUT/where/'audit.json') and s.read(parent.OUT/where/'audit.json')['passed']
 for pid in (meta['pid'],s.read(parent.OUT.with_name(parent.OUT.name+'.started.json'))['supervisor_pid']):assert gone(pid)
 ex=s.read(parent.OUT/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit']
 for n in (key+'.fit_exit.json','once_tests/'+key+'.test_exit.json'):
  z=s.read(parent.OUT/n);assert z['returncode']==0 and z['actual_exit'] and gone(z['pid'])
 arm=parent.OUT/key;fit=s.read(arm/'fit_result.json');assert fit['actual_joint_epochs']==70 and fit['steps']==13790 and fit['best_sha256']==s.sha(arm/'best.pt');return arm,fit

def cpu_ready():
 z=s.read(CPU/'checks.json');assert z['passed'] and z['sources']==sources();engine.verify_sources(z['sources'],z['git_commit'])

def training_data(arm,state):
 a={role:load_np(arm/('selected_'+role+'.npz')) for role in ('train','val')}
 for role,z in a.items():
  assert z['ids'].tolist()==parent.base.manifest()['split_ids'][role] and np.max(np.abs(probability(logits(state,z['features']))-z['probabilities']))<3e-6
 teacher=torch.load(parent.teacher.OUT/'selected_head.pt',map_location='cpu',weights_only=False);tf=load_np(parent.teacher.OUT/'train_features.npz');assert np.array_equal(tf['ids'],a['train']['ids']) and np.array_equal(tf['labels'],a['train']['labels'])
 tx=((tf['features']-teacher['mean'])/teacher['std']).astype(np.float32).astype(np.float64);tz=tx@teacher['model']['weight'].double().numpy().T+teacher['model']['bias'].double().numpy();valid=(tz.argmax(1)==tf['labels'])&(probability(tz).max(1)>=config()['teacher_correct_confidence_gate']);return a,probability(tz/config()['teacher_target_temperature']),valid

# Exact own-namespace transformations; frozen old modules/globals are not patched.
objective_text=inspect.getsource(math_source.objective);assert 'counts.rsqrt()' in objective_text;objective_text=objective_text.replace('counts.rsqrt()','counts.reciprocal()');exec(compile(objective_text,__file__+':objective','exec'),globals())
fit_text=inspect.getsource(math_source.fit)
changes={"parent/'fit/selected.pt'":"parent/'best.pt'",'parameter=torch.nn.Parameter(torch.tensor(anchor))':'parameter=torch.nn.Parameter(torch.tensor(anchor).contiguous())',"scale=old.temperature(vz,a['val']['labels'],cfg['calibration_inverse_temperature_bounds'],cfg['calibration_iterations'])":'scale=1.',"accepted=cv['accuracy']>pv['accuracy']+1e-12 or (abs(cv['accuracy']-pv['accuracy'])<1e-12 and cv['balanced_nll']<pv['balanced_nll']-1e-6)":"accepted=cv['balanced_nll']<pv['balanced_nll']",'optical_actual_epochs=80':'optical_actual_epochs=70','VAL_labels_used_for_scalar_calibration=True':'VAL_labels_used_for_scalar_calibration=False',"parent/'fit'/('selected_'+r+'.npz')":"parent/('selected_'+r+'.npz')"}
for old_text,new_text in changes.items():assert old_text in fit_text,old_text;fit_text=fit_text.replace(old_text,new_text)
exec(compile(fit_text,__file__+':fit','exec'),globals())

def campaign(key):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cpu_ready();arm,_=admit(key);root=folder(key);assert not root.exists();root.parent.mkdir(parents=True,exist_ok=True);claim(root.with_name(root.name+'.claim.json'),'same5140_head_one_fullbatch_TRAIN_fit');root.mkdir();s.save(root/'metadata.json',dict(identity(),pid=os.getpid(),case=key,config=config(),parent=str(arm)),exclusive=True)
 with (root/'fit.log').open('x') as log:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fit','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
 s.save(root/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0
 claim(root/'audit_claim.json','own_NumPy_objective_gradient_and_fixed_core_audit')
 with (root/'audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(HERE/FILES[3]),'--phase','fit','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT).returncode
 assert rc==0 and s.read(root/'science/audit.json')['passed'];s.save(root/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(root/'science/audit.json')),exclusive=True);s.save(root/'status.json',dict(state='same_head_science_complete_awaiting_review_noTEST'))

def test(key):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';arm,pf=admit(key);root=folder(key);proof=s.read(root/'audit_completed.json');assert proof['actual_audit_returncode']==0 and proof['audit_sha256']==s.sha(root/'science/audit.json') and s.read(root/'science/audit.json')['passed'];ex=s.read(root/'campaign_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and gone(ex['pid']);out=root/'once_test';assert not out.exists();claim(root/'test_claim.json','once_cached_features_same_linear_test');out.mkdir()
 f=s.read(root/'fit/fit_result.json');selected=torch.load(root/'fit/selected.pt',map_location='cpu',weights_only=False)['model'];ps=torch.load(arm/'best.pt',map_location='cpu',weights_only=False)['model'];assert core_digest(selected)==core_digest(ps);reused=s.digest(selected)==s.digest(ps);cached=parent.OUT/'once_tests'/key;z=load_np(cached/'predictions.npz');assert z['ids'].tolist()==parent.base.manifest()['split_ids']['test']
 if reused:shutil.copyfile(cached/'predictions.npz',out/'predictions.npz');met=s.read(cached/'result.json')['metrics']
 else:z['probabilities']=probability(logits(selected,z['features']));parent.base.save_np(out/'predictions.npz',z);met=metric_source.metrics(z['labels'],z['probabilities'])
 s.save(out/'result.json',dict(passed=True,metrics=met,test_reused=reused,production_head_TEST_evaluations=0 if reused else 1,new_optical_TEST_forwards=0,selected_sha256=f['selected_sha256'],model_digest=s.digest(selected),parent_TEST_prediction_sha256=s.sha(cached/'predictions.npz'),test_used_for_selection=False,VAL_used_for_scalar_calibration=False,test_informed_development=True),exclusive=True)


def group(key=None):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cpu_ready();[admit(k) for k in config()['cases']];root=folder(config()['cases'][0]).parent;assert not root.exists();claim(root.with_name(root.name+'.group_claim.json'),'three_frozen70_same_head_CPU_campaigns');root.mkdir();s.save(root/'metadata.json',dict(identity(),pid=os.getpid(),config=config()),exclusive=True);jobs=[];done=[];failures=[]
 for key in config()['cases']:
  log=(root/(key+'.campaign.log')).open('x');p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','campaign','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,pid=p.pid,process=p,log=log))
 import time
 while jobs:
  for job in list(jobs):
   rc=job['process'].poll()
   if rc is None:continue
   job['log'].close();rec=dict(key=job['key'],pid=job['pid'],returncode=rc,actual_exit=True);s.save(folder(job['key'])/'campaign_exit.json',rec,exclusive=True);(failures if rc else done).append(rec);jobs.remove(job)
  s.save(root/'status.json',dict(state='failed' if failures else 'CPU_head_fits_running' if jobs else 'CPU_heads_science_complete_awaiting_review',active=[dict(key=j['key'],pid=j['pid']) for j in jobs],complete=done,failures=failures))
  if jobs:time.sleep(1)
 assert not failures and len(done)==len(config()['cases'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test','group'],required=True);p.add_argument('--case');a=p.parse_args()
 try:globals()[a.phase](a.case)
 except BaseException as e:
  out=folder(a.case);mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
