"""Finite matched refinement queue; automatic evaluation is inside each trainer."""
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,time
from pathlib import Path
import sys
root=Path(sys.argv[1]);source=Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=False)
for task in ['sonyc','sen12ms']:(root/(task+'_data')).symlink_to(source/(task+'_data'),target_is_directory=True)
py=sys.executable
recipe=dict(epochs=60,lr=.003,phase_dropout=.1,moe_balance=.1,weight_power=.5,augmentation=True,selection='validation only; compare common recipe by mean validation score across architectures, never test score')
(root/'recipe.json').write_text(json.dumps(recipe,indent=2))
def worker(gpu,tasks):
 for task,arch in tasks:
  name=task+'_'+arch;cmd=[py,'-m','LightGenV2.tasks.t11_multimodal_nature.recovery_train','--data',str(root/(task+'_data')),'--out',str(root/name),'--arch',arch,'--epochs','60','--lr','.003','--phase-dropout','.1','--balance','.1','--weight-power','.5','--augment']
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1')
  with (root/(name+'.log')).open('w') as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
   (root/(name+'_launch.json')).write_text(json.dumps(dict(pid=p.pid,gpu=gpu,command=cmd),indent=2));print('LAUNCHED',name,p.pid,flush=True);rc=p.wait()
  print('FINISHED',name,rc,flush=True)
  if rc:raise RuntimeError(name+' failed; see log')
with ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(worker,'4',[('sonyc','moe')]),pool.submit(worker,'5',[('sonyc','d2nn')]),pool.submit(worker,'6',[('sen12ms','moe'),('sen12ms','d2nn')])]
 failures=[]
 for f in futures:
  try:f.result()
  except Exception as e:failures.append(str(e))
(root/'queue_final.json').write_text(json.dumps(dict(status='failed' if failures else 'complete',failures=failures),indent=2))
if failures:raise RuntimeError(failures)
subprocess.run([py,'-m','LightGenV2.tasks.t11_multimodal_nature.recovery_report','--root',str(root)],check=True)
