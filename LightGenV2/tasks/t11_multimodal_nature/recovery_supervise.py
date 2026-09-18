"""One finite queue: prepare -> train -> test, bounded to three selected GPUs."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path

def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def alive(pid):
 p=Path(f'/proc/{pid}/stat')
 return p.exists() and p.read_text().split()[2]!='Z'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root;prep=json.loads((root/'preparation_jobs.json').read_text());py='/home/guest3/miniconda3/envs/xml/bin/python';jobs={};state={};audio_launched=False;sen_stage=0
 configs={'sonyc_moe':('sonyc','moe','4'),'sonyc_d2nn':('sonyc','d2nn','5'),'sen12ms_moe':('sen12ms','moe','6'),'sen12ms_d2nn':('sen12ms','d2nn','6')}
 def launch(name):
  task,arch,gpu=configs[name];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1');cmd=[py,'-m','LightGenV2.tasks.t11_multimodal_nature.recovery_train','--data',str(root/(task+'_data')),'--out',str(root/name),'--arch',arch,'--epochs','30','--batch','32']
  with (root/(name+'.log')).open('w') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  jobs[name]=p;state[name]=dict(status='running',pid=p.pid,gpu=gpu,command=cmd);print('LAUNCHED '+name+' '+str(p.pid),flush=True)
 while True:
  for task in prep:
   if (root/(task+'_data')/'manifest.json').is_file():state[task+'_prepare']={'status':'complete'}
   elif not alive(prep[task]['pid']):state[task+'_prepare']={'status':'failed','log':str(root/(task+'_prepare.log'))}
   else:state[task+'_prepare']={'status':'running','pid':prep[task]['pid']}
  if not audio_launched and state['sonyc_prepare']['status']=='complete':
   launch('sonyc_moe');launch('sonyc_d2nn');audio_launched=True
  if sen_stage==0 and state['sen12ms_prepare']['status']=='complete':launch('sen12ms_moe');sen_stage=1
  for name,proc in jobs.items():
   rc=proc.poll()
   if rc is not None:state[name].update(status='complete' if rc==0 and (root/name/'results.json').is_file() else 'failed',returncode=rc)
  if sen_stage==1 and state['sen12ms_moe']['status'] in ['complete','failed']:launch('sen12ms_d2nn');sen_stage=2
  save(root/'queue_status.json',dict(updated=time.time(),jobs=state))
  if all(s['status'] in ['complete','failed'] for s in state.values()) and (audio_launched or state['sonyc_prepare']['status']=='failed') and (sen_stage==2 or state['sen12ms_prepare']['status']=='failed'):break
  time.sleep(15)
 print('QUEUE FINISHED '+json.dumps(state),flush=True)
if __name__=='__main__':main()
