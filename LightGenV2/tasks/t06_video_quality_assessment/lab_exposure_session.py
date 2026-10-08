"""Create a new exposure session, reusing ONLY unchanged completed stages.

Original PNG/records remain untouched. New envelopes retain original record SHA
and acquisition hardware identity; changed effective settings forbid reuse.
"""
import argparse,copy,shutil
from pathlib import Path
from types import SimpleNamespace
from .lab_runtime import STAGES,read,write,sha
from .lab_bench import config,session,initialize,identity,effective_stage_identity

def retained_stages(measured,retain_prefix=None):
 if not measured or measured!=list(STAGES[:len(measured)]):raise ValueError('Expected contiguous source stages')
 n=len(measured) if retain_prefix is None else retain_prefix
 if not 1<=n<=len(measured) or n>=len(STAGES):raise ValueError('Retained prefix must be 1..5 and already complete')
 return measured[:n]

def rebase(root,source_name,new_name,new_config,retain_prefix=None):
 root=Path(root).resolve();old=session(root,source_name);dest=session(root,new_name)
 state=read(old/'session.json');c=config(new_config);stages=retained_stages(state['measured_stages'],retain_prefix)
 if state['release_sha256']!=sha(root/'release.json'):raise ValueError('Release changed')
 if state['hardware_sha256']!=identity(state['hardware_config']):raise ValueError('Source configuration identity invalid')
 for stage in stages:
  if effective_stage_identity(c,stage)!=effective_stage_identity(state['hardware_config'],stage):raise ValueError('Cannot inherit changed stage: '+stage)
  mf=read(old/'play'/stage/'manifest.json')
  if mf['hardware_sha256']!=state['hardware_sha256'] or mf['release_sha256']!=state['release_sha256']:raise ValueError('Source manifest identity mismatch')
  if sha(old/mf['phase_file'])!=mf['phase_sha256']:raise ValueError('Source phase changed')
  for e in mf['entries']:
   png=old/'ccd'/stage/(e['key']+'.png');r=read(png.with_suffix('.record.json'))
   if sha(png)!=r['sha256'] or sha(old/'play'/stage/e['bmp'])!=e['sha256']:raise ValueError('Source payload changed')
   if r['hardware_sha256']!=state['hardware_sha256'] or r['phase_sha256']!=mf['phase_sha256'] or r['amplitude_sha256']!=e['sha256']:raise ValueError('Source capture identity mismatch')
   if r['upstream_ccd_sha256']!=e['upstream_ccd_sha256']:raise ValueError('Source upstream manifest mismatch')
   for prev,digest in r['upstream_ccd_sha256'].items():
    if sha(old/'ccd'/prev/(e['key']+'.png'))!=digest:raise ValueError('Source upstream CCD changed')
 initialize(SimpleNamespace(project=str(root),config=str(new_config),session=new_name,fields=0))
 new=read(dest/'session.json')
 if new['fields']!=state['fields']:raise ValueError('Source must use full unchanged field manifest')
 for stage in stages:
  for folder in ('play','ccd','theoretical_ccd'):
   shutil.copytree(old/folder/stage,dest/folder/stage)
  mf=read(dest/'play'/stage/'manifest.json');mf['hardware_sha256']=new['hardware_sha256'];write(dest/'play'/stage/'manifest.json',mf)
  for e in mf['entries']:
   oldrec=old/'ccd'/stage/(e['key']+'.record.json');r=read(oldrec)
   r['inherited_acquisition']=dict(source_session=source_name,original_record_sha256=sha(oldrec),original_hardware_sha256=r['hardware_sha256'],effective_stage_identity=effective_stage_identity(c,stage))
   r['hardware_sha256']=new['hardware_sha256'];write(dest/'ccd'/stage/oldrec.name,r)
 new['measured_stages']=stages;new['inheritance']=dict(source_session=source_name,source_session_sha256=sha(old/'session.json'),retained_prefix=len(stages),excluded_source_stages=state['measured_stages'][len(stages):],reason='Explicit verified prefix reuse; all downstream inputs and CCDs must be regenerated; effective inherited settings unchanged')
 write(dest/'session.json',new);write(dest/'status.json',dict(status='inherited_verified_stages',stages=stages))
 print('REBASED',dest,stages,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.');p.add_argument('--source-session',required=True);p.add_argument('--new-session',required=True);p.add_argument('--config',required=True);p.add_argument('--retain-prefix',type=int,help='Keep only this many verified leading stages; never copies downstream data');a=p.parse_args();rebase(a.project,a.source_session,a.new_session,a.config,a.retain_prefix)
if __name__=='__main__':main()
