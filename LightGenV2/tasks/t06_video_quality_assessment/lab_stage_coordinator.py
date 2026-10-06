"""Pure generation of the historical leased stage CLI script.

Calling desktop_code returns source only: it opens no connection, imports no
hardware SDK, schedules no task and starts no child. The returned script is not
executed here. Preserved from lab_manual_stage at commit 7093ec46082eed2fae127ec5028d3e2e8548b592.
"""

STAGES = ('vision_router','vision_expert','vision_global','language_router','language_expert','language_global')


def desktop_code(project,bench,session,config,stage,stem,stages=STAGES,capture_only=False):
    """Fixed CLI only; phase release first cancels this job's child tree."""
    return f'''import json,subprocess,time,sys
from pathlib import Path
p=Path({project!r});stem=Path({stem!r});py={bench!r}+'/.venv_gpu/Scripts/pythonw.exe'
status=stem.with_suffix('.json');heartbeat=stem.with_suffix('.heartbeat')
commands=[['capture','--stage',{stage!r},'--phase-ready']]
stages={list(stages)!r};i=stages.index({stage!r})
if not {capture_only!r}:commands.append(['prepare','--stage',stages[i+1],'--device','cuda'] if i<len(stages)-1 else ['evaluate','--device','cuda'])
child=None
def save(data):status.write_text(json.dumps(data,indent=2),encoding='utf-8')
try:
 with stem.with_suffix('.log').open('x',encoding='utf-8') as log:
  for command in commands:
   if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>20:raise RuntimeError('Phase lease expired before start')
   save(dict(state='running',action=command[0],stage={stage!r}))
   child=subprocess.Popen([py,'-u','run.py',*command,'--session',{session!r},'--config',{config!r},'--bench-root',{bench!r}],cwd=p,stdout=log,stderr=subprocess.STDOUT)
   while child.poll() is None:
    if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>20:
     subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait(timeout=10);raise RuntimeError('Phase lease expired; owned child stopped')
    time.sleep(.25)
   if child.returncode:raise RuntimeError('CLI failed: '+str(command))
 save(dict(state='done',stage={stage!r},phase_unchanged=True))
except BaseException as e:
 if child and child.poll() is None:
  subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait(timeout=10)
 save(dict(state='failed',error=str(e),stage={stage!r}))
'''
