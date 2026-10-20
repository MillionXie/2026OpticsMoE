"""User-authorized native-only tail; unchanged published strict GPU worker.

The 11:17 admission supersedes only the old 11:10 rental admission cap.
It does not change timing, forwards, precision, source guards or accounting.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from . import rental_strict_tf_dual_20261020 as old

DEADLINE = 1792466220
LEASE_END = 1792466607
EXTENDED_DEADLINE = 1792466880  # user extended paid lease to11:35; reserve seven minutes
EXTENDED_LEASE_END = 1792467300
ROOT = old.ROOT / 'native_tail01'
PLAN = [('deepseek','temporal',100,2),('kimi','mps',100,1),
        ('kimi','mps',50,1),('kimi','temporal',100,6)]


def validate(deadline):
    if deadline not in (DEADLINE,EXTENDED_DEADLINE) or not 0 < deadline-time.time() <= 7200:
        raise ValueError('Only explicit paid native-tail authorizations')
    if (LEASE_END if deadline==DEADLINE else EXTENDED_LEASE_END)-deadline < 360:
        raise ValueError('Preserve at least six minutes for backup')


def daemon(start):
    if start:
        Path(old.PIPE).mkdir(mode=0o700,exist_ok=True)
        (ROOT/'mps_logs').mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=old.placement('kimi','native')[0],
                 CUDA_MPS_PIPE_DIRECTORY=old.PIPE,CUDA_MPS_LOG_DIRECTORY=str(ROOT/'mps_logs'))
        subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True,timeout=10)
    else:
        for server in old.old_busy.numeric_lines(old.mps('get_server_list')):
            if old.old_busy.numeric_lines(old.mps('get_client_list '+str(server))):
                raise RuntimeError('Do not terminate MPS with clients')
        old.mps('quit')
        for _ in range(15):
            if not old.records():return
            time.sleep(1)
        raise RuntimeError('Own GPU contexts remain')


def suite(args):
    validate(args.deadline)
    if ROOT.exists() or old.records():raise RuntimeError('Fresh tail and idle allocated GPUs required')
    previous=old.ROOT/'attempt02'
    required=['mps_deepseek_native_100','mps_deepseek_native_50',
              'mps_deepseek_fp32_100','mps_deepseek_fp32_50','temporal_deepseek_fp32_100',
              'mps_kimi_fp32_100','mps_kimi_fp32_50','temporal_kimi_fp32_100']
    for name in required:
        if json.loads((previous/name/'status.json').read_text())['status']!='complete':
            raise RuntimeError('Predecessor not complete: '+name)
    ROOT.mkdir()
    state=dict(status='running',deadline=args.deadline,lease_end=LEASE_END if args.deadline==DEADLINE else EXTENDED_LEASE_END,completed=[],
               predecessor_complete=required,sources=old.sources(),
               admission_only_override=True,worker_unchanged=True,pid=os.getpid())
    def save():old.base.old.write(ROOT/'suite_status.json',state)
    active=False
    try:
        for model,method,cap,batch in PLAN:
            if args.continuation==2:
                predecessor=old.ROOT/'native_tail01'/f'{method}_{model}_native_{cap}'
                if predecessor.exists():
                    if not (predecessor/'status.json').exists() or json.loads((predecessor/'status.json').read_text())['status']!='complete':
                        raise RuntimeError('Preserve unfinished predecessor; no automatic retry')
                    state.setdefault('preserved_complete',[]).append(predecessor.name);save();continue
            if time.time()>=args.deadline-180:raise TimeoutError('No new worker in final three minutes')
            if active and method!='mps':daemon(False);active=False
            if method=='mps' and not active:
                if old.records():raise RuntimeError('Foreign process before MPS')
                daemon(True);active=True
            name=f'{method}_{model}_native_{cap}'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=old.placement(model,'native')[0],
                     OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     HF_HOME='/data/cache',TMPDIR='/data/tmp')
            if method=='mps':env.update(CUDA_MPS_PIPE_DIRECTORY=old.PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap))
            else:
                env.pop('CUDA_MPS_PIPE_DIRECTORY',None);env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
            cmd=[f'/data/venvs/{model}/bin/python','-u','-m',__package__+'.resume_strict_native_20261020',
                 '--model',model,'--method',method,'--cap',str(cap),'--batch',str(batch),
                 '--deadline',str(args.deadline),'--continuation',str(args.continuation)]
            old.base.old.write(ROOT/(name+'.command.json'),cmd)
            with (ROOT/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=old.REPO,env=env,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(phase=name,child_pid=child.pid);save()
                try:code=child.wait(timeout=max(1,args.deadline-time.time()))
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=20);raise
            if code:raise RuntimeError(f'{name} exit {code}; no retry')
            state['completed'].append(name);save()
        state.update(status='complete',finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:daemon(False);state['mps_active']=False
            except BaseException as error:state['mps_cleanup_error']=repr(error)
        save()


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',action='store_true')
    parser.add_argument('--continuation',type=int,choices=(1,2),default=1)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--model',choices=('deepseek','kimi'))
    parser.add_argument('--method',choices=('mps','temporal'))
    parser.add_argument('--cap',type=int,choices=(50,100))
    parser.add_argument('--batch',type=int)
    args=parser.parse_args();validate(args.deadline)
    if args.continuation==2:
        if args.deadline!=EXTENDED_DEADLINE:raise ValueError('New paid deadline for continuation2')
        ROOT=old.ROOT/'native_tail02'
    if args.suite:return suite(args)
    if (args.model,args.method,args.cap,args.batch) not in PLAN:raise ValueError('Fixed missing native point only')
    args.precision='native';args.accuracy=False;args.dual_fp32=False;args.power_limit_w=400
    args.gpu_uuids=old.placement(args.model,'native');args.gpu_uuid=args.gpu_uuids[0]
    args.output=ROOT/f'{args.method}_{args.model}_native_{args.cap}'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Native single-card only')
    old.worker(args)


if __name__=='__main__':main()
