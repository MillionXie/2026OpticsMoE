"""Run a finite admitted panel-D sequence, one GPU child at a time."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def queue_profiles(paths,gpu):
    values=[json.loads(p.read_text()) for p in paths]
    for v in values:
        if not v.get('panel_d_topk') or v['architecture']!='LG' or v['main_layers']!=2 or v['top_k'] not in (1,2,3,4):raise ValueError('Only approved two-layer panel D profiles')
        if v.get('selection_source')!='original_TEST_development' or gpu not in v['authorized_gpu_uuids']:raise ValueError('Explicit TEST development selection and GPU required')
    if len({v['dataset'] for v in values})!=1 or len({v['top_k'] for v in values})!=len(values):raise ValueError('One dataset and unique k per queue')
    return values


def save(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profiles',nargs='+',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--admission',type=Path,required=True);a=ap.parse_args();gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');configs=queue_profiles(a.profiles,gpu);admit=json.loads(a.admission.read_text())
    task=Path(__file__).resolve().parent
    for name,expected in admit['source_sha256'].items():
        if hashlib.sha256((task/name).read_bytes()).hexdigest()!=expected:raise ValueError('Admitted source SHA changed: '+name)
    a.out.mkdir(parents=True,exist_ok=False);history=[]
    for path,cfg in zip(a.profiles,configs):
        key=cfg['dataset']+f'_k{cfg["top_k"]}'
        if key not in admit['models']:raise ValueError('Exact model not admitted')
        destination=a.out/f'k{cfg["top_k"]}';command=[sys.executable,'-m','LightGenV2.tasks.t17_router_classification.train_figure_revision','--profile',str(path),'--out',str(destination)]
        env=dict(os.environ,T17_SOURCE_COMMIT=admit['source_commit'],PYTHONUNBUFFERED='1')
        with (a.out/f'k{cfg["top_k"]}.log').open('wb') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);save(a.out/'queue_status.json',dict(state='training',queue_pid=os.getpid(),child_pid=child.pid,top_k=cfg['top_k'],GPU=gpu,command=command,completed=history));rc=child.wait()
        history.append(dict(top_k=cfg['top_k'],child_pid=child.pid,returncode=rc,run=str(destination),time=time.time()))
        if rc:save(a.out/'queue_status.json',dict(state='failed',queue_pid=os.getpid(),completed=history));raise RuntimeError('Panel D child failed; remaining jobs were not started')
    save(a.out/'queue_status.json',dict(state='complete',queue_pid=os.getpid(),completed=history))


if __name__=='__main__':main()
