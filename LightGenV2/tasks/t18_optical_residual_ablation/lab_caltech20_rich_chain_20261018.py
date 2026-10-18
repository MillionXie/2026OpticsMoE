"""One-shot follow-up: wait current residual full, fit original head, next arm.

Not a recurring automation. There is no restart, changed exposure, alternate PT,
new layer or overlapping SDK owner. Every failure preserves evidence and stops.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
from lab_caltech10_20261015 import save,sha


def read(path):return json.loads(Path(path).read_text())


def await_full(a,deadline):
    folder=a.session/'full_residual';report=folder/'report.json';last=0.
    while not report.exists():
        if (folder/'failure.json').exists():raise RuntimeError('Existing residual capture failed; preserve, do not restart')
        if time.monotonic()>deadline:raise TimeoutError('Follow-up deadline; no automatic retry')
        if time.monotonic()-last>=60:
            progress=read(folder/'progress.json') if (folder/'progress.json').exists() else None
            save(a.out/'progress.json',dict(state='waiting_existing_residual_capture',capture=progress))
            last=time.monotonic()
        time.sleep(5)
    value=read(report);assert value['state']=='complete' and value['sdk_released'] and value['physical_ccd']==4233 and not value['pilot'] and value['arm']=='residual'
    assert value['release_sha256']==sha(a.session/'release.json')


def await_unlock():
    lock=HERE/'runs/hardware/.t18_sdk_owner.lock';deadline=time.monotonic()+60
    while lock.exists():
        if time.monotonic()>deadline:raise RuntimeError('SDK owner remains after completed report; no overlap')
        time.sleep(.5)


def main(a):
    assert not a.out.exists();a.out.mkdir(parents=True);started=time.perf_counter()
    release=a.session/'release.json';digest=sha(release);manifest=read(release)
    assert manifest['capture_released'] and manifest['TRAIN']==1000 and manifest['TEST']==411 and manifest['VALID']==0
    source_names=('lab_caltech20_rich_model_20261018.py','lab_caltech20_rich_capture_20261018.py','lab_caltech20_rich_readout_20261018.py',Path(__file__).name)
    sources={name:sha(HERE/name) for name in source_names}
    assert sources[source_names[0]]==manifest['model_source_sha256'] and sources[source_names[1]]==manifest['capture_source_sha256']
    save(a.out/'contract.json',dict(source_files_SHA256=sources,release_sha256=digest,human_followup='User confirmed optical conditions and requested full capture despite0of20 pilot, then electronic fine-tuning',order=['wait existing residual full4233','fit same residual5140 head on TRAIN1000/TEST411 development selection','control pilot60','control full4233','fit same control5140 head'],VALID=0,one_shot=True,automatic_retry=False,controller_pid=os.getpid()))
    deadline=time.monotonic()+3*3600;await_full(a,deadline);await_unlock();steps=[]
    def run(step,args):
        assert sha(release)==digest and sources=={name:sha(HERE/name) for name in source_names}
        save(a.out/'progress.json',dict(state='running',step=step,elapsed_s=time.perf_counter()-started))
        print(json.dumps(dict(step=step,argv=args)),flush=True)
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        child=Path(sys.executable).with_name('python.exe') if os.name=='nt' else Path(sys.executable)
        with (a.out/(step+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run([str(child),*map(str,args)],cwd=HERE.parents[2],env=env,creationflags=flags,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(step,result.returncode)
        steps.append(dict(step=step,returncode=result.returncode,elapsed_s=time.perf_counter()-started))
        save(a.out/'steps.json',steps)
    def fit(arm):
        target=a.session/('readout_'+arm+'_120')
        run('fit_'+arm,[HERE/'lab_caltech20_rich_readout_20261018.py','--arm',arm,'--capture',a.session/('full_'+arm),'--assets',a.session/'assets','--release',release,'--out',target])
        result=read(target/'report.json');assert result['state']=='complete' and result['epochs']==120
        print(json.dumps(dict(event='adapted_result',arm=arm,baseline_TEST=result['baseline_TEST_development']['accuracy'],best_TEST_development=result['results']['best']['TEST_development']['accuracy'],best_epoch=result['results']['best']['epoch'])),flush=True)
    fit('residual')
    common=['--arm','control','--assets',a.session/'assets','--release',release,'--release-sha',digest,'--bridge',a.session/'cpu_final_release_bridge','--previous-arm',a.session/'full_residual','--coordinates',a.coordinates,'--machine-config',a.machine_config,'--phase-lut',a.phase_lut,'--response-lut',a.response_lut,'--phase-sdk',a.phase_sdk,'--amplitude-sdk',a.amplitude_sdk]
    for mode in ('pilot','full'):
        await_unlock();target=a.session/(mode+'_control');args=[HERE/'lab_caltech20_rich_capture_20261018.py','--mode',mode,'--out',target,*common]
        if mode=='full':args+=['--pilot',a.session/'pilot_control']
        run('capture_'+mode+'_control',args)
        report=read(target/'report.json');assert report['state']=='complete' and report['sdk_released'] and report['physical_ccd']==(60 if mode=='pilot' else 4233)
    await_unlock();fit('control')
    save(a.out/'report.json',dict(state='complete',steps=steps,elapsed_s=time.perf_counter()-started,sdk_released=True,physical_full_CCD=8466,extra_pilot_CCD=120,source_files_SHA256=sources,release_sha256=digest,TEST_is_development_not_independent=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('session','out','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:main(a)
    except BaseException as error:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
