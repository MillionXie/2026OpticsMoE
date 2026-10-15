"""Finite one-way lab receiver campaign: current TRAIN -> TEST -> tiny readout."""
import argparse,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path


def main(a):
    assert not a.out.exists();a.out.mkdir(parents=True)
    root=Path(__file__).resolve().parents[3]
    (a.out/'claim.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),arguments={k:str(v) for k,v in vars(a).items()},git_commit=subprocess.check_output([a.git,'rev-parse','HEAD'],cwd=root,text=True).strip(),sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('lab_chain_20261016.py','lab_readout_20261016.py','lab_caltech10_20261015.py')},laboratory_receiver_only=True,lab_git_push=False),indent=2),encoding='utf-8')
    deadline=time.monotonic()+8*3600
    while not (a.train/'report.json').exists():
        if (a.train/'failure.json').exists():raise RuntimeError('TRAIN capture failed; retain data, no automatic optical retry')
        if time.monotonic()>deadline:raise TimeoutError('TRAIN wait exceeded8h')
        time.sleep(15)
    report=json.loads((a.train/'report.json').read_text());assert report['state']=='complete' and report['sdk_released'] and report['physical_ccd']==6088
    released=time.monotonic()+30
    lock=Path(__file__).parent/'runs/hardware/.t18_sdk_owner.lock'
    while lock.exists():
        if time.monotonic()>released:raise RuntimeError('SDK lock not released after TRAIN completion')
        time.sleep(.5)
    script=Path(__file__).with_name('lab_caltech10_20261015.py')
    command=[sys.executable,str(script),'--mode','capture','--assets',str(a.assets),'--out',str(a.test),'--selftest',str(a.selftest),'--role','test','--exposure-us','2000','--response-lut',str(a.response_lut),'--coordinates',str(a.coordinates),'--machine-config',str(a.machine_config),'--amplitude-sdk',str(a.amplitude_sdk),'--phase-sdk',str(a.phase_sdk),'--phase-lut',str(a.phase_lut)]
    with (a.out/'test_capture.log').open('x',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=12*3600)
    assert result.returncode==0,'TEST capture failed: no adaptation or automatic restart'
    with (a.out/'readout.log').open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(script.with_name('lab_readout_20261016.py')),'--train',str(a.train),'--test',str(a.test),'--out',str(a.out/'readout'),'--epochs','300'],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    assert result.returncode==0,'Electronic fit/audit failed; preserve all results'
    (a.out/'complete.json').write_text(json.dumps(dict(state='complete',train=str(a.train),test=str(a.test),readout=str(a.out/'readout'),val_used=False,lab_git_push=False,source_git_commit=json.loads((a.out/'claim.json').read_text())['git_commit']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('out','train','test','assets','selftest','response-lut','coordinates','machine-config','amplitude-sdk','phase-sdk','phase-lut','git'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:main(a)
    except BaseException as error:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False),indent=2),encoding='utf-8')
        raise
