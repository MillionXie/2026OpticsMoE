"""Finite data-ready dependency for SAME simulation-head physical fine-tuning."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path


def save(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def main(a):
    assert not a.out.exists();a.out.mkdir(parents=True)
    manifest=json.loads(a.manifest.read_text());assert manifest['depth']==a.depth
    save(a.out/'claim.json',dict(pid=os.getpid(),depth=a.depth,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),handoff_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),capture_pair=str(a.capture_pair),manifest=str(a.manifest),sdk_opened=False,random_head_fallback=False,laboratory_receiver_only=True))
    deadline=time.monotonic()+12*3600
    while not (a.capture_pair/'report.json').exists():
        for path in (a.capture_pair/'failure.json',a.capture_pair.parent/'failure.json'):
            if path.exists():raise RuntimeError('Capture queue failed; preserve data, do not fit a partial dataset')
        if time.monotonic()>deadline:raise TimeoutError('Accepted optical dataset did not complete within12h')
        time.sleep(15)
    report=json.loads((a.capture_pair/'report.json').read_text())
    assert report['state']=='complete' and report['sdk_released'] and not report['pilot'] and report['main_depth']==a.depth
    assert report['physical_ccd']==2*(761+1833)*(a.depth+1)
    script=Path(__file__).with_name('lab_readout_20261016.py');root=script.parents[3]
    command=[sys.executable,str(script),'--train',str(a.capture_pair/'train'),'--test',str(a.capture_pair/'test'),'--out',str(a.out/'readout'),'--assets',str(a.assets),'--simulation-head-manifest',str(a.manifest)]
    with (a.out/'readout.log').open('x',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    assert result.returncode==0,'Same-head physical fine-tuning/audit failed; no automatic retry'
    fit=json.loads((a.out/'readout/report.json').read_text());assert fit['state']=='complete' and not fit['validation_used']
    save(a.out/'complete.json',dict(state='complete',depth=a.depth,sdk_opened=False,head_parameters=8554,normalization_frozen=True,readout_report_sha256=hashlib.sha256((a.out/'readout/report.json').read_bytes()).hexdigest()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('capture-pair','manifest','assets','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--depth',type=int,choices=[2,4],required=True);a=parser.parse_args()
    try:main(a)
    except BaseException as error:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
