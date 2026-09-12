"""Bounded SLM/SHS joint test; manual phase unchanged, no task accuracy claims."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from slm_camera import Controller
from capture import save_json,snapshot


def metrics(a):
    return {'mean':float(a.mean()),'max':int(a.max()),'p99':float(np.percentile(a,99)),
            'saturation_fraction':float((a==255).mean())}


def compare(a,b):
    a=a.astype(float).ravel();b=b.astype(float).ravel()
    den=np.linalg.norm(a-a.mean())*np.linalg.norm(b-b.mean())
    pcc=float(np.dot(a-a.mean(),b-b.mean())/den) if den else None
    return dict(pcc=pcc,nmae=float(np.mean(np.abs(a-b))/max(float(b.mean()),1)),
                mean_ratio=float(a.mean()/max(b.mean(),1)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='LAB.local.json');p.add_argument('--out',required=True,type=Path)
    p.add_argument('--mode',choices=['scout','gray','timing'],default='scout')
    p.add_argument('--exposures-us',nargs='+',type=float,default=[50,200,1000])
    args=p.parse_args();c=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if len(args.exposures_us)>5:raise ValueError('At most 5 scout exposures')
    args.out.mkdir(parents=True,exist_ok=False);patterns=args.out/'patterns';patterns.mkdir()
    # Same 8.126 mm support as ABO; large asymmetric halves identify swapped/old frames.
    paths={}
    for name in ('black','left','right','active'):
        a=np.zeros((1080,1920),np.uint8)
        if name=='left':a[32:1048,452:960]=255
        if name=='right':a[32:1048,960:1468]=255
        if name=='active':a[32:1048,452:1468]=255
        paths[name]=patterns/(name+'.bmp');Image.fromarray(a).save(paths[name])
    if args.mode=='gray':
        for gray in (0,64,128,192,255):
            a=np.zeros((1080,1920),np.uint8);a[32:1048,452:1468]=gray
            name=f'gray{gray:03d}';paths[name]=patterns/(name+'.bmp');Image.fromarray(a).save(paths[name])
    report={'config':c,'mode':args.mode,'phase':'operator-loaded uniform black, not changed by software',
            'frames':[],'complete':False,'camera_restored':False,'postprocessing':'raw Mono8; none'}
    source=Path(__file__).resolve().parent/'CODE_MANIFEST.json'
    if source.exists():report['source_commit']=json.loads(source.read_text())['commit']
    save_json(args.out/'report.json',report)
    with Controller(c) as hw:
        report['devices']=hw.info;report['before']=hw.before
        def grab(name,label):
            frame,meta=hw.capture(paths[name]);meta.update(pattern=name,label=label,**metrics(frame))
            file=f'{len(report["frames"]):03d}_{label}_{name}.png';Image.fromarray(frame).save(args.out/file)
            meta.update(file=file,sha256=hashlib.sha256((args.out/file).read_bytes()).hexdigest())
            report['frames'].append(meta);save_json(args.out/'report.json',report)
            print(label,name,'mean',meta['mean'],'p99',meta['p99'],'sat',meta['saturation_fraction'],flush=True)
            return frame,meta
        try:
            if args.mode in ('scout','gray'):
                for exposure in args.exposures_us:
                    hw.camera.stop();hw.camera.set('ExposureTime',exposure);hw.camera.start()
                    hw.camera_settings=snapshot(hw.camera)
                    c['settle_delay_ms']=200
                    names=paths if args.mode=='scout' else [n for n in paths if n.startswith('gray')]
                    for name in names:grab(name,f'e{exposure:g}')
            else:
                refs={};c['settle_delay_ms']=400
                for name in ('left','right'):refs[name]=grab(name,'reference')[0]
                repeats={name:grab(name,'reference_repeat')[0] for name in ('left','right')}
                separation=float(np.mean(np.abs(refs['left'].astype(float)-refs['right'])))
                noise=max(float(np.mean(np.abs(refs[n].astype(float)-repeats[n]))) for n in refs)
                report['reference_check']={'mean_absolute_difference':separation,
                    'repeat_difference':noise,'required_ratio':5,
                    'passed':separation>5*max(noise,.1)}
                save_json(args.out/'report.json',report)
                if not report['reference_check']['passed']:
                    raise ValueError('Cannot resolve SLM input changes; timing scan stopped, no recommended wait')
                # Alternation prevents accidentally validating a stale identical pattern.
                for delay in (0,20,50,100,200,400):
                    c['settle_delay_ms']=delay
                    for rep in range(2):
                        for name in ('left','right'):
                            a,row=grab(name,f'd{delay}_r{rep}')
                            row['same_reference']=compare(a,refs[name])
                            row['opposite_reference']=compare(a,refs['right' if name=='left' else 'left'])
                            save_json(args.out/'report.json',report)
            report['complete']=True
        finally:
            # Only amplitude changes; do not address the phase SLM.
            hw.slm.preload_files([paths['black']]);hw.slm.display_file(paths['black'])
            save_json(args.out/'report.json',report)
    report['camera_restored']=True;save_json(args.out/'report.json',report)


if __name__=='__main__':main()
