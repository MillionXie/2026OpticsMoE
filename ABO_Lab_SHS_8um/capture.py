"""Bounded camera-only diagnostic; raw PNG, exact settings readback, restore on exit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from sdk import Camera


NODES=('AcquisitionFrameRate','ExposureTime','Gain','TestPattern','SyncMode','PixelFormat','Width','Height','OffsetX','OffsetY')


def snapshot(camera):
    result={}
    for name in NODES:
        try:result[name]={'value':camera.get(name),'info':camera.info(name)}
        except RuntimeError as ex:result[name]={'unavailable':str(ex)}
    return result


def save_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config.json');p.add_argument('--out',required=True,type=Path)
    p.add_argument('--frames',type=int,default=3)
    p.add_argument('--exposures-us',type=float,nargs='+',help='Omit to use config exposure_us')
    p.add_argument('--fps',type=int,help='Set frame rate BEFORE exposure; restored afterwards')
    p.add_argument('--test-pattern',choices=['Normal','VStrip','HStrip','Black','White'])
    p.add_argument('--no-png',action='store_true',help='Benchmark only; no PNG encoding/disk writes in frame loop')
    a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))['camera']
    if not 1<=a.frames<=300:raise ValueError('Use 1..300 frames per bounded test')
    exposures=a.exposures_us or [c.get('exposure_us')]
    if len(exposures)>16:raise ValueError('At most 16 exposure settings')
    fps=a.fps if a.fps is not None else c.get('frame_rate_hz')
    a.out.mkdir(parents=True,exist_ok=False)
    report={'operation':'camera-only; no SLM connected/commanded','config':c,'frames':[],
            'postprocessing':'none: raw integer samples; no minmax/log/gamma/resize/flip',
            'complete':False,'restored':False,'png_saved':not a.no_png}
    path=a.out/'capture.json';save_json(path,report)
    with Camera(c) as camera:
        before=snapshot(camera);report['before']=before;save_json(path,report)
        changed=[]
        def change(name,value):
            if name not in changed:changed.append(name)
            return camera.set(name,value)
        try:
            if fps is not None:change('AcquisitionFrameRate',fps)
            if c.get('gain') is not None:change('Gain',c['gain'])
            if a.test_pattern is not None:change('TestPattern',a.test_pattern)
            elif before.get('TestPattern',{}).get('value')!='Normal':
                raise RuntimeError('Camera has a synthetic test pattern active; use --test-pattern Normal for real scene')
            for ei,exposure in enumerate(exposures):
                if exposure is not None:change('ExposureTime',exposure)
                settings=snapshot(camera);camera.start()
                for _ in range(c.get('buffer_count',4)+2):camera.grab()
                last_id=None;start=time.perf_counter()
                for i in range(a.frames):
                    frame,meta=camera.grab()
                    if last_id is not None and meta['frame_id']<=last_id:raise RuntimeError('Repeated/non-increasing frame ID')
                    last_id=meta['frame_id']
                    meta.update(exposure_us=float(settings['ExposureTime']['value']),frame_rate_hz=float(settings['AcquisitionFrameRate']['value']),
                                gain=settings['Gain'].get('value'),minimum=int(frame.min()),maximum=int(frame.max()),mean=float(frame.mean()),
                                p99=float(np.percentile(frame,99)),saturated_fraction=float((frame==(1<<meta['significant_bits'])-1).mean()))
                    if not a.no_png:
                        filename=f'e{ei:02d}_f{i:04d}.png';Image.fromarray(frame).save(a.out/filename)
                        with Image.open(a.out/filename) as im:
                            if not np.array_equal(frame,np.asarray(im)):raise RuntimeError('PNG round-trip mismatch')
                        meta.update(file=filename,sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest())
                    report['frames'].append(meta)
                    if not a.no_png:save_json(path,report)
                report.setdefault('groups',[]).append({'settings':settings,'frames':a.frames,'elapsed_s':time.perf_counter()-start,
                    'note':'Includes copying/statistics and optional PNG, not maximum sensor rate'})
                camera.stop();save_json(path,report)
            report['complete']=True
        finally:
            camera.stop()
            errors=[]
            # Reverse order puts old exposure back before increasing frame rate.
            for name in reversed(changed):
                try:camera.set(name,before[name]['value'])
                except Exception as ex:errors.append(f'{name}: {ex}')
            report['restore_errors']=errors;report['after']=snapshot(camera)
            report['restored']=not errors
            save_json(path,report)
    print(json.dumps({'complete':report['complete'],'frames':len(report['frames']),'restored':report['restored'],'out':str(a.out.resolve())}),flush=True)


if __name__=='__main__':main()
