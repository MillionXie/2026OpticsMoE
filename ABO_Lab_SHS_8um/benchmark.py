"""Short direct-pull benchmark; no PNG, plots or percentiles inside capture loop."""
import argparse,json,time
from pathlib import Path
import numpy as np
from sdk import Camera
from capture import snapshot,save_json,restore_settings


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--fps',type=int,default=1000)
    p.add_argument('--exposure-us',type=float,default=100);p.add_argument('--frames',type=int,default=200)
    p.add_argument('--mode',choices=['copy','buffer'],default='copy')
    a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))['camera']
    if not 10<=a.frames<=1000:raise ValueError('Use 10..1000 frames for bounded benchmark')
    a.out.mkdir(parents=True,exist_ok=False);report={'complete':False,'restored':False}
    with Camera(c) as camera:
        before=snapshot(camera);report['before']=before;save_json(a.out/'benchmark.json',report)
        try:
            camera.set('AcquisitionFrameRate',a.fps);camera.set('ExposureTime',a.exposure_us)
            report['settings']=snapshot(camera);camera.start()
            for _ in range(8):camera.grab()
            rows=[];t=time.perf_counter()
            for _ in range(a.frames):
                if a.mode=='copy':frame,meta=camera.grab()
                else:meta=camera.receive_buffer_only()
                rows.append(meta)
            elapsed=time.perf_counter()-t;camera.stop()
            ids=np.array([r['frame_id'] for r in rows],np.int64);diff=np.diff(ids)
            if (diff<=0).any():raise RuntimeError('Frame IDs repeated/reversed')
            report.update(complete=True,mode=a.mode,received_frames=a.frames,elapsed_s=elapsed,host_received_fps=a.frames/elapsed,
                skipped_sensor_frame_ids=int(np.maximum(diff-1,0).sum()),
                median_grab_ms=float(np.median([r['copy_wait_decode_ms' if a.mode=='copy' else 'receive_release_excluded_ms'] for r in rows])),frames=rows,
                scope=('Host raw copy + decode + completeness checks.' if a.mode=='copy' else 'DMA buffer get/return + cached metadata only; no pixel copy or full completeness validation.')+' NO SLM, PNG encoding, inference, or hardware trigger. Not end-to-end optical latency.')
            if a.mode=='copy':report['host_copy_decode_fps']=a.frames/elapsed
        finally:
            errors=restore_settings(camera,before,['ExposureTime','AcquisitionFrameRate'])
            report['restore_errors']=errors;report['restored']=not errors;report['after']=snapshot(camera)
            save_json(a.out/'benchmark.json',report)
    print({k:v for k,v in report.items() if k not in ('frames','before','after','settings')})


if __name__=='__main__':main()
