"""Human-authorized bounded white-frame diagnosis; never resumes a dataset.

Keep2000us/GainX4/240ms and the exact frozen expert phase. Alternate a zero
input, sourceTRAIN293 and failed sourceTRAIN294. Retain every diagnostic frame
and complete GenTL metadata. No retries, exposure changes or LUT updates.
"""
import argparse,hashlib,io,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]


def statistics(x):
    return dict(shape=list(x.shape),minimum=int(x.min()),maximum=int(x.max()),mean=float(x.mean()),p99=float(np.percentile(x,99)),saturation_fraction=float((x==255).mean()))


def plans():return ['black','black','previous','previous','failed','failed','previous','failed','black','black']


def selftest():
    assert len(plans())==10 and set(plans())=={'black','previous','failed'}
    assert statistics(np.full((4,4),255,np.uint8))['saturation_fraction']==1
    assert statistics(np.zeros((4,4),np.uint8))['mean']==0


def diagnose(a):
    import cv2
    from lab_caltech10_20261015 import Bench,corners,native,save,sha
    assert not a.out.exists();assert sha(a.release)==a.release_sha
    release=json.loads(a.release.read_text());assert release['capture_released']
    assert sha(a.coordinates)==release['coordinates_sha256'] and sha(a.machine_config)==release['machine_config_sha256']
    assert sha(a.phase_lut)==release['phase_lut_sha256'] and sha(a.response_lut)==release['response_lut_sha256']
    a.exposure_us=2000
    source=a.session/'full_residual/residual/train/0';phase=a.session/'full_residual/phases/residual_0.bmp'
    failure=json.loads((a.session/'full_residual/failure.json').read_text());assert failure['physical_ccd']==1706
    gray={};protected={}
    for name,index in (('previous',293),('failed',294)):
        prefix=source/f'{index:06d}';receipt=json.loads(prefix.with_suffix('.json').read_text())
        path=prefix.with_name(prefix.name+'_input.png');assert sha(path)==receipt['canonical_sha256']
        assert receipt['phase_sha256']==sha(phase)
        gray[name]=np.asarray(Image.open(path));assert gray[name].shape==(478,478)
        for suffix in ('.json','_raw.png','.png','_input.png'):
            p=prefix.with_name(prefix.name+suffix);protected[str(p)]=sha(p)
    gray['black']=np.zeros((478,478),np.uint8)
    response=json.loads(a.response_lut.read_text());assert response['passed'];drive=np.asarray(response['drive'],np.uint8)
    assert drive.shape==(256,) and drive[0]==0 and drive[-1]==255
    warp=cv2.getPerspectiveTransform(np.float32(corners(a.coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    a.out.mkdir(parents=True);paths={}
    for name,g in gray.items():
        p=a.out/(name+'.bmp');Image.fromarray(native(drive[g],'amplitude')).save(p);paths[name]=p
        if name!='black':
            index=293 if name=='previous' else 294;original=json.loads((source/f'{index:06d}.json').read_text())
            assert sha(p)==original['amplitude_sha256'],'Replay must be exact same native BMP bytes'
    save(a.out/'contract.json',dict(scope='bounded_diagnostic_not_formal_dataset_or_resume',release_sha256=a.release_sha,phase_sha256=sha(phase),exposure_us=2000,gain='Gain_X4',wait_ms=240,maximum_frames=11,source_TRAIN_indices=[293,294],new_TEST_frames=0,original_guard_unchanged=True,source_sha256=sha(__file__),protected_original_files=protected))
    lock=HERE/'runs/hardware/.t18_sdk_owner.lock';rows=[];started=time.perf_counter();stop_reason='fixed_sequence_complete';black_white=0
    with lock.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),out=str(a.out),kind='bounded_white_frame_diagnostic')))
    try:
        with Bench(a) as bench:
            save(a.out/'camera_before_controller_changes.json',bench.hw.before);bench.stage(phase)
            def frame(name):
                assert len(rows)<11 and time.perf_counter()-started<120
                settings={node:bench.hw.camera.get(node) for node in ('ExposureTime','Gain','TestPattern','AcquisitionFrameRate','SyncMode','PixelFormat')}
                assert abs(float(settings['ExposureTime'])-2000)<1 and settings['Gain']=='Gain_X4' and settings['TestPattern']=='Normal'
                raw,meta=bench.capture(paths[name]);assert raw.shape==(1080,1920)
                crop=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                tag=f'{len(rows):02d}_{name}';Image.fromarray(raw).save(a.out/(tag+'_raw.png'));Image.fromarray(crop).save(a.out/(tag+'.png'))
                row=dict(index=len(rows),pattern=name,time_UTC=datetime.now(timezone.utc).isoformat(),camera_current_readback=settings,raw=statistics(raw),crop=statistics(crop),frame_metadata=meta,raw_file=tag+'_raw.png',crop_file=tag+'.png',amplitude_BMP_sha256=sha(paths[name]),phase_sha256=sha(phase))
                rows.append(row);save(a.out/(tag+'.json'),row);save(a.out/'progress.json',dict(state='diagnosing',frames=len(rows),latest=row));print(json.dumps({k:row[k] for k in ('index','pattern','time_UTC','raw','crop','camera_current_readback')}),flush=True)
                return row
            for name in plans():
                row=frame(name);white=row['crop']['saturation_fraction']>.95
                black_white=black_white+1 if name=='black' and white else 0
                if black_white>=2:stop_reason='persistent_white_with_zero_input';break
                if name!='black' and row['crop']['saturation_fraction']>.15:
                    frame('black');stop_reason='illuminated_guard_failed_then_single_black_diagnostic';break
            save(a.out/'camera_before_exit.json',{node:bench.hw.camera.get(node) for node in ('ExposureTime','Gain','TestPattern','AcquisitionFrameRate','SyncMode','PixelFormat')})
        assert all(sha(path)==digest for path,digest in protected.items())
        save(a.out/'report.json',dict(state='complete',scope='diagnosis_only_not_resume_permission',frames=len(rows),stop_reason=stop_reason,rows=rows,sdk_released=True,source_sha256=sha(__file__),release_sha256=a.release_sha,original_files_unchanged=True,elapsed_s=time.perf_counter()-started,formal_dataset_modified=False,exposure_or_LUT_changed=False))
    except BaseException as error:
        save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),frames=len(rows),automatic_retry=False));raise
    finally:lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    for name in ('session','out','release','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+name,type=Path)
    p.add_argument('--release-sha');a=p.parse_args();selftest()
    if a.selftest:print('Bounded diagnostic sequence/statistics selftest passed; SDK not imported')
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        diagnose(a)
