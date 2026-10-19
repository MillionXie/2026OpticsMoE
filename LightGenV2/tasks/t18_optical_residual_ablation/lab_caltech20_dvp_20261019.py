"""Small-camera session admission. No changed model, TEST use or device LUT fit.

Known calibration patterns only. A new camera/geometry creates a new session;
neither high-speed-camera CCDs nor its response curve is an admissible input.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
from lab_caltech10_20261015 import corners,native,phase_native,save,sha
from amplitude_response import inverse_field

LOCK=HERE/'runs/hardware/.t18_sdk_owner.lock'


def variants(a):
    return dict(identity=a,flip_h=np.fliplr(a),flip_v=np.flipud(a),
                rot180=np.rot90(a,2),transpose=a.T,rot90=np.rot90(a),
                rot270=np.rot90(a,3),anti_transpose=np.rot90(a.T,2))


def correlation(a,b):
    x=np.asarray(a,dtype=float).ravel();y=np.asarray(b,dtype=float).ravel()
    x=x-x.mean();y=y-y.mean();d=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/d) if d else 0.


def orient(a,name):return variants(a)[name].copy()


def validate_frame(raw,meta,cfg):
    assert raw.shape==tuple(cfg['sensor_shape']) and raw.dtype==np.uint8
    assert all(not meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
    assert meta['format']==0 and meta['bits']==0


class Bench:
    """Reuse validated native SLM drivers, replace only the camera interface."""
    def __init__(self,cfg,out):self.cfg=cfg;self.out=Path(out);self.last_id=-1
    def __enter__(self):
        from experiments.hardware_sdk.devices import build_slm
        from LightGenV2.hardware_common.dvp_legacy import Camera
        from LightGenV2.hardware_common.shs.phase_hdmi import PhaseHDMI
        c=self.cfg;self.stack=ExitStack()
        try:
            assert sha(c['camera_dll'])==c['camera_dll_SHA256']
            assert sha(c['coordinates'])==c['coordinates_SHA256']
            m=json.loads(Path(c['machine_config']).read_text(encoding='utf-8-sig'))
            m['amplitude_slm']['sdk_path']=c['amplitude_sdk']
            self.phase=self.stack.enter_context(PhaseHDMI(c['phase_sdk'],c['phase_lut'],settle_s=.8,pixel_format='rgba'))
            self.slm=self.stack.enter_context(build_slm(m['amplitude_slm'],Path(c['machine_config']).parent))
            self.camera=self.stack.enter_context(Camera(c['camera_dll']))
            self.original=self.camera.settings();self.stack.callback(self.restore)
            self.settings=self.camera.settings(exposure=c['exposure_us'],gain=1.)
            assert abs(self.settings['gain']-1)<1e-6
            save(self.out/'devices.json',dict(camera=self.settings,original=self.original,SLM=self.slm.device_info(),phase=self.phase.info,camera_DLL_SHA256=sha(c['camera_dll'])))
            return self
        except BaseException:self.stack.close();raise
    def restore(self):self.camera.settings(exposure=self.original['exposure_us'],gain=self.original['gain'])
    def __exit__(self,*args):return self.stack.__exit__(*args)
    def stage(self,path):
        self.phase.show(path)
        for _ in range(self.cfg['discard_frames']):self.read(check_settings=False)
    def read(self,check_settings=True):
        raw,meta=self.camera.capture();validate_frame(raw,meta,self.cfg)
        assert meta['frame_id']>self.last_id,'Nonmonotonic camera frame ID'
        self.last_id=meta['frame_id']
        if check_settings:assert abs(meta['exposure_us']-self.settings['exposure_us'])<max(2,self.settings['exposure_us']*.01)
        assert abs(meta['gain']-1)<1e-6
        return raw,meta
    def capture(self,path,frames=None,wait_ms=None):
        start=time.perf_counter();self.slm.preload_files([Path(path)]);self.slm.display_file(Path(path));visible=time.perf_counter()
        time.sleep((self.cfg['wait_ms'] if wait_ms is None else wait_ms)/1000)
        history=[]
        count=frames or self.cfg['discard_frames']+1
        for k in range(count):
            raw,meta=self.read(check_settings=k==count-1);history.append(meta)
        meta=dict(history[-1],frame_history=history,capture_total_ms=(time.perf_counter()-start)*1000,visible_to_capture_ms=(time.perf_counter()-visible)*1000,wait_ms=self.cfg['wait_ms'] if wait_ms is None else wait_ms,input_SHA256=sha(path))
        return raw,meta


def calibration(a):
    import cv2
    cfg=json.loads(a.profile.read_text());assert not a.out.exists();a.out.mkdir(parents=True)
    assert cfg['scope']=='calibration_only_no_TEST'
    pts=np.float32(corners(Path(cfg['coordinates'])));h,w=cfg['sensor_shape']
    assert np.all(pts[:,0]>=0) and np.all(pts[:,0]<w) and np.all(pts[:,1]>=0) and np.all(pts[:,1]<h)
    warp=cv2.getPerspectiveTransform(pts,np.float32([[0,0],[477,0],[477,477],[0,477]]))
    def crop(raw):return cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    with LOCK.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),out=str(a.out),camera='DVP',kind='calibration_only')))
    rows=[]
    def pattern(bench,tag,gray,orientation='identity',frames=None):
        path=a.out/(tag+'.bmp');Image.fromarray(native(gray,'amplitude')).save(path)
        raw,meta=bench.capture(path,frames=frames);v=orient(crop(raw),orientation)
        Image.fromarray(raw).save(a.out/(tag+'_raw.png'));Image.fromarray(v).save(a.out/(tag+'_crop.png'))
        row=dict(tag=tag,mean=float(v.mean()),p99=float(np.percentile(v,99)),saturation_fraction=float((v==255).mean()),maximum=int(v.max()),metadata=meta,raw_SHA256=sha(a.out/(tag+'_raw.png')))
        rows.append(row);save(a.out/(tag+'.json'),row);save(a.out/'progress.json',dict(state='calibrating',patterns=len(rows),latest=row));print(json.dumps(row),flush=True)
        return v,row
    try:
        with Bench(cfg,a.out) as bench:
            flat=a.out/'flat_phase.bmp';Image.fromarray(phase_native(np.zeros((478,478),np.uint8))).save(flat);bench.stage(flat)
            zero=np.zeros((478,478),np.uint8);white=np.full_like(zero,255);ladder=[]
            for exposure in cfg['exposure_probe_us']:
                bench.settings=bench.camera.settings(exposure=exposure,gain=1.)
                for _ in range(cfg['discard_frames']):bench.read(check_settings=False)
                _,dark=pattern(bench,f'exposure_{exposure}_dark',zero)
                _,bright=pattern(bench,f'exposure_{exposure}_white',white)
                ladder.append(dict(requested=exposure,readback=bench.settings.copy(),dark=dark['mean'],white=bright['mean'],p99=bright['p99'],saturation=bright['saturation_fraction']))
            usable=[v for v in ladder if v['p99']>=40 and v['p99']<=235 and v['saturation']<.001 and v['white']-v['dark']>=10]
            save(a.out/'exposure_ladder.json',ladder)
            assert usable,'No unclipped adequately illuminated exposure in predeclared range; stop and diagnose'
            selected=usable[-1];bench.settings=bench.camera.settings(exposure=selected['readback']['exposure_us'],gain=1.)
            for _ in range(cfg['discard_frames']):bench.read(check_settings=False)
            marker=np.full_like(zero,24);marker[35:215,35:215]=210;marker[35:215,263:443]=90;marker[263:443,35:215]=150;marker[290:390,290:390]=255
            image,_=pattern(bench,'orientation',marker)
            scores={k:correlation(v.astype(float),marker.astype(float)**2) for k,v in variants(image).items()};ranking=sorted(scores,key=scores.get,reverse=True);direction=ranking[0]
            save(a.out/'orientation.json',dict(scores=scores,selected=direction,method='known asymmetric TRAIN-independent calibration pattern, not TEST accuracy'))
            assert scores[direction]>=.4 and scores[direction]-scores[ranking[1]]>=.1,'Ambiguous orientation; stop rather than choose by TEST'
            # Repeat toggles; every sampled image must match the new, not previous, pattern.
            timing=[]
            for k,gray in enumerate((marker,np.fliplr(marker).copy(),marker,np.fliplr(marker).copy())):
                v,r=pattern(bench,f'timing_{k}',gray,direction);same=correlation(v,gray.astype(float)**2);other=correlation(v,np.fliplr(gray).astype(float)**2)
                timing.append(dict(same_pattern_PCC=same,previous_pattern_PCC=other,capture_total_ms=r['metadata']['capture_total_ms'],frame_ids=[m['frame_id'] for m in r['metadata']['frame_history']]))
                assert same>=.4 and same>other+.1,'Stale/inverted image detected; do not admit timing'
            grays=list(range(0,256,16))+[255];means=[]
            for g in grays:
                _,r=pattern(bench,f'response_{g:03d}',np.full_like(zero,g),direction);assert r['saturation_fraction']<.001;means.append(r['mean'])
            drive,response=inverse_field(grays,means);checks=[]
            for desired in (32,64,96,128,160,192,224):
                _,r=pattern(bench,f'heldout_{desired}',np.full_like(zero,drive[desired]),direction);measured=(r['mean']-means[0])/(means[-1]-means[0]);checks.append(dict(desired=desired,measured=measured,target=(desired/255)**2,error=measured-(desired/255)**2));assert r['saturation_fraction']<.001
            errors=np.array([v['error'] for v in checks]);passed=bool(np.max(np.abs(errors))<=.08 and np.sqrt(np.mean(errors**2))<=.04)
            save(a.out/'response_lut.json',dict(passed=passed,drive=drive,fit_grays=grays,fit_means=means,response=response,heldout=checks,validation_rmse=float(np.sqrt(np.mean(errors**2))),validation_max_error=float(np.max(np.abs(errors))),scope='new DVP session only; device LUT unchanged',dark_subtracted_in_network=False,exposure_us=bench.settings['exposure_us'],gain=1.,coordinates_SHA256=sha(cfg['coordinates'])))
            assert passed,'Held-out radiometry failed; preserve evidence, no automatic retry'
            accepted=dict(cfg,exposure_us=bench.settings['exposure_us'],orientation=direction,response_lut=str((a.out/'response_lut.json').resolve()),response_lut_SHA256=sha(a.out/'response_lut.json'),calibration_source_SHA256=sha(__file__),phase_LUT_SHA256=sha(cfg['phase_lut']))
        save(a.out/'admitted_hardware.json',accepted)
        save(a.out/'report.json',dict(state='complete',passed=True,SDK_released=True,patterns=len(rows),timing=timing,hardware_SHA256=sha(a.out/'admitted_hardware.json'),model_inference_run=False,TEST_used=False))
    except BaseException as ex:
        save(a.out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),patterns=len(rows),automatic_retry=False,all_evidence_preserved=True));raise
    finally:LOCK.unlink()


def selftest():
    a=np.arange(25).reshape(5,5)
    for name,v in variants(a).items():assert np.isclose(correlation(v,v),1)
    markers=np.zeros((478,478),np.uint8);markers[:100,:100]=255
    p=native(markers,'amplitude');assert p.shape==(1080,1920) and not p[:32].any() and not p[:, :452].any()
    ph=phase_native(markers);assert np.array_equal(np.flipud(np.fliplr(255-ph[92:1108,452:1468])),np.asarray(Image.fromarray(markers).resize((1016,1016),Image.Resampling.NEAREST)))
    lut,r=inverse_field([0,64,128,192,255],np.square(np.array([0,64,128,192,255])/255)*100+3);assert len(lut)==256 and lut[0]==0 and lut[-1]==255
    print(json.dumps(dict(passed=True,SDK_opened=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['selftest','calibrate'],required=True);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='selftest':selftest()
    else:calibration(a)
