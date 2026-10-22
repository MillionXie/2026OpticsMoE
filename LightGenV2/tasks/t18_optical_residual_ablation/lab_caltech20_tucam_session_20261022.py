"""T18 TUCam session admission; native raw evidence, no TEST or model changes.

Reuse the Git-published camera backend. The old DVP hardware identity is never
overwritten. A fixed sensor-full-scale conversion defines logical uint8 CCD;
there is no per-image scaling, camera ROI, detector resize or optical LUT edit.
"""
import argparse,json,os,time,traceback
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from PIL import Image
from lab_caltech20_dvp_20261019 import LOCK,variants,correlation,orient
from lab_caltech10_20261015 import corners,native,phase_native,save,sha


def logical(raw,warp,full_scale,orientation='identity'):
    import cv2
    assert raw.ndim==2 and raw.dtype in (np.uint8,np.uint16)
    value=cv2.warpPerspective(raw.astype(np.float32),warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return orient(np.rint(value*255/full_scale).clip(0,255).astype(np.uint8),orientation)


def camera_for(cfg):
    from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
    assert sha(Path(cfg['sdk'])/'TUCam.py')==cfg['binding_SHA256']
    assert sha(Path(cfg['sdk'])/'lib/x64/TUCam.dll')==cfg['DLL_SHA256']
    return TucamCamera(Path(cfg['sdk']),timeout_ms=10000,auto_exposure=False,
                       exposure_us=cfg['exposure_us'],analog_gain=None,
                       warmup_frames=3,discard_frames_after_display=cfg['discard_frames'])


def spot_centroid(value,background):
    w=(value.astype(float)-background.astype(float)).clip(0);peak=np.percentile(w,99.7)
    assert peak>=5,'No reliable calibration spot response'
    w=np.where(w>=.25*peak,w,0);yy,xx=np.indices(w.shape);s=w.sum();assert s>0
    return np.array([(w*xx).sum()/s,(w*yy).sum()/s])


def spot_direction(images,background):
    """Independent X/Y isolated positions, robust to quadrant illumination."""
    scores={};points={}
    for name in variants(background):
        cc=np.stack([spot_centroid(variants(v)[name],variants(background)[name]) for v in images]);dx=cc[1]-cc[0];dy=cc[2]-cc[0]
        assert np.linalg.norm(dx)>60 and np.linalg.norm(dy)>60
        scores[name]=float(dx[0]/np.linalg.norm(dx)+dy[1]/np.linalg.norm(dy));points[name]=cc.tolist()
    order=sorted(scores,key=scores.get,reverse=True);best=order[0]
    assert scores[best]>=1.8 and scores[best]-scores[order[1]]>=.5,'Ambiguous independent X/Y spot axes'
    cc=np.array(points[best]);assert np.linalg.norm(cc[3]-cc[0])<8,'Spot repeat differs; no timing admission'
    assert np.linalg.norm(cc[3]-cc[2])>60,'Latest spot stale from previous pattern'
    return best,dict(scores=scores,selected=best,centroids=points[best],repeat_error=float(np.linalg.norm(cc[3]-cc[0])),coordinates_changed=False,method='independent isolated X/Y source positions and repeat; no global quadrant brightness fit')


def known_spots(bench,out,warp,full_scale):
    out=Path(out);out.mkdir(exist_ok=True);images=[];positions=((150,150),(330,150),(150,330),(150,150))
    def capture(tag,gray):
        path=out/(tag+'.bmp');Image.fromarray(native(gray,'amplitude')).save(path);raw,meta=bench.capture(path);crop=logical(raw,warp,full_scale)
        Image.fromarray(raw).save(out/(tag+'_raw.png'),compress_level=1);Image.fromarray(crop).save(out/(tag+'_crop.png'));return crop,dict(metadata=meta,raw_SHA256=sha(out/(tag+'_raw.png')))
    background,bg=capture('spot_background',np.zeros((478,478),np.uint8));rows=[]
    for k,(x,y) in enumerate(positions):
        gray=np.zeros((478,478),np.uint8);gray[y-22:y+22,x-22:x+22]=255;image,row=capture('spot_'+str(k),gray);images.append(image);rows.append(row)
    direction,proof=spot_direction(images,background);save(out/'report.json',dict(passed=True,**proof,rows=rows,background=bg,TEST_used=False,network_background_subtraction=False))
    return direction,rows


class Bench:
    def __init__(self,cfg,out):self.cfg=cfg;self.out=Path(out);self.last=-1
    def __enter__(self):
        from experiments.hardware_sdk.devices import build_slm
        from LightGenV2.hardware_common.shs.phase_hdmi import PhaseHDMI
        self.stack=ExitStack();c=self.cfg
        try:
            assert sha(c['coordinates'])==c['coordinates_SHA256'] and sha(c['phase_lut'])==c['phase_LUT_SHA256']
            m=json.loads(Path(c['machine_config']).read_text(encoding='utf-8-sig'));m['amplitude_slm']['sdk_path']=c['amplitude_sdk']
            self.phase=self.stack.enter_context(PhaseHDMI(c['phase_sdk'],c['phase_lut'],settle_s=.8,pixel_format='rgba'))
            self.slm=self.stack.enter_context(build_slm(m['amplitude_slm'],Path(c['machine_config']).parent))
            self.camera=camera_for(c);self.camera.open();self.stack.callback(self.camera.close)
            info=self.camera.device_info();assert info['auto_exposure_enabled']==0 and abs(info['Exposure']-c['exposure_us'])<max(2,c['exposure_us']*.01)
            self.settings=info;save(self.out/'devices.json',dict(camera=info,SLM=self.slm.device_info(),phase=self.phase.info))
            return self
        except BaseException:self.stack.close();raise
    def __exit__(self,*args):return self.stack.__exit__(*args)
    def stage(self,path):self.phase.show(path)
    def read(self):
        scratch=self.out/'native_frame_scratch.npy';self.camera.capture(scratch);raw=np.load(scratch);info=self.camera.device_info();meta=info['last_capture']
        assert raw.shape==(2048,2048) and raw.dtype==np.uint16
        # The vendor manual labels uiIndex RESERVED, and current runtime returns
        # 0. Never pretend a software counter is a hardware frame index. ucDepth
        # currently returns code2, not a usable numeric bit count. Preserve it;
        # declared full-scale is the uint16 storage contract, not fitted pixels.
        reserved_index=meta.pop('camera_frame_index');depth_code=meta.pop('sensor_bit_depth');self.last+=1
        exposure=self.camera._get_property(self.camera._module.TUCAM_IDPROP.TUIDP_EXPOSURETM)*1000
        assert abs(exposure-self.cfg['exposure_us'])<max(2,self.cfg['exposure_us']*.01)
        gain=self.camera._get_property(self.camera._module.TUCAM_IDPROP.TUIDP_GLOBALGAIN)
        assert gain==self.settings['analog_gain'] and self.camera._get_capability(self.camera._module.TUCAM_IDCAPA.TUIDC_ATEXPOSURE)==0
        assert self.cfg['sensor_full_scale']==65535
        return raw,dict(meta,frame_id=self.last,frame_id_source='host_successful_capture_counter_NOT_hardware',SDK_reserved_uiIndex=reserved_index,SDK_ucDepth_raw=depth_code,exposure_us=exposure,shape=list(raw.shape),sensor_full_scale=self.cfg['sensor_full_scale'],
                        camera_model=info['camera_model'],driver='tucam',auto_exposure=False,
                        gain=gain,SDK_flips_applied=False)
    def capture(self,path,wait_ms=None):
        start=time.monotonic();self.slm.preload_files([Path(path)]);self.slm.display_file(Path(path));visible=time.monotonic()
        wait=self.cfg['wait_ms'] if wait_ms is None else wait_ms;time.sleep(wait/1000)
        raw,meta=self.read();meta.update(wait_ms=wait,capture_total_ms=(time.monotonic()-start)*1000,
                                       visible_to_capture_ms=(time.monotonic()-visible)*1000,input_SHA256=sha(path),discard_frames=self.cfg['discard_frames'])
        return raw,meta


def probe(cfg,out):
    assert not out.exists() and not LOCK.exists();out.mkdir(parents=True)
    save(LOCK,dict(pid=os.getpid(),camera='tucam',kind='native_probe',out=str(out)))
    camera=camera_for(cfg)
    try:
        camera.open();rows=[]
        for k in range(3):
            path=out/f'native_{k}.npy';camera.capture(path);array=np.load(path);info=camera.device_info()
            rows.append(dict(shape=list(array.shape),dtype=str(array.dtype),minimum=int(array.min()),maximum=int(array.max()),p99=float(np.percentile(array,99)),SHA256=sha(path),metadata=info['last_capture']))
        save(out/'report.json',dict(passed=True,device_info=info,frames=rows,SDK_released_after_return=True,SLM_touched=False,TEST_used=False,source_SHA256=sha(__file__)))
    finally:
        camera.close();assert json.loads(LOCK.read_text())['pid']==os.getpid();LOCK.unlink()


def calibrate(cfg,out):
    import cv2
    from amplitude_response import inverse_field
    assert not out.exists() and not LOCK.exists();out.mkdir(parents=True)
    pts=np.float32(corners(Path(cfg['coordinates'])));assert (pts>=0).all() and (pts<2048).all()
    warp=cv2.getPerspectiveTransform(pts,np.float32([[0,0],[477,0],[477,477],[0,477]]))
    save(out/'protocol.json',dict(config=cfg,source_SHA256=sha(__file__),TEST_used=False,model_inference=False,new_hardware_session=True,phase_LUT_scope='linear voltage, not verified linear phase'))
    save(LOCK,dict(pid=os.getpid(),camera='tucam',kind='session_calibration',out=str(out)))
    rows=[];full_scale=None
    def snap(bench,tag,gray,direction='identity',wait=None):
        nonlocal full_scale
        path=out/(tag+'.bmp');Image.fromarray(native(gray,'amplitude')).save(path)
        raw,meta=bench.capture(path,wait_ms=wait)
        if full_scale is None:full_scale=meta['sensor_full_scale']
        assert full_scale==meta['sensor_full_scale']
        crop=logical(raw,warp,full_scale,direction);Image.fromarray(raw).save(out/(tag+'_raw.png'),compress_level=1);Image.fromarray(crop).save(out/(tag+'_crop.png'))
        row=dict(tag=tag,mean=float(crop.mean()),p99=float(np.percentile(crop,99)),saturation_fraction=float((crop==255).mean()),metadata=meta,raw_SHA256=sha(out/(tag+'_raw.png')))
        rows.append(row);save(out/(tag+'.json'),row);save(out/'progress.json',dict(state='calibrating',patterns=len(rows),latest=row));print(json.dumps(row),flush=True)
        return crop,row
    try:
        ladder=[]
        # A bounded sequence of camera connections is deliberate exposure
        # calibration, not automatic recovery or repeated acquisition retries.
        if cfg.get('reuse_exposure_ladder'):
            assert sha(cfg['reuse_exposure_ladder'])==cfg['reuse_exposure_ladder_SHA256'];ladder=json.loads(Path(cfg['reuse_exposure_ladder']).read_text());full_scale=65535
        for exposure in (() if ladder else cfg['exposure_probe_us']):
            case=out/f'exposure_{exposure}';case.mkdir();current={**cfg,'exposure_us':exposure}
            with Bench(current,case) as bench:
                flat=out/'flat_phase.bmp';Image.fromarray(phase_native(np.zeros((478,478),np.uint8))).save(flat);bench.stage(flat)
                _,dark=snap(bench,f'exposure_{exposure}_dark',np.zeros((478,478),np.uint8))
                _,white=snap(bench,f'exposure_{exposure}_white',np.full((478,478),255,np.uint8))
                ladder.append(dict(exposure_us=white['metadata']['exposure_us'],dark=dark['mean'],white=white['mean'],p99=white['p99'],saturation_fraction=white['saturation_fraction']))
            save(out/'exposure_ladder.json',ladder)
            # First adequate unclipped exposure; do not sweep into saturation.
            if 80<=white['p99']<=235 and white['saturation_fraction']<.001 and white['mean']-dark['mean']>=10:break
        usable=[v for v in ladder if 40<=v['p99']<=235 and v['saturation_fraction']<.001 and v['white']-v['dark']>=10]
        assert usable,'No usable exposure in bounded calibration; no automatic retry'
        selected=usable[-1];current={**cfg,'exposure_us':selected['exposure_us']};case=out/'selected';case.mkdir()
        with Bench(current,case) as bench:
            flat=out/'flat_phase.bmp';Image.fromarray(phase_native(np.zeros((478,478),np.uint8))).save(flat);bench.stage(flat)
            direction,timing=known_spots(bench,out/'direction_spots',warp,full_scale)
            save(out/'orientation.json',json.loads((out/'direction_spots/report.json').read_text()))
            grays=list(range(0,256,16))+[255];means=[]
            for g in grays:
                _,r=snap(bench,f'response_{g:03d}',np.full((478,478),g,np.uint8),direction);assert r['saturation_fraction']<.001;means.append(r['mean'])
            drive,response=inverse_field(grays,means);checks=[]
            for g in (32,64,96,128,160,192,224):
                _,r=snap(bench,f'heldout_{g}',np.full((478,478),drive[g],np.uint8),direction);actual=(r['mean']-means[0])/(means[-1]-means[0]);checks.append(dict(desired=g,measured=actual,error=actual-(g/255)**2));assert r['saturation_fraction']<.001
            errors=np.array([v['error'] for v in checks]);assert abs(errors).max()<=.08 and np.sqrt(np.mean(errors**2))<=.04,'Heldout amplitude response failed'
            save(out/'response_lut.json',dict(drive=drive,fit_grays=grays,fit_means=means,response=response,heldout=checks,passed=True,device_LUT_unchanged=True,network_background_subtraction=False))
            assert sha(cfg['phase_lut'])==cfg['phase_LUT_SHA256']
            accepted={**current,'sensor_shape':[2048,2048],'sensor_full_scale':full_scale,'orientation':direction,'response_lut':str((out/'response_lut.json').resolve()),'response_lut_SHA256':sha(out/'response_lut.json'),'camera_device_info':bench.settings}
        save(out/'admitted_hardware.json',accepted);save(out/'report.json',dict(passed=True,state='complete',SDK_released=True,patterns=len(rows),timing=timing,hardware_SHA256=sha(out/'admitted_hardware.json'),TEST_used=False))
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),patterns=len(rows),automatic_retry=False));raise
    finally:
        assert json.loads(LOCK.read_text())['pid']==os.getpid();LOCK.unlink()


def selftest():
    import cv2
    v=np.arange(478*478,dtype=np.uint32).reshape(478,478)%65536;v=v.astype(np.uint16);warp=np.eye(3,dtype=np.float32)
    result=logical(v,warp,65535);assert np.array_equal(result,np.rint(v.astype(np.float32)*255/65535).astype(np.uint8))
    assert np.array_equal(logical(v,warp,65535,'flip_v'),np.flipud(result))
    assert logical(np.zeros((478,478),np.uint16),warp,65535).max()==0
    bg=np.zeros((478,478),np.uint8);images=[]
    for x,y in ((150,150),(330,150),(150,330),(150,150)):
        a=bg.copy();a[y-22:y+22,x-22:x+22]=200;images.append(a)
    assert spot_direction(images,bg)[0]=='identity'
    assert spot_direction([np.flipud(v) for v in images],bg)[0]=='flip_v'
    print(json.dumps(dict(passed=True,SDK_opened=False,original_16bit_preserved=True,fixed_full_scale=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('probe','calibrate','selftest'),required=True);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='selftest':selftest()
    else:
        cfg=json.loads(a.profile.read_text());(probe if a.mode=='probe' else calibrate)(cfg,a.out)
