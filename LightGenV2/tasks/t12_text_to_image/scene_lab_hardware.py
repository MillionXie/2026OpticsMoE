"""T12 session-only native calibration. No model, training, TEST or SDK on import.

The site mapping below is explicit and tested, not a universal camera mapping.
Device LUT/exposure/gain and the user's corner file are never rewritten.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import traceback
import numpy as np
from PIL import Image
from .scene_lab_protocol import sha, put, array_sha


def native(code, kind, inverse=None):
    a=np.asarray(code)
    if a.shape!=(478,478) or a.dtype!=np.uint8:raise ValueError('Canonical 478 uint8 only')
    if kind not in ('amplitude','phase'):raise ValueError('Unknown panel')
    if inverse is not None:
        lut=np.asarray(inverse)
        if kind!='amplitude' or lut.shape!=(256,) or lut[0]!=0 or lut[-1]!=255 or (np.diff(lut)<0).any():raise ValueError('Invalid field inverse')
        a=lut[a].astype(np.uint8)
    # 478*17um/8um=1015.75 physical pixels, rounded to the site's 1016.
    if kind=='phase':a=np.flipud(np.fliplr(a))
    active=np.asarray(Image.fromarray(a).resize((1016,1016),Image.Resampling.NEAREST))
    h=1080 if kind=='amplitude' else 1200
    result=np.zeros((h,1920),np.uint8);result[(h-1016)//2:(h+1016)//2,452:1468]=active
    return result if kind=='amplitude' else 255-result


def corners(path):
    rows={k:[float(x),float(y)]for k,x,y in re.findall(r'(TL|TR|BL|BR)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',Path(path).read_text())}
    if set(rows)!={'TL','TR','BR','BL'}:raise ValueError('Four explicit user corners required')
    return np.float32([rows[k]for k in ('TL','TR','BR','BL')])


def orientations(a):
    return dict(identity=a,flip_h=np.fliplr(a),flip_v=np.flipud(a),rotate180=np.flipud(np.fliplr(a)),
                transpose=a.T,transpose_flip_h=np.fliplr(a.T),transpose_flip_v=np.flipud(a.T),transpose_rotate180=np.flipud(np.fliplr(a.T)))


def centroid(a,b):
    w=(a.astype(float)-b.astype(float)).clip(0);peak=np.percentile(w,99.7)
    if peak<50:raise ValueError('No reliable native-count spot response')
    w=np.where(w>=.25*peak,w,0);y,x=np.indices(w.shape)
    return np.array([(w*x).sum()/w.sum(),(w*y).sum()/w.sum()])


def direction(images,background):
    scores={};points={}
    for name,bg in orientations(background).items():
        cc=np.stack([centroid(orientations(v)[name],bg)for v in images]);dx=cc[1]-cc[0];dy=cc[2]-cc[0]
        scores[name]=float(dx[0]/np.linalg.norm(dx)+dy[1]/np.linalg.norm(dy));points[name]=cc
    order=sorted(scores,key=scores.get,reverse=True);name=order[0];cc=points[name]
    if scores[name]<1.8 or scores[name]-scores[order[1]]<.5 or np.linalg.norm(cc[3]-cc[0])>8:raise ValueError('Ambiguous/stale orientation')
    return name,dict(scores=scores,centroids=cc.tolist(),repeat_error=float(np.linalg.norm(cc[3]-cc[0])))


def field_inverse(grays,means):
    # Pool-adjacent-violators monotone response; fitted only on calibration flats.
    blocks=[]
    for value in means:
        blocks.append([float(value),1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            b,a=blocks.pop(),blocks.pop();n=a[1]+b[1];blocks.append([(a[0]*a[1]+b[0]*b[1])/n,n])
    v=np.concatenate([np.full(n,x)for x,n in blocks])
    if v[-1]-v[0]<100:raise ValueError('Insufficient camera dynamic range')
    response=((v-v[0])/(v[-1]-v[0])).clip(0,1);u,first=np.unique(response,return_index=True)
    drive=np.rint(np.interp((np.arange(256)/255.)**2,u,np.asarray(grays)[first])).astype(int)
    drive[0]=0;drive[-1]=255
    return drive.tolist(),response.tolist()


class Bench:
    def __init__(self,config,out):self.c=config;self.out=Path(out)
    def __enter__(self):
        from experiments.hardware_sdk.devices import build_slm
        from experiments.hardware_sdk.drivers.tucam_camera import TucamCamera
        from LightGenV2.hardware_common.shs.phase_hdmi import PhaseHDMI
        c=self.c;self.stack=ExitStack()
        try:
            for path,digest in c['fixed_files'].items():
                if sha(path)!=digest:raise ValueError(('Changed device identity',path))
            m=json.loads(Path(c['machine_config']).read_text(encoding='utf-8-sig'));m['amplitude_slm']['sdk_path']=c['amplitude_sdk']
            self.phase=self.stack.enter_context(PhaseHDMI(c['phase_sdk'],c['phase_lut'],settle_s=.8,pixel_format='rgba'))
            self.slm=self.stack.enter_context(build_slm(m['amplitude_slm'],Path(c['machine_config']).parent))
            self.camera=self.stack.enter_context(TucamCamera(c['camera_sdk'],timeout_ms=10000,auto_exposure=None,exposure_us=None,analog_gain=None,
                warmup_frames=3,discard_frames_after_display=6,saved_frame_size_wh=None,saved_frame_resize_mode='none',saved_frame_bit_depth=None))
            self.info=self.camera.device_info()
            if self.info['camera_count']!=1 or self.info['auto_exposure_enabled']!=0 or abs(self.info['Exposure']-c['exposure_us'])>2 or self.info['analog_gain']!=c['gain']:raise ValueError('Camera settings changed')
            put(self.out/'devices.json',dict(camera=self.info,amplitude=self.slm.device_info(),phase=self.phase.info))
            return self
        except BaseException:self.stack.close();raise
    def __exit__(self,*args):return self.stack.__exit__(*args)
    def capture(self,tag,amp,phase):
        dest=self.out/tag;dest.mkdir(exist_ok=False)
        ap=dest/'amplitude.bmp';pp=dest/'phase.bmp'
        Image.fromarray(native(amp,'amplitude')).save(ap);Image.fromarray(native(phase,'phase')).save(pp)
        phase_receipt=self.phase.show(pp)
        self.slm.preload_files([ap]);self.slm.display_file(ap);time.sleep(self.c['wait_ms']/1000)
        rawpath=dest/'native.npy';self.camera.capture(rawpath);raw=np.load(rawpath,allow_pickle=False)
        if raw.shape!=(2048,2048) or raw.dtype!=np.uint16:raise ValueError('Unexpected native camera format')
        info=self.camera.device_info()
        # A reserved vendor frame index is never promoted to a unique frame ID.
        row=dict(capture_UTC=datetime.now(timezone.utc).isoformat(),raw_file_sha256=sha(rawpath),amplitude_file_sha256=sha(ap),phase_file_sha256=sha(pp),
                 camera=info,phase=phase_receipt,wait_ms=self.c['wait_ms'],p99=float(np.percentile(raw,99)),saturation_fraction=float((raw==65535).mean()))
        if abs(info['Exposure']-self.c['exposure_us'])>2 or info['analog_gain']!=self.c['gain'] or info['auto_exposure_enabled']!=0:raise ValueError('Camera settings drift')
        put(dest/'receipt.json',row)
        Image.fromarray(raw).save(dest/'native.png')
        return raw,row


def calibrate(c,out):
    import cv2
    out=Path(out);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[3]
    lock=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    others=[root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']
    if any(p.exists()for p in others):raise RuntimeError('Another device owner exists')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(out))).encode());os.close(fd)
    warp=cv2.getPerspectiveTransform(corners(c['coordinates']),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    put(out/'protocol.json',dict(config=c,source_sha256=sha(__file__),TEST_used=False,model_inference=False,phase_lut_scope='linear voltage, not verified linear phase'))
    rows=[];zero=np.zeros((478,478),np.uint8)
    try:
        with Bench(c,out) as bench:
            def snap(tag,amp,phase=zero,orientation='identity'):
                raw,receipt=bench.capture(tag,amp,phase)
                crop=cv2.warpPerspective(raw.astype(np.float32),warp,(478,478),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                crop=orientations(crop)[orientation].copy();np.save(out/tag/'canonical.npy',crop,allow_pickle=False)
                row=dict(tag=tag,mean=float(crop.mean()),p99=float(np.percentile(crop,99)),saturation_fraction=float((crop>=65534).mean()))
                rows.append(row);print(json.dumps(row),flush=True)
                if row['saturation_fraction']>.001:raise ValueError('Calibration saturation; no automatic exposure edit')
                return crop
            bg=snap('dark',zero);white=snap('white',np.full_like(zero,255))
            if np.percentile(white-bg,99)<1000:raise ValueError('Low dynamic range; stop before model pilot')
            positions=((150,150),(330,150),(150,330),(150,150));images=[]
            for k,(x,y) in enumerate(positions):
                a=zero.copy();a[y-22:y+22,x-22:x+22]=255;images.append(snap('axis_'+str(k),a))
            orientation,proof=direction(images,bg)
            put(out/'orientation.json',dict(selected=orientation,**proof))
            geometry=[]
            for k,(x,y) in enumerate(((100,100),(370,100),(100,370),(370,370),(240,240))):
                a=zero.copy();a[y-22:y+22,x-22:x+22]=255;v=snap('geometry_'+str(k),a,orientation=orientation)
                point=centroid(v,orientations(bg)[orientation]);err=float(np.linalg.norm(point-[x-.5,y-.5]));geometry.append(dict(expected=[x-.5,y-.5],observed=point.tolist(),error_px=err))
            put(out/'geometry.json',dict(rows=geometry,max_error_px=max(r['error_px']for r in geometry),coordinates_changed=False))
            if max(r['error_px']for r in geometry)>8:raise ValueError('User corner mapping failed independent spots; do not silently refit coordinates')
            center=zero.copy();center[217:261,217:261]=255
            base=snap('phase_flat',center,orientation=orientation);origin=centroid(base,orientations(bg)[orientation]);phase_rows=[]
            yy,xx=np.indices(zero.shape)
            for name,axis,sign in [('x_plus',xx,1),('x_minus',xx,-1),('y_plus',yy,1),('y_minus',yy,-1)]:
                phase=np.floor(np.remainder(sign*axis*32/478,1)*256).astype(np.uint8)
                v=snap('phase_'+name,center,phase,orientation);point=centroid(v,orientations(bg)[orientation]);phase_rows.append(dict(name=name,centroid=point.tolist(),delta=(point-origin).tolist()))
            put(out/'phase_response.json',dict(rows=phase_rows,native_transform='hv_inverse',device_LUT_changed=False))
            for pos,neg,axis in [(0,1,0),(2,3,1)]:
                delta=np.asarray(phase_rows[pos]['delta'])-np.asarray(phase_rows[neg]['delta'])
                if delta[axis]<6 or abs(delta[1-axis])>abs(delta[axis])*.5:raise ValueError('Phase sign/axis response disagrees with total phase; no pilot admission')
            grays=list(range(0,256,16))+[255];means=[]
            for g in grays:means.append(float(snap('response_'+str(g),np.full_like(zero,g),orientation=orientation).mean()))
            drive,response=field_inverse(grays,means);checks=[]
            for g in (32,64,96,128,160,192,224):
                actual=float(snap('heldout_'+str(g),np.full_like(zero,drive[g]),orientation=orientation).mean());error=(actual-means[0])/(means[-1]-means[0])-(g/255.)**2;checks.append(dict(desired=g,error=error))
            if max(abs(v['error'])for v in checks)>.08:raise ValueError('Held-out field inverse response failed')
            put(out/'amplitude_inverse.json',dict(drive=drive,fit_grays=grays,fit_means=means,response=response,heldout=checks,device_LUT_changed=False))
            dark=orientations(bg)[orientation][None].astype(np.float32);gain=np.full_like(dark,1/(means[-1]-means[0]),dtype=np.float32)
            np.save(out/'dark.npy',dark,allow_pickle=False);np.save(out/'gain_map.npy',gain,allow_pickle=False)
        put(out/'report.json',dict(passed=True,orientation=orientation,frames=len(rows),SDK_released=True,dark_sha256=array_sha(dark),gain_map_sha256=array_sha(gain),new_T12_session=True,TEST_used=False,actual_model_TRAIN_CCD=0,rows=rows))
    except BaseException as error:
        put(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),rows=rows,automatic_retry=False));raise
    finally:
        if json.loads(lock.read_text())['pid']!=os.getpid():raise RuntimeError('Lock ownership changed')
        lock.unlink();put(out/'release.json',dict(own_lock_released=True,process_pid=os.getpid(),closed_context=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    calibrate(json.loads(a.config.read_text()),a.out)


if __name__=='__main__':main()
