"""Fixed-session T12 capture: native frames, no model/FFT or parameter changes.

check saves two non-model dark/white frames. stage consumes prepared native
Qwen tickets and accepts only actual CCDs, one complete cohort at a time.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from .scene_lab_hardware import Bench, native, orientations
from .scene_lab_protocol import Ledger, STAGES, array_sha, identity, put, sha
from .scene_lab_sdk_canary import signal_pair


def drive_codes(amplitude, phase, inverse):
    a,p=np.asarray(amplitude),np.asarray(phase)
    if a.shape!=(1,478,478)or p.shape!=a.shape or a.dtype!=np.uint8 or p.dtype!=np.uint8:
        raise ValueError('Expected one canonical uint8 amplitude/total-phase pair')
    return native(a[0],'amplitude',inverse),native(p[0],'phase')


def mapping(config):
    path=Path(config['derived_mapping'])
    if sha(path)!=config['derived_mapping_sha256']:raise ValueError('Mapping changed')
    value=json.loads(path.read_text())
    if not value['calibration_approved']or not value['no_TEST_or_model_frames']:raise ValueError('Unreviewed map')
    if value['original_coordinate_SHA256']!=sha(config['coordinates']):raise ValueError('Original TXT changed')
    warp=np.asarray(value['canonical_warp'],np.float64)
    if warp.shape!=(3,3)or not np.isfinite(warp).all():raise ValueError('Invalid fixed map')
    return warp


def live_settings(camera,config):
    if hasattr(camera,'live_settings'):actual=camera.live_settings()
    else:
        module=camera._module
        actual=dict(exposure_us=camera._get_property(module.TUCAM_IDPROP.TUIDP_EXPOSURETM)*1000,
                gain=camera._get_property(module.TUCAM_IDPROP.TUIDP_GLOBALGAIN),
                auto=camera._get_capability(module.TUCAM_IDCAPA.TUIDC_ATEXPOSURE))
    if abs(actual['exposure_us']-config['exposure_us'])>2 or actual['gain']!=config['gain']or actual['auto']!=0:
        raise ValueError('Live camera setting mismatch; no automatic edits')
    return actual


def validate_session(ledger,config,calibration,dark,gain):
    s=ledger.session
    if s['scope']!='reviewed_physical' or s['orientation']!='identity':raise ValueError('Reviewed physical identity session required')
    expected=dict(exposure_us=config['exposure_us'],gain=config['gain'],wait_ms=config['wait_ms'],
        mapping_sha256=config['derived_mapping_sha256'],phase_lut_sha256=sha(config['phase_lut']),
        amplitude_inverse_sha256=sha(calibration/'amplitude_inverse.json'),dark_sha256=array_sha(dark),gain_map_sha256=array_sha(gain))
    if any(s[k]!=v for k,v in expected.items()):raise ValueError('Session/settings/calibration mismatch')
    if s['amplitude_native_shape']!=[1080,1920]or s['phase_native_shape']!=[1200,1920]:raise ValueError('Panel identity mismatch')


def await_payload(folder,run,seconds,pump):
    deadline=time.monotonic()+seconds
    while not(folder/'input.ready').exists():
        abort=run/'input_abort.json'
        if abort.exists():raise RuntimeError(('Input producer/transfer stopped',json.loads(abort.read_text())))
        if time.monotonic()>=deadline:raise TimeoutError('Timed out waiting for the next complete native input')
        pump();time.sleep(.1)


def run(config,calibration,out,*,ledger_root=None,stage=None,camera_process=False,input_wait_seconds=0):
    import cv2
    from LightGenV2.hardware_common.shs.phase_owner import message_pump
    out=Path(out);calibration=Path(calibration);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    own=root/'LightGenV2/tasks/t12_text_to_image/runs/hardware/.t12_scene_device_owner.lock'
    other=[root/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/.t18_sdk_owner.lock',root/'LightGenV2/hardware_common/shs/results/phase_sdk_owner.lock']
    if any(p.exists()for p in other):raise RuntimeError('Another device owner exists')
    fd=os.open(own,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(pid=os.getpid(),out=str(out))).encode());os.close(fd)
    rows=[];accepted=[]
    try:
        warp=mapping(config);inverse=json.loads((calibration/'amplitude_inverse.json').read_text())['drive']
        admission=json.loads((calibration/'report.json').read_text())
        if not admission['passed']or admission['orientation']!='identity':raise ValueError('Spatial/radiometric calibration not admitted')
        ledger=None;dark=np.load(calibration/'dark.npy',allow_pickle=False);gain=np.load(calibration/'gain_map.npy',allow_pickle=False)
        if ledger_root is not None:
            c=json.loads((Path(ledger_root)/'contract.json').read_text())
            ledger=Ledger(ledger_root,c['session'],c['models'],c['requests']);validate_session(ledger,config,calibration,dark,gain);admission=ledger.admit_stage(stage)
        put(out/'protocol.json',dict(config=config,source_SHA256=sha(__file__),stage=stage,TRAIN_only=True,TEST_used=False,
            parameter_changes=False,camera_process_isolated=camera_process,phase_linearity_certified=False,model_CCD_requested=ledger is not None,calibration_report_SHA256=sha(calibration/'report.json')))
        factory=None
        if camera_process:
            from .scene_lab_camera_process import CameraProcess
            factory=lambda path,**kwargs:CameraProcess(path,allowed_root=out,**kwargs)
        with Bench(config,out,camera_factory=factory)as bench:
            pump=message_pump()
            def capture(tag,ag,pg):
                folder=out/tag;folder.mkdir();ap=folder/'amplitude.bmp';pp=folder/'phase.bmp'
                amp,phase=drive_codes(ag,pg,inverse);Image.fromarray(amp).save(ap);Image.fromarray(phase).save(pp)
                phase_receipt=bench.phase.show(pp,pump=pump);bench.slm.preload_files([ap]);bench.slm.display_file(ap)
                until=time.monotonic()+config['wait_ms']/1000
                while time.monotonic()<until:pump();time.sleep(.01)
                rawfile=folder/'native.npy';bench.camera.capture(rawfile);raw=np.load(rawfile,allow_pickle=False)
                if raw.shape!=(2048,2048)or raw.dtype!=np.uint16:raise ValueError('Native camera format changed')
                canonical=cv2.warpPerspective(raw.astype(np.float32),warp,(478,478),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[None]
                np.save(folder/'canonical.npy',canonical,allow_pickle=False)
                row=dict(tag=tag,capture_UTC=datetime.now(timezone.utc).isoformat(),native_SHA256=sha(rawfile),canonical_raw_SHA256=array_sha(canonical),
                    amplitude_BMP_SHA256=sha(ap),phase_BMP_SHA256=sha(pp),actual_camera=live_settings(bench.camera,config),phase=phase_receipt,
                    mean=float(raw.mean()),p99=float(np.percentile(raw,99)),canonical_mean=float(canonical.mean()),saturation_fraction=float((raw==65535).mean()))
                put(folder/'receipt.json',row);rows.append(row)
                print(json.dumps({k:v for k,v in row.items()if k!='phase'}),flush=True)
                if row['saturation_fraction']>.001:raise ValueError('Saturation guard; no automatic exposure edits')
                return raw,canonical,row
            zero=np.zeros((1,478,478),np.uint8)
            d,dc,_=capture('canary_dark',zero,zero);w,wc,_=capture('canary_white',np.full_like(zero,255),zero)
            signal=signal_pair(d,w);signal['canonical_p99_delta']=float(np.percentile(wc-dc,99));put(out/'signal_check.json',signal)
            if not signal['passed']or signal['canonical_p99_delta']<1000:raise ValueError('New native/canonical white response absent; model capture not admitted')
            if ledger is not None:
                for request in ledger.requests:
                    folder=ledger.folder(stage,request)
                    if(folder/'receipt.json').exists():
                        ledger.receipt(stage,request)
                        if not(folder/'capture.ready').exists():(folder/'capture.ready').touch(exist_ok=False)
                        continue
                    if input_wait_seconds:await_payload(folder,out.parent,input_wait_seconds,pump)
                    ticket=json.loads((folder/'ticket.json').read_text())
                    with np.load(folder/'logical_fields.npz',allow_pickle=False)as z:
                        ag=z['amplitude_code'].copy();pg=z['phase_code'].copy()
                        if array_sha(ag)!=ticket['logical_amplitude_code_sha256']or array_sha(pg)!=ticket['logical_phase_code_sha256']:raise ValueError('Pending codes changed')
                    raw,canonical,row=capture(request['sample_id'],ag,pg)
                    metadata=dict(ticket_sha256=identity(ticket),session_sha256=identity(ledger.session),stage=stage,request_sha256=identity(request),
                        canonical_orientation=ledger.session['orientation'],exposure_us=config['exposure_us'],gain=config['gain'],wait_ms=config['wait_ms'],
                        frame_provenance='physical_linear_camera',native_frame_sha256=row['native_SHA256'],canonical_raw_sha256=array_sha(canonical),capture_UTC=row['capture_UTC'])
                    receipt=ledger.accept(stage,request,canonical,metadata,dark,gain,admission=admission);accepted.append(dict(sample_id=request['sample_id'],receipt_sha256=identity(receipt)))
                    (folder/'capture.ready').touch(exist_ok=False)
        put(out/'report.json',dict(complete=True,accepted=accepted,model_TRAIN_CCD=len(accepted),canary_frames=2,raw_frames=len(rows),SDK_released=True,TEST_used=False,phase_linearity_certified=False,physical_SIM_EXP_quality_accepted=False))
    except BaseException as error:
        put(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),frames_saved=len(rows),model_TRAIN_CCD=len(accepted),automatic_retry=False));raise
    finally:
        assert json.loads(own.read_text())['pid']==os.getpid();own.unlink();put(out/'release.json',dict(pid=os.getpid(),own_lock_released=True,closed_context=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ledger',type=Path);p.add_argument('--stage',choices=STAGES);p.add_argument('--camera-process',action='store_true');p.add_argument('--input-wait-seconds',type=float,default=0);a=p.parse_args()
    if bool(a.ledger)!=bool(a.stage):raise ValueError('ledger and stage required together')
    if not 0<=a.input_wait_seconds<=600:raise ValueError('Bounded input wait required')
    run(json.loads(a.config.read_text()),a.calibration,a.out,ledger_root=a.ledger,stage=a.stage,camera_process=a.camera_process,input_wait_seconds=a.input_wait_seconds)


if __name__=='__main__':main()
