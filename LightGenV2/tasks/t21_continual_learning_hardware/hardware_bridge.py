"""T21 native1016 intensity boundary and lossless BMP deployment.

Never changes coordinates, device LUT, exposure or model structure. Admission
uses calibration patterns only. Pilot uses deterministic TRAIN only. A phase
captures all optical layers before any terminal readout fitting is permitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from .model import build_model
from .train_native import sha,put,tensor_sha


def hardware_identity(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def legacy_session():
    # Published, read-only device session; old experiment runs are not restarted.
    previous=Path(__file__).parent.parent/'t18_optical_residual_ablation'
    if str(previous) not in sys.path:sys.path.insert(0,str(previous))
    from lab_caltech20_tucam_session_20261022 import Bench,LOCK,spot_centroid
    from lab_caltech10_20261015 import corners
    return Bench,LOCK,spot_centroid,corners


def native_image(gray,phase=False):
    gray=np.asarray(gray)
    if gray.shape!=(1016,1016) or gray.dtype!=np.uint8:raise ValueError('Native uint8 only; no implicit resize')
    height=1200 if phase else 1080
    panel=np.zeros((height,1920),np.uint8)
    value=np.flipud(np.fliplr(gray)) if phase else gray
    panel[(height-1016)//2:(height+1016)//2,452:1468]=value
    return 255-panel if phase else panel


def crop_native(raw,warp):
    if raw.shape!=(2048,2048) or raw.dtype!=np.uint16:raise ValueError('Changed sensor contract')
    # One intensity warp, fixed full scale; no uint8 CCD conversion or peak fit.
    value=cv2.warpPerspective(raw.astype(np.float32),warp,(1016,1016),
        flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)/65535.
    if not np.isfinite(value).all() or (value<0).any():raise ValueError('Invalid native CCD')
    return value


def hardware(cfg):
    Bench,LOCK,centroid,corners=legacy_session()
    for key,expected in (('coordinates','coordinates_SHA256'),('phase_lut','phase_LUT_SHA256'),
                         ('response_lut','response_lut_SHA256')):
        if sha(cfg[key])!=cfg[expected]:raise ValueError('Changed hardware identity: '+key)
    if cfg['orientation']!='identity' or cfg['phase_transform']!='hv_inverse':
        raise ValueError('Only independently admitted identity CCD / hv_inverse phase supported')
    response=json.loads(Path(cfg['response_lut']).read_text());drive=np.asarray(response['drive'],np.uint8)
    if not response['passed'] or drive.shape!=(256,) or drive[0]!=0:
        raise ValueError('Unadmitted amplitude response inverse')
    # Coordinates describe aperture boundaries, not edge-pixel centers.
    points=np.float32(corners(Path(cfg['coordinates'])))
    warp=cv2.getPerspectiveTransform(points,np.float32([[-.5,-.5],[1015.5,-.5],[1015.5,1015.5],[-.5,1015.5]]))
    return Bench,LOCK,centroid,drive,warp


def signal(value,mask=None,dark=None):
    readout=value if mask is None else value[mask]
    row=dict(mean=float(value.mean()),p99_gray=float(np.percentile(value,99)*255),
                maximum=float(value.max()),saturation_fraction=float((value>=1).mean()),
                readout_p99_counts=float(np.percentile(readout,99)*65535),
                readout_p10_counts=float(np.percentile(readout,10)*65535))
    if dark is not None:
        if dark.shape!=value.shape or not np.isfinite(dark).all():raise ValueError('Fixed admitted dark frame shape/values changed')
        delta=value-dark
        row['fixed_dark_difference_p99_counts']=float(np.percentile(delta if mask is None else delta[mask],99)*65535)
        row['network_background_subtraction']=False
    return row


def guard(row,noise=None,router=False,policy='legacy',minimum_snr=10):
    if policy not in ('legacy','pointwise_dark_p99'):raise ValueError('Unknown signal guard policy')
    if minimum_snr not in (7.5,8,10) or (minimum_snr!=10 and policy!='pointwise_dark_p99'):raise ValueError('Only explicitly scoped pointwise SNR7.5/8 or unchanged10')
    if policy=='pointwise_dark_p99':
        if noise is None or 'fixed_dark_difference_p99_counts' not in row:raise ValueError('Pointwise guard requires fixed admitted dark diagnostics')
        margin=max(64.,minimum_snr*noise['read_noise_upper_counts'])
        dynamic_margin=max(64.,10*noise['read_noise_upper_counts'])
        metrics=[row['fixed_dark_difference_p99_counts'],row['readout_p99_counts'],row['readout_p10_counts'],row['saturation_fraction'],noise['read_noise_upper_counts']]
        dark=(not np.isfinite(metrics).all() or noise['read_noise_upper_counts']<=0 or
              row['fixed_dark_difference_p99_counts']<margin or row['readout_p99_counts']-row['readout_p10_counts']<dynamic_margin)
    elif noise is None:dark=row['p99_gray']<15
    else:
        # Original logical8-bit p99=15 is not a bit-depth-invariant dark test.
        # This fixed TRAIN/control admission is frozen before any TEST capture.
        floor=noise['dark_router_ROI_p99_counts' if router else 'dark_global_p99_counts']
        margin=max(64.,10*noise['read_noise_upper_counts'])
        dark=(row['readout_p99_counts']<floor+margin or
              row['readout_p99_counts']-row['readout_p10_counts']<margin)
    if dark or row['saturation_fraction']>.15:
        raise RuntimeError('Pilot/capture dynamic-range stop; preserve evidence, no automatic exposure/retry: '+repr(row))


def signal_guard_policy(policy=None,exposure_us=None):
    """Legacy default; human-authorized pointwise dark SNR changes diagnostics only."""
    if policy is None:return 'legacy'
    expected={'mode','authorization','minimum_signal_counts','minimum_SNR','exposure_us','single_frame','network_background_subtraction'}
    if (set(policy)!=expected or policy['mode']!='pointwise_dark_p99' or not isinstance(policy['authorization'],str) or
        not policy['authorization'].strip() or policy['minimum_signal_counts']!=64 or policy['minimum_SNR']!=10 or
        policy['single_frame'] is not True or policy['network_background_subtraction'] is not False or
        policy['exposure_us']!=2000 or not isinstance(exposure_us,(int,float)) or not np.isfinite(exposure_us) or abs(exposure_us-2000)>.5):
        raise ValueError('Signal policy requires explicit fixed2ms single-frame SNR10/64 authorization; no network dark subtraction')
    return 'pointwise_dark_p99'


def stage_SNR_threshold(package,cfg,stage):
    """Exact human-authorized C/D cases, not a global lowered threshold."""
    if stage not in ('router','first','final'):raise ValueError('Unknown optical stage')
    d_scoped=package.get('D_SNR75_authorization')
    if d_scoped is not None:
        expected=json.loads(Path(__file__).with_name('config_LG_D_SNR75_20261026.json').read_text(encoding='utf-8'))
        if package.get('C_SNR8_authorization') is not None or d_scoped!=expected or package.get('architecture')!='moe' or package.get('stage')!='D' or package.get('checkpoint_sha256')!=expected['checkpoint_sha256'] or package.get('hardware_sha256')!=expected['hardware_sha256']:
            raise ValueError('SNR7.5 requires exact published LG D/PT/hardware permission')
        if signal_guard_policy(package.get('signal_guard_policy'),cfg.get('exposure_us'))!='pointwise_dark_p99' or not package.get('TRAIN_stage_signal_calibration'):
            raise ValueError('SNR7.5 retains fixed2ms TRAIN/control pointwise admission')
        return expected['minimum_SNR_by_optical_stage'][stage]
    scoped=package.get('C_SNR8_authorization')
    if scoped is None:return 10
    expected=json.loads(Path(__file__).with_name('config_LG_C_SNR8_20261025.json').read_text(encoding='utf-8'))
    if scoped!=expected or package.get('architecture')!='moe' or package.get('stage')!='C' or package.get('checkpoint_sha256')!=expected['checkpoint_sha256'] or package.get('hardware_sha256')!=expected['hardware_sha256']:
        raise ValueError('SNR8 requires exact published LG C/PT/hardware permission')
    if signal_guard_policy(package.get('signal_guard_policy'),cfg.get('exposure_us'))!='pointwise_dark_p99' or not package.get('TRAIN_stage_signal_calibration'):
        raise ValueError('SNR8 retains fixed2ms TRAIN/control pointwise admission')
    return expected['minimum_SNR_by_optical_stage'][stage]


def admitted_guard_dark(layer,noise,mode):
    if mode=='legacy':return None
    p=layer/'TRAIN_signal_admission/dark_ccd.npy'
    if noise is None or noise.get('TEST_used') is not False or sha(p)!=noise.get('dark_CCD_sha256'):raise ValueError('Fixed same-stage TRAIN/control dark SHA missing/changed')
    value=np.load(p)
    if value.shape!=(1016,1016) or not np.isfinite(value).all():raise ValueError('Invalid fixed dark frame')
    return value


def repeat_policy(policy=None,exposure_us=None):
    """Default enforcement; explicit human-approved single-frame 2 ms exception."""
    if policy is None:return 'enforce'
    if (set(policy)!={'mode','authorization','exposure_us','single_frame'} or
        policy['mode']!='record_only' or not isinstance(policy['authorization'],str) or
        not policy['authorization'].strip() or policy['exposure_us']!=2000 or
        policy['single_frame'] is not True or not isinstance(exposure_us,(int,float)) or
        not np.isfinite(exposure_us) or abs(exposure_us-2000)>.5):
        raise ValueError('Repeat exception requires explicit fixed single-frame 2 ms authorization')
    return 'record_only'


def admission_checks(snr,delta99,correlation,repeat_drift,marker_change,noise,saturation,mode='enforce',minimum_snr=10):
    if mode not in ('enforce','record_only'):raise ValueError('Unknown repeat policy')
    if minimum_snr not in (7.5,8,10):raise ValueError('Only authorized SNR7.5/8 or original10')
    checks=dict(finite_metrics=bool(np.isfinite([snr,delta99,correlation,repeat_drift,marker_change,noise,saturation]).all()),
        SNR=bool(snr>=minimum_snr),signal_margin=bool(delta99>=64),
        repeat_correlation=bool(correlation>=.95),consumer_drift=bool(repeat_drift<=.05),
        marker_update=bool(marker_change>=3*noise),saturation=bool(saturation<=.15))
    enforced={k:v for k,v in checks.items() if k!='repeat_correlation' or mode=='enforce'}
    return bool(all(enforced.values())),checks


def stage_signal_admission(bench,folder,amplitudes,drive,warp,mask,stage,hardware_sha,model,repeat_mode='enforce',minimum_snr=10):
    """Fixed dark/A/A/B/A TRAIN/control test before any EVAL in this layer.

    Repeat fluctuations include shot/display noise, not just camera read noise;
    they therefore must be measured with this stage's phase and intensity.
    This does not subtract a background from network frames or fit any labels.
    """
    folder.mkdir();values={};rows={}
    patterns={'dark':np.zeros((1016,1016),np.uint8),'A1':amplitudes[0],
              'A2':amplitudes[0],'B':amplitudes[1],'A3':amplitudes[0]}
    marker=np.zeros((1016,1016),np.uint8);marker[145:215,145:215]=128
    patterns.update(marker=marker,A4=amplitudes[0])
    if mask is None:mask=np.ones((1016,1016),bool)
    for name,gray in patterns.items():
        f=folder/(name+'.bmp');Image.fromarray(native_image(drive[gray])).save(f)
        raw,metadata=bench.capture(f);np.save(folder/(name+'_raw.npy'),raw)
        value=crop_native(raw,warp);np.save(folder/(name+'_ccd.npy'),value);values[name]=value*65535
        rows[name]=dict(signal=signal(value,mask),metadata=metadata,raw_sha256=sha(folder/(name+'_raw.npy')))
    dark=values['dark'];A1=values['A1'];A2=values['A2'];A3=values['A3'];B=values['B']
    noise=max(1.,float(np.std((A1-A2)[mask])/np.sqrt(2)))
    delta99=float(np.percentile((A1-dark)[mask],99));snr=delta99/noise
    correlation=float(np.corrcoef((A1-dark)[mask],(A3-dark)[mask])[0,1])
    raw_repeat_drift=float(np.linalg.norm((A1-A3)[mask])/max(1.,np.linalg.norm(A1[mask])))
    # Check exactly what the unchanged next operation consumes. Raw uniform
    # illumination drift is already cancelled by routing/OEO/readout norms.
    def consumer(value):
        tensor=torch.from_numpy(value/65535)[None]
        if stage=='router':return model.route_from_ccd(tensor)[0].numpy()
        if stage=='first':return model.normalize(torch.nn.functional.softsign(tensor/tensor.mean((-2,-1),keepdim=True)))[0].numpy().ravel()
        return model.features_from_ccd(tensor)[0].numpy()
    ca,cb=consumer(A1),consumer(values['A4'])
    repeat_drift=float(np.abs(ca-cb).sum() if stage=='router' else np.linalg.norm(ca-cb)/max(1e-20,np.linalg.norm(ca)))
    changed=float(np.sqrt(np.mean((B-A1)[mask]**2)))
    marker_change=float(np.sqrt(np.mean((values['marker']-A1)[mask]**2)))
    # Semantic records may legitimately have similar optical outputs; they are
    # not a timing standard. An independent fixed calibration marker verifies
    # actual input updates, followed by return to exactly the same TRAIN A.
    passed,checks=admission_checks(snr,delta99,correlation,repeat_drift,marker_change,noise,
        max(r['signal']['saturation_fraction'] for r in rows.values()),repeat_mode,minimum_snr)
    receipt=dict(passed=bool(passed),stage=stage,hardware_payload_sha256=hardware_sha,
        minimum_SNR=minimum_snr,legacy_SNR10_passed=bool(snr>=10),legacy_SNR8_passed=bool(snr>=8),legacy_SNR_threshold=10,
        admission_checks=checks,repeat_correlation_policy=repeat_mode,
        repeat_correlation_threshold=.95,repeat_correlation_enforced=repeat_mode=='enforce',
        read_noise_upper_counts=noise,noise_scope='repeat shot/display/camera fluctuations at this stage, NOT pure read noise',
        dark_global_p99_counts=float(np.percentile(dark,99)),dark_router_ROI_p99_counts=float(np.percentile(dark[mask],99)),
        dark_CCD_sha256=sha(folder/'dark_ccd.npy'),
        SNR=snr,signal_ROI_delta_p99_counts=delta99,repeat_correlation=correlation,
        repeat_consumer_drift=repeat_drift,repeat_consumer='q L1' if stage=='router' else 'actual OEO/readout relative L2',
        raw_repeat_relative_L2=raw_repeat_drift,changed_input_RMS_counts=changed,rows=rows,
        semantic_input_difference_is_diagnostic_not_a_gate=True,marker_change_RMS_counts=marker_change,
        TEST_used=False,labels_used=False,exposure_changed=False,network_background_subtraction=False,
        patterns=['dark','first_TRAIN','first_TRAIN','last_TRAIN','first_TRAIN','fixed_independent_marker','first_TRAIN'])
    put(folder/'report.json',receipt)
    if not passed:raise RuntimeError('Stage-specific TRAIN/control SNR admission failed; no EVAL or automatic retry')
    return receipt


def native_admission(cfg,out):
    Bench,LOCK,centroid,drive,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK owned by another session')
    out.mkdir(parents=True,exist_ok=False);put(LOCK,dict(pid=os.getpid(),task='T21',scope='native_admission',out=str(out)))
    rows=[];positions=((180,180),(800,180),(180,800),(800,800),(510,510),(180,180))
    try:
        with Bench(cfg,out) as bench:
            phase=out/'zero_phase.bmp';Image.fromarray(native_image(np.zeros((1016,1016),np.uint8),True)).save(phase);bench.stage(phase)
            def snap(name,gray):
                p=out/(name+'.bmp');Image.fromarray(native_image(drive[gray])).save(p)
                raw,metadata=bench.capture(p);np.save(out/(name+'_raw.npy'),raw)
                value=crop_native(raw,warp);np.save(out/(name+'_ccd.npy'),value)
                return value,dict(metadata=metadata,raw_sha256=sha(out/(name+'_raw.npy')),signal=signal(value))
            dark,bg=snap('background',np.zeros((1016,1016),np.uint8))
            # A fixed scale is used only for centroid diagnostics, never network frames.
            for k,(x,y) in enumerate(positions):
                gray=np.zeros((1016,1016),np.uint8);gray[y-35:y+35,x-35:x+35]=255
                value,row=snap('spot_'+str(k),gray)
                point=centroid(value*255,dark*255);error=float(np.linalg.norm(point-[x-.5,y-.5]))
                rows.append(dict(expected=[x-.5,y-.5],observed=point.tolist(),error_native_px=error,**row))
                put(out/'progress.json',dict(patterns=len(rows),latest=rows[-1]))
            repeat=float(np.linalg.norm(np.array(rows[0]['observed'])-rows[-1]['observed']))
            passed=max(r['error_native_px'] for r in rows)<=17 and repeat<=8
            put(out/'report.json',dict(passed=passed,rows=rows,background=bg,repeat_error_native_px=repeat,
                active_pixels=1016,sensor_full_scale=65535,network_background_subtraction=False,
                TEST_used=False,fit_points_reused=False,warp=warp.tolist(),SDK_released_after_return=True,
                hardware_payload_sha256=hardware_identity(cfg)))
            if not passed:raise RuntimeError('Independent native1016 spatial/temporal admission failed')
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK ownership changed')
        LOCK.unlink()


def load_checkpoint(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('PT SHA mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(checkpoint['architecture']);model.load_state_dict(checkpoint['model'],strict=True)
    model.configure_stage(checkpoint['stage']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,checkpoint


def read_source(path,expected_sha,split,indices):
    if sha(path)!=expected_sha:raise ValueError('Source encoded input SHA mismatch')
    manifest=json.loads(Path(path).with_name('manifest.json').read_text())
    if manifest['split']!=split or manifest['inputs_sha256']!=expected_sha:raise ValueError('Changed split identity')
    with np.load(path,allow_pickle=False) as a:
        return dict(x=a['amplitudes'][indices].copy(),y=a['labels'][indices].copy(),
                    source_indices=a['source_indices'][indices].copy(),manifest=manifest)


def acquisition_order(entries,order):
    """Optional fixed full permutation, never changes IDs, labels or split."""
    if order is None:return entries
    if (not isinstance(order,list) or any(type(i) is not int for i in order) or
        len(order)!=len(entries) or sorted(order)!=list(range(len(entries)))):
        raise ValueError('Acquisition order must be an exact full permutation')
    return [entries[i] for i in order]


def TRAIN_control_indices(entries,references=None):
    indices=[i for i,e in enumerate(entries) if e['split']=='train']
    if len(indices)<2:raise ValueError('Two TRAIN-only stage controls required')
    if references is None:return [indices[0],indices[-1]]
    if len(references)!=2:raise ValueError('Exactly two fixed TRAIN control references required')
    selected=[]
    for ref in references:
        if ref.get('split','train')!='train':raise ValueError('TEST control forbidden')
        matches=[i for i in indices if entries[i]['task']==ref['task'] and entries[i]['local_index']==ref['local_index']]
        if len(matches)!=1:raise ValueError('Ambiguous or missing TRAIN control identity')
        selected.append(matches[0])
    if selected[0]==selected[1]:raise ValueError('Distinct TRAIN control identities required')
    return selected


@torch.no_grad()
def capture(package,cfg,out,pilot=False):
    repeat_mode=repeat_policy(package.get('repeat_correlation_policy'),cfg.get('exposure_us'))
    guard_mode=signal_guard_policy(package.get('signal_guard_policy'),cfg.get('exposure_us'))
    thresholds={s:stage_SNR_threshold(package,cfg,s) for s in ('router','first','final')}
    if guard_mode!='legacy' and not package.get('TRAIN_stage_signal_calibration',False):raise ValueError('Pointwise guard requires fixed per-stage TRAIN/control admission')
    Bench,LOCK,_,drive,warp=hardware(cfg)
    admission=json.loads(Path(package['native_admission']).read_text())
    if not admission['passed'] or sha(package['native_admission'])!=package['native_admission_sha256']:
        raise ValueError('Native admission missing or changed')
    if admission['hardware_payload_sha256']!=hardware_identity(cfg):raise ValueError('Admission uses another hardware session')
    noise=None
    if 'signal_admission' in package:
        if sha(package['signal_admission'])!=package['signal_admission_sha256']:raise ValueError('Signal admission SHA changed')
        noise=json.loads(Path(package['signal_admission']).read_text())
        if not noise['passed'] or noise['hardware_payload_sha256']!=hardware_identity(cfg) or noise['TEST_used']:
            raise ValueError('No same-session TRAIN/control signal admission')
    if LOCK.exists():raise RuntimeError('SDK already owned')
    model,checkpoint=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    out.mkdir(parents=True,exist_ok=False)
    put(out/'protocol.json',dict(package=package,hardware=cfg,hardware_sha256=package['hardware_sha256'],
        pilot=pilot,source_sha256=sha(__file__),signal_guard_policy=guard_mode,interlayer_tuning=False,TEST_gradients=False))
    entries=[]
    for spec in package['datasets']:
        if pilot and spec['split']!='train':continue
        ix=np.asarray(spec['indices'],int)
        if pilot:ix=ix[np.linspace(0,len(ix)-1,min(24,len(ix)),dtype=int)]
        data=read_source(spec['path'],spec['sha256'],spec['split'],ix)
        for k in range(len(ix)):
            entries.append(dict(task=spec['task'],split=spec['split'],local_index=int(ix[k]),
                source_index=int(data['source_indices'][k]),label=int(data['y'][k]),amplitude=data['x'][k],
                manifest_sha256=data['manifest']['identity_sha256']))
    if not pilot:entries=acquisition_order(entries,package.get('entry_order'))
    order=model.STAGES if model.architecture=='moe' else model.STAGES[1:]
    reused=package.get('premeasured_stages',{})
    if set(reused)-set(order):raise ValueError('Unknown premeasured boundary')
    if reused and list(reused)!=list(order[:len(reused)]):raise ValueError('Only a contiguous immutable prefix may be reused')
    stage_paths={s:Path(reused[s]) if s in reused else out/s for s in order}
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='pilot' if pilot else 'capture',out=str(out)))
    started=time.time();rows=[];frozen={k:tensor_sha(v) for k,v in model.state_dict().items()}
    try:
        with Bench(cfg,out) as bench:
            for stage in order:
                layer=stage_paths[stage]
                if stage not in reused:layer.mkdir()
                phase_file=layer/'phase.bmp';phase_sha=None
                mask=None
                if stage=='router':
                    mask=np.zeros((1016,1016),bool)
                    for index in model.active_indices[:int(model.active_count)].tolist():
                        y,x=model.router_centers[index];y-=model.border;x-=model.border;h=model.router_side//2
                        mask[y-h:y+h,x-h:x+h]=True
                stage_noise=noise
                if stage in reused:
                    prior_noise=layer/'TRAIN_signal_admission/report.json'
                    if prior_noise.exists():
                        stage_noise=json.loads(prior_noise.read_text())
                        if not stage_noise['passed'] or stage_noise['stage']!=stage or stage_noise['hardware_payload_sha256']!=hardware_identity(cfg):
                            raise ValueError('Reused prefix signal admission is not valid for this stage/session')
                    stage_dark=admitted_guard_dark(layer,stage_noise,guard_mode)
                    for i,entry in enumerate(entries):
                        key=f'{i:05d}';folder=layer/key;r=json.loads((folder/'receipt.json').read_text())
                        for field in ('task','split','local_index','source_index','manifest_sha256'):
                            if r[field]!=entry[field]:raise ValueError('Reused prefix source identity mismatch: '+field)
                        if r['checkpoint_sha256']!=package['checkpoint_sha256'] or r['hardware_sha256']!=package['hardware_sha256']:
                            raise ValueError('Reused prefix PT/hardware mismatch')
                        for filename,field in (('ccd.npy','ccd_sha256'),('raw.npy','raw_sha256'),('input.bmp','input_BMP_sha256')):
                            if sha(folder/filename)!=r[field]:raise ValueError('Reused prefix data SHA mismatch')
                        if sha(phase_file)!=r['phase_BMP_sha256']:raise ValueError('Reused phase BMP SHA mismatch')
                        prefix={s:torch.from_numpy(np.load(stage_paths[s]/key/'ccd.npy'))[None] for s in order[:order.index(stage)]}
                        amp,phase,_=model.payload(torch.from_numpy(entry['amplitude'])[None],stage,prefix)
                        gray=np.rint(amp[0].numpy()*255).astype(np.uint8)
                        if hashlib.sha256(gray.tobytes()).hexdigest()!=r['input_canonical_sha256']:raise ValueError('Reused prefix canonical input mismatch')
                        value=np.load(folder/'ccd.npy');guard(signal(value,mask,stage_dark),stage_noise,router=stage=='router',policy=guard_mode,minimum_snr=thresholds[stage])
                        rows.append(dict(r,reused_from=str(folder)))
                    put(out/(stage+'_prefix_audit.json'),dict(passed=True,records=len(entries),source=str(layer),new_frames=0,
                        same_PT=True,same_hardware=True,all_source_ids_and_raw_CCD_BMP_SHA_verified=True))
                    continue
                if package.get('TRAIN_stage_signal_calibration',False):
                    train_indices=TRAIN_control_indices(entries,package.get('signal_control_TRAIN_records'))
                    examples=[]
                    for index in train_indices:
                        key=f'{index:05d}'
                        prefix={s:torch.from_numpy(np.load(stage_paths[s]/key/'ccd.npy'))[None] for s in order[:order.index(stage)]}
                        amplitude,phase,_=model.payload(torch.from_numpy(entries[index]['amplitude'])[None],stage,prefix)
                        examples.append(np.rint(amplitude[0].numpy()*255).astype(np.uint8))
                        candidate=native_image(np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8),True)
                        digest=hashlib.sha256(candidate.tobytes()).hexdigest()
                        if phase_sha is None:
                            Image.fromarray(candidate).save(phase_file);bench.stage(phase_file);phase_sha=digest
                        elif digest!=phase_sha:raise ValueError('Calibration phase changed across inputs')
                    stage_noise=stage_signal_admission(bench,layer/'TRAIN_signal_admission',examples,drive,warp,mask,
                        stage,hardware_identity(cfg),model,repeat_mode,thresholds[stage])
                stage_dark=admitted_guard_dark(layer,stage_noise,guard_mode)
                for i,entry in enumerate(entries):
                    frame_start=time.monotonic()
                    key=f'{i:05d}';folder=layer/key;folder.mkdir()
                    prefix={s:torch.from_numpy(np.load(stage_paths[s]/key/'ccd.npy'))[None] for s in order[:order.index(stage)]}
                    x=torch.from_numpy(entry['amplitude'])[None]
                    amplitude,phase,_=model.payload(x,stage,prefix)
                    gray=np.rint(amplitude[0].numpy()*255).astype(np.uint8)
                    phase_gray=np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8)
                    candidate=native_image(phase_gray,True)
                    digest=hashlib.sha256(candidate.tobytes()).hexdigest()
                    if phase_sha is None:
                        Image.fromarray(candidate).save(phase_file);bench.stage(phase_file);phase_sha=digest
                    elif digest!=phase_sha:raise ValueError('Input-dependent phase violates frozen layer contract')
                    amplitude_file=folder/'input.bmp';Image.fromarray(native_image(drive[gray])).save(amplitude_file)
                    raw,metadata=bench.capture(amplitude_file);np.save(folder/'raw.npy',raw)
                    value=crop_native(raw,warp);np.save(folder/'ccd.npy',value);quality=signal(value,mask,stage_dark)
                    if guard_mode=='pointwise_dark_p99':
                        quality.update(minimum_SNR=thresholds[stage],fixed_stage_noise_counts=stage_noise['read_noise_upper_counts'],
                            SNR_vs_fixed_stage_noise=quality['fixed_dark_difference_p99_counts']/stage_noise['read_noise_upper_counts'],
                            legacy_SNR8_passed=bool(quality['fixed_dark_difference_p99_counts']>=max(64.,8*stage_noise['read_noise_upper_counts'])),
                            legacy_SNR10_passed=bool(quality['fixed_dark_difference_p99_counts']>=max(64.,10*stage_noise['read_noise_upper_counts'])))
                    row={k:v for k,v in entry.items() if k!='amplitude'}
                    row.update(index=i,stage=stage,checkpoint_sha256=package['checkpoint_sha256'],
                        input_canonical_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
                        input_BMP_sha256=sha(amplitude_file),phase_BMP_sha256=sha(phase_file),
                        prefix_CCD_sha256={s:sha(stage_paths[s]/key/'ccd.npy') for s in prefix},
                        raw_sha256=sha(folder/'raw.npy'),ccd_sha256=sha(folder/'ccd.npy'),
                        signal=quality,metadata=metadata,hardware_sha256=package['hardware_sha256'])
                    row['workflow_frame_ms']=(time.monotonic()-frame_start)*1000
                    put(folder/'receipt.json',row);rows.append(row)
                    put(out/'progress.json',dict(stage=stage,stage_frames=i+1,stage_total=len(entries),
                        total_frames=len(rows),expected_frames=len(entries)*len(order),latest=row,updated=time.time()))
                    guard(quality,stage_noise,router=stage=='router',policy=guard_mode,minimum_snr=thresholds[stage])
            features=[]
            for i in range(len(entries)):
                value=torch.from_numpy(np.load(stage_paths['final']/f'{i:05d}'/'ccd.npy'))[None]
                features.append(model.features_from_ccd(value)[0].numpy())
            np.savez_compressed(out/'features.npz',features=np.stack(features),labels=[r['label'] for r in entries],
                tasks=[r['task'] for r in entries],splits=[r['split'] for r in entries],source_indices=[r['source_index'] for r in entries])
        if {k:tensor_sha(v) for k,v in model.state_dict().items()}!=frozen:raise RuntimeError('Frozen deployed PT changed')
        new_frames=sum('reused_from' not in r for r in rows);elapsed=time.time()-started
        put(out/'report.json',dict(passed=True,pilot=pilot,records=len(entries),frames=len(rows),
            minimum_SNR_by_optical_stage=thresholds,C_SNR8_authorization=package.get('C_SNR8_authorization'),
            D_SNR75_authorization=package.get('D_SNR75_authorization'),
            frames_below_legacy_SNR8=sum(r.get('signal',{}).get('legacy_SNR8_passed') is False for r in rows),
            frames_below_legacy_SNR10=sum(r.get('signal',{}).get('legacy_SNR10_passed') is False for r in rows),
            stages=list(order),features_sha256=sha(out/'features.npz'),checkpoint_sha256=package['checkpoint_sha256'],
            new_network_frames=sum('reused_from' not in r for r in rows),reused_network_frames=sum('reused_from' in r for r in rows),
            extra_TRAIN_control_frames=7*(len(order)-len(reused)) if package.get('TRAIN_stage_signal_calibration',False) else 0,
            seconds=elapsed,seconds_per_frame=elapsed/new_frames if new_frames else None,
            timing_scope='current segment per NEW network frame, including calibration/IO; prior prefix acquisition time excluded',
            steady_workflow_frame_ms=[r['workflow_frame_ms'] for r in rows if 'reused_from' not in r],
            TRAIN_records=sum(r['split']=='train' for r in entries),EVAL_records=sum(r['split']=='test' for r in entries),
            SDK_released=True,interlayer_training=False,full_physical_prefix=True))
    except BaseException:
        put(out/'failure.json',dict(traceback=traceback.format_exc(),valid_frames=len(rows),
            automatic_retry=False,all_valid_and_bad_evidence_preserved=True));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK ownership changed')
        LOCK.unlink()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=('admit','pilot','capture'),required=True)
    p.add_argument('--hardware',type=Path,required=True);p.add_argument('--package',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);cfg=json.loads(a.hardware.read_text())
    if a.mode=='admit':native_admission(cfg,a.out)
    else:
        package=json.loads(a.package.read_text())
        if sha(a.hardware)!=package['hardware_sha256']:raise ValueError('Hardware JSON SHA mismatch')
        capture(package,cfg,a.out,pilot=a.mode=='pilot')


if __name__=='__main__':main()
