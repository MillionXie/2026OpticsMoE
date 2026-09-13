"""Bounded exposure replay of existing network BMPs, never changes a session.

Camera settings are restored on exit. Statistics use raw pixels in the measured
ROI polygon, BEFORE interpolation; small PNGs are diagnostic views only.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from slm_camera import Controller
from capture import snapshot
from phase_fingerprint import pcc
from guarded_workflow import read,write
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def image_stats(raw,mask):
    x=np.asarray(raw)[mask].astype(float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid ROI pixels')
    q=np.percentile(x,[1,50,95,99,99.9])
    return dict(mean=float(x.mean()),std=float(x.std()),p01=float(q[0]),p50=float(q[1]),
        p95=float(q[2]),p99=float(q[3]),p999=float(q[4]),maximum=float(x.max()),
        saturation_fraction=float((x>=255).mean()),near_clip_fraction=float((x>=250).mean()),
        low_pixel_fraction=float((x<=2).mean()),dynamic_range=float(q[3]-q[0]))

def recommend(rows):
    # Safe range is the INTERSECTION across every sampled input and repeat.
    # 0.1% clipping limit + p99.9<=245 leaves headroom; no max-only rejection.
    candidates=[]
    for e in sorted({r['exposure_us'] for r in rows}):
        group=[r for r in rows if r['exposure_us']==e];reasons=[]
        if max(r['stats']['saturation_fraction'] for r in group)>.001:reasons.append('clipping')
        if max(r['stats']['p999'] for r in group)>245:reasons.append('insufficient_headroom')
        if min(r['stats']['p99'] for r in group)<16:reasons.append('weak_signal')
        if min(r['stats']['dynamic_range'] for r in group)<8:reasons.append('weak_dynamic_range')
        if min(r.get('repeat_pcc',1) for r in group)<.97:reasons.append('unstable_repeat')
        candidates.append(dict(exposure_us=e,passed=not reasons,
            photometric_passed=not any(r!='unstable_repeat' for r in reasons),reasons=reasons,n=len(group)))
    good=[r['exposure_us'] for r in candidates if r['passed']]
    return dict(recommended_exposure_us=max(good) if good else None,candidates=candidates,
        scope='Sampled BMPs only; not full-data guarantee or photometric calibration',
        thresholds=dict(maximum_saturation_fraction=.001,maximum_p999=245,minimum_p99=16,minimum_dynamic_range=8))

def run(spec,c,out):
    source_config=json.loads(json.dumps(c));c=json.loads(json.dumps(c))
    wait=spec.get('capture_wait_ms')
    if wait is not None:
        if not np.isfinite(wait) or not 200<=wait<=1000:raise ValueError('Audit capture wait must be200..1000ms')
        c['settle_delay_ms']=float(wait)
    rows=spec['exposure_rows'];exposures=spec['exposures_us'];repeats=int(spec.get('repeats',2))
    if not 1<=len(rows)<=16 or not 1<=len(exposures)<=10 or not 2<=repeats<=4:raise ValueError('Bounded scan only')
    if any(not np.isfinite(e) or not 1<=e<1e6/c['camera']['frame_rate_hz'] for e in exposures):raise ValueError('Exposure outside frame period')
    if len(set(exposures))!=len(exposures) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate identity')
    if not 0<=spec.get('timing_repeats',0)<=10:raise ValueError('Bounded timing repeats only')
    original_wait=c['settle_delay_ms']
    for r in rows+[spec['probe']]:
        p=(ROOT/r['bmp']).resolve()
        if not p.is_relative_to(ROOT) or p.suffix.lower()!='.bmp' or sha(p)!=r['sha256']:raise ValueError('BMP path/hash mismatch')
    if spec['phase_receipt']['phase_sha256']!=spec['phase_sha256']:raise ValueError('Phase mismatch')
    points=[c['logical_corners_full_sensor_xy'][k] for k in ('top_left','top_right','bottom_right','bottom_left')]
    im=Image.new('1',(1920,1080));ImageDraw.Draw(im).polygon([tuple(v) for v in points],fill=1);mask=np.asarray(im,dtype=bool)
    bbox=im.getbbox();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',config=c,source_config=source_config,source_spec=spec,rows=[],probe=[],timing=[],
        input_scope='Frozen prior six-stage inputs; no regenerated downstream features and no retraining',
        raw_roi_before_warp=True,source_sha256=sha(__file__))
    def save():write(out/'report.json',report)
    save()
    try:
        with Controller(c) as hw:
            def set_exposure(e):
                hw.camera.stop();hw.camera.set('ExposureTime',e);hw.camera.start()
                hw.camera_settings=snapshot(hw.camera)
                actual=float(hw.camera_settings['ExposureTime']['value'])
                if abs(actual-e)>max(1.,e*.01):raise RuntimeError('Exposure readback mismatch')
                if hw.camera_settings['Gain']['value']!=c['camera']['gain']:raise RuntimeError('Gain drift')
                # Drain after each restart; does not add to per-image formal settle.
                start=time.perf_counter()
                while time.perf_counter()-start<.15:hw.camera.grab()
            def capture(bmp):
                raw,meta=hw.capture(ROOT/bmp)
                if raw.shape!=(1080,1920):raise ValueError('Unexpected sensor size')
                return raw,meta
            def preview(raw,name):
                Image.fromarray(raw).crop(bbox).resize((478,478),Image.Resampling.BOX).save(out/name)
            set_exposure(spec.get('probe_exposure_us',150))
            raw,meta=capture(spec['probe']['bmp']);before=raw[mask].astype(np.float32)
            report['probe'].append(dict(which='before',stats=image_stats(raw,mask),meta=meta))
            preview(raw,'probe_before.png');save()
            ps=report['probe'][0]['stats']
            if ps['p99']<16 or ps['std']<2 or ps['saturation_fraction']>.01:
                raise RuntimeError('Fixed checker probe has insufficient signal or clipping; no exposure scan: '+str(ps))
            rng=np.random.default_rng(20260913)
            for e in exposures:
                set_exposure(e);last={}
                for repeat in range(repeats):
                    for j in rng.permutation(len(rows)):
                        r=rows[j];raw,meta=capture(r['bmp']);v=raw[mask].astype(np.float32)
                        row=dict(id=r['id'],bmp_sha256=r['sha256'],exposure_us=e,repeat=repeat,
                            stats=image_stats(raw,mask),meta=meta)
                        if r['id'] in last:row['repeat_pcc']=pcc(last[r['id']],v)
                        last[r['id']]=v
                        report['rows'].append(row)
                        if repeat==0:preview(raw,f"e{e:g}_sample{j:02d}.png")
                save();print('Exposure',e,'complete',flush=True)
            # Independent 400ms reference frames, never reference compared with itself.
            # Two distinct inputs alternate; tests existing 200ms, not seeks shorter delay.
            if spec.get('timing_repeats',0):
                pair=spec['timing_rows'];refs={}
                for r in pair:
                    p=(ROOT/r['bmp']).resolve()
                    if not p.is_relative_to(ROOT) or sha(p)!=r['sha256']:raise ValueError('Timing asset mismatch')
                for e in [min(exposures),max(exposures)]:
                    set_exposure(e);hw.c['settle_delay_ms']=400
                    for r in pair:refs[r['id']]=capture(r['bmp'])[0][mask].astype(np.float32)
                    contrast=pcc(*refs.values())
                    hw.c['settle_delay_ms']=spec.get('test_wait_ms',200)
                    for repeat in range(spec['timing_repeats']):
                        for r in pair:
                            raw,meta=capture(r['bmp']);v=raw[mask].astype(np.float32)
                            scores={key:pcc(v,x) for key,x in refs.items()};pred=max(scores,key=scores.get)
                            report['timing'].append(dict(exposure_us=e,repeat=repeat,id=r['id'],scores=scores,
                                correct=pred==r['id'],reference_pair_pcc=contrast,meta=meta,
                                valid_reference=contrast<.95,stats=image_stats(raw,mask)))
                hw.c['settle_delay_ms']=original_wait
            set_exposure(spec.get('probe_exposure_us',150))
            raw,meta=capture(spec['probe']['bmp']);report['probe'].append(dict(which='after',stats=image_stats(raw,mask),meta=meta))
            report['phase_hold_pcc']=pcc(before,raw[mask]);preview(raw,'probe_after.png')
            report['recommendation']=recommend(report['rows']);save()
        report.update(status='complete',camera_restored=True)
    except BaseException as e:
        report.update(status='failed',error=str(e));raise
    finally:save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    if not a.spec.resolve().is_relative_to(ROOT/'results/dual_jobs'):raise ValueError('Job outside controlled directory')
    spec=read(a.spec);out=(ROOT/spec['out']).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output outside results')
    run(spec,read(a.config),out)
