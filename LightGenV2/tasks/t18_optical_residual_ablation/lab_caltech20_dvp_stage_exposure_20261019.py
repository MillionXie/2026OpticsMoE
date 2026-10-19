"""Bounded TRAIN-only per-stage exposure admission; no model normalization/refit."""
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from lab_caltech10_20261015 import corners,native,phase_native,save,sha
from lab_caltech20_dvp_20261019 import Bench,orient,LOCK


def exposure_passes(model_rows,control_rows,base_controls,ratio,cfg):
    """Fixed photon/range criteria, independent of labels or prediction accuracy."""
    assert len(model_rows)==cfg['TRAIN_probe_samples'] and [r['tag'] for r in control_rows]==['dark','gray64','gray96']
    usable=[]
    for row in control_rows[1:]:
        baseline=base_controls[row['tag']]-base_controls['dark']
        expected=baseline*ratio;actual=row['mean']-control_rows[0]['mean']
        if baseline>=.4 and row['saturation_fraction']<.001:
            usable.append(abs(actual-expected)<=max(cfg['linearity_absolute_gray'],cfg['linearity_relative_tolerance']*expected))
    return bool(usable and all(usable) and control_rows[0]['p99']<=cfg['max_dark_p99'] and all(r['p99']>=cfg['probe_min_p99'] and r['saturation_fraction']<=cfg['probe_max_saturation'] for r in model_rows))


def calibration(out,packet,hardware,base_digest,cfg,recheck):
    """Packet contains only fixed TRAIN0..39, exact stage inputs and immutable PT."""
    import cv2
    out=Path(out);assert not out.exists() and not LOCK.exists();out.mkdir(parents=True)
    assert len(packet['records'])==cfg['TRAIN_probe_samples']==40 and all(r['role']=='train' for r in packet['records'])
    assert len({r['label'] for r in packet['records']})==20
    warp=cv2.getPerspectiveTransform(np.float32(corners(Path(hardware['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    drive=np.asarray(json.loads(Path(hardware['response_lut']).read_text())['drive'],np.uint8)
    flat=out/'flat_phase.bmp';Image.fromarray(phase_native(np.zeros((478,478),np.uint8))).save(flat)
    model_phase=out/'model_phase.bmp';Image.fromarray(phase_native(packet['phase'])).save(model_phase)
    trials=[];selected=None;rows=[];started=time.monotonic()
    def shot(bench,folder,tag,canonical,apply_inverse,record=None):
        folder.mkdir(parents=True,exist_ok=True);base=folder/tag
        input_path=base.with_name(base.name+'_input.png');Image.fromarray(canonical).save(input_path,compress_level=1)
        bmp=base.with_suffix('.bmp');Image.fromarray(native(drive[canonical] if apply_inverse else canonical,'amplitude')).save(bmp)
        raw,meta=bench.capture(bmp);crop=orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),hardware['orientation'])
        raw_path=base.with_name(base.name+'_raw.png');Image.fromarray(raw).save(raw_path,compress_level=1);Image.fromarray(crop).save(base.with_suffix('.png'),compress_level=1)
        row=dict(tag=tag,mean=float(crop.mean()),p99=float(np.percentile(crop,99)),saturation_fraction=float((crop==255).mean()),raw_SHA256=sha(raw_path),canonical_SHA256=sha(input_path),input_SHA256=sha(bmp),frame_metadata=meta,captured_at_ns=time.time_ns(),record=record)
        save(base.with_suffix('.json'),row);bmp.unlink();rows.append(row);save(out/'progress.json',dict(state='TRAIN_only_exposure_admission',stage=packet['stage'],patterns=len(rows),latest=row))
        return row
    with LOCK.open('x') as stream:stream.write(json.dumps(dict(pid=os.getpid(),kind='TRAIN_only_layer_exposure',stage=packet['stage'],out=str(out))))
    try:
        with Bench(hardware,out) as bench:
            recheck(bench,out,hardware,warp)
            bench.stage(flat);base_controls={}
            for tag,g in (('dark',0),('gray64',64),('gray96',96)):
                base_controls[tag]=shot(bench,out/'reference_controls',tag,np.full((478,478),g,np.uint8),False)['mean']
            for position,requested in enumerate(cfg['exposure_probe_us']):
                bench.settings=bench.camera.settings(exposure=requested,gain=1.)
                for _ in range(hardware['discard_frames']):bench.read(check_settings=False)
                current=out/f'exposure_{position:02d}';bench.stage(flat);controls=[]
                for tag,g in (('dark',0),('gray64',64),('gray96',96)):
                    controls.append(shot(bench,current/'controls',tag,np.full((478,478),g,np.uint8),False))
                bench.stage(model_phase);model_rows=[]
                for item in packet['records']:
                    index=item['index'];model_rows.append(shot(bench,current/'train',f'{index:06d}',packet['inputs'][index],True,item))
                ratio=bench.settings['exposure_us']/hardware['exposure_us'];passed=exposure_passes(model_rows,controls,base_controls,ratio,cfg)
                trial=dict(requested_exposure_us=requested,actual=bench.settings.copy(),passed=passed,TRAIN=40,TEST=0,min_p99=min(r['p99'] for r in model_rows),max_saturation=max(r['saturation_fraction'] for r in model_rows),probe_directory=str(current/'train'))
                trials.append(trial);save(out/'trials.json',trials);print(json.dumps(trial),flush=True)
                if passed:selected=trial;break
                if trial['max_saturation']>.15:break
            assert selected,'No exposure in fixed ladder met TRAIN signal/clip/linearity admission; preserve all frames, no retry or relaxation'
        report=dict(state='complete',passed=True,selected=selected,trials=trials,TRAIN=40,TEST=0,VALID=0,TEST_used=False,PT_SHA256=packet['PT_SHA256'],data_SHA256=packet['data_SHA256'],prefix_SHA256=packet['prefix_SHA256'],phase_SHA256=sha(model_phase),base_admission_SHA256=base_digest,settings_selection='first fixed ladder exposure passing all40 TRAIN signal and known-pattern radiometric checks',inverse_LUT_unchanged=True,no_input_peak_normalization=True,SDK_released=True,elapsed_s=time.monotonic()-started,source_SHA256=sha(__file__),protocol=cfg)
        report.update(source_git=packet['source_git'],layerwise_source_SHA256=packet['layerwise_source_SHA256']);save(out/'report.json',report)
        admitted=dict(hardware,scope='TRAIN-only per-stage exposure admission for capture',exposure_us=selected['actual']['exposure_us'],base_admission_SHA256=base_digest,stage=packet['stage'],PT_SHA256=packet['PT_SHA256'],data_SHA256=packet['data_SHA256'],stage_admission_report=str(out/'report.json'),stage_admission_report_SHA256=sha(out/'report.json'),probe_directory=selected['probe_directory'],probe_indices=[r['index'] for r in packet['records']],phase_SHA256=sha(model_phase),exposure_admission='TRAIN/known patterns only, no TEST',response_lut_scope='original amplitude-SLM inverse unchanged; exposure transfer checked using unsaturated known gray controls')
        save(out/'admitted_hardware.json',admitted)
        return admitted
    except BaseException as error:
        save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),patterns=len(rows),automatic_retry=False));raise
    finally:LOCK.unlink()
