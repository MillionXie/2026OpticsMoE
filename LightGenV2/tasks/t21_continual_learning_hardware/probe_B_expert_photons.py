"""Single bounded B expert photon-budget control:6/8/12ms, TRAIN only.

Uses four immutable TRAIN pilot payloads as controls, never as new network CCD.
Every candidate has a new hardware identity; all formal layers must be new.
"""
import argparse,copy,hashlib,json,os,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .hardware_bridge import hardware,hardware_identity,native_admission,native_image,load_checkpoint,read_source,stage_signal_admission
from .train_native import sha,put


def candidate(parent,exposure,parent_sha):
    if exposure not in (6000,8000,12000) or parent['exposure_us']!=4000:raise ValueError('Only fixed B4->6/8/12ms comparison')
    cfg=copy.deepcopy(parent)
    for key in ('camera_device_info','same_phase_TRAIN_exposure_calibrated','same_phase_TRAIN_phase_SHA256',
                'same_phase_TRAIN_exposure_parent_SHA256','connection_spot_reference_exposure_us','spatial_geometry_verified'):
        cfg.pop(key,None)
    cfg.update(exposure_us=exposure,exposure_probe_us=[exposure],parent_hardware_SHA256=parent_sha,
        scope='T21 B8 expert fixed TRAIN photon-budget comparison, not TEST calibration',calibration_TEST_used=False,
        automatic_retry=False,response_inverse_reused_unchanged=True,scalar_exposure_only=True,network_background_subtraction=False)
    return cfg


def fixed_receipt_indices(count):
    if count!=48:raise ValueError('Only the predeclared B pilot48 accepted')
    return (0,12,24,47)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('hardware','package','source_capture','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true',help='Validate real TRAIN payloads/phase/SHA without opening SDK or writing a run')
    a=p.parse_args();torch.set_num_threads(2);cfg=json.loads(a.hardware.read_text());package=json.loads(a.package.read_text())
    source=json.loads((a.source_capture/'protocol.json').read_text())
    if sha(a.hardware)!=package['hardware_sha256'] or source['package']!=package or not source['pilot'] or package['stage']!='B':
        raise ValueError('Original B TRAIN pilot / hardware / package identity differs')
    old=json.loads((a.source_capture/'first/TRAIN_signal_admission/report.json').read_text())
    if old['passed']:raise ValueError('No diagnosed B expert failure')
    progress=json.loads((a.source_capture/'progress.json').read_text());indices=fixed_receipt_indices(progress['stage_frames'])
    if progress['stage']!='router' or progress['total_frames']!=48:raise ValueError('Unexpected or reused pilot prefix')
    model,cp=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    if model.architecture!='moe' or cp['stage']!=2 or int(model.active_count)!=8:raise ValueError('Frozen B eight-expert graph only')
    examples=[];identities=[];phase_image=None
    for index in indices:
        folder=a.source_capture/'router'/f'{index:05d}';r=json.loads((folder/'receipt.json').read_text())
        if r['split']!='train' or r['checkpoint_sha256']!=package['checkpoint_sha256'] or r['hardware_sha256']!=package['hardware_sha256']:
            raise ValueError('Controls must come from original TRAIN B prefix')
        spec=next(s for s in package['datasets'] if s['task']==r['task'] and s['split']=='train')
        if r['local_index'] not in spec['indices']:raise ValueError('TRAIN control outside fixed budget')
        data=read_source(spec['path'],spec['sha256'],'train',np.array([r['local_index']]))
        if int(data['source_indices'][0])!=r['source_index'] or int(data['y'][0])!=r['label'] or data['manifest']['identity_sha256']!=r['manifest_sha256']:
            raise ValueError('TRAIN source ID/label/manifest changed')
        for name,key in (('raw.npy','raw_sha256'),('ccd.npy','ccd_sha256'),('input.bmp','input_BMP_sha256')):
            if sha(folder/name)!=r[key]:raise ValueError('TRAIN control prefix SHA changed')
        if sha(a.source_capture/'router/phase.bmp')!=r['phase_BMP_sha256']:raise ValueError('TRAIN phase changed')
        amp,phase,_=model.payload(torch.from_numpy(data['x']),'first',{'router':torch.from_numpy(np.load(folder/'ccd.npy'))[None]})
        examples.append(np.rint(amp[0].numpy()*255).astype(np.uint8));image=native_image(np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8),True)
        if phase_image is not None and not np.array_equal(image,phase_image):raise ValueError('Input-dependent phase')
        phase_image=image;identities.append({k:r[k] for k in ('task','split','local_index','source_index','label','manifest_sha256','ccd_sha256')})
    if a.prepare_only:
        print(json.dumps(dict(passed=True,TRAIN_control_identities=identities,payload_shapes=[list(v.shape) for v in examples],
            phase_canonical_SHA256=hashlib.sha256(phase_image.tobytes()).hexdigest(),SDK_opened=False,GPU=False,run_written=False)))
        return
    Bench,LOCK,_,drive,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK already owned')
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(package_SHA256=sha(a.package),parent_hardware_SHA256=sha(a.hardware),checkpoint_SHA256=package['checkpoint_sha256'],
        fixed_pilot_indices=indices,TRAIN_control_identities=identities,candidate_exposure_us=[6000,8000,12000],maximum_control_frames=63,
        first_passing_exposure=True,thresholds_unchanged=True,coordinates_changed=False,LUT_changed=False,TEST_used=False,
        automatic_retry=False,network_frames=0,old_network_CCD_reuse_authorized=False,
        old_prefix_scope='Immutable TRAIN-derived displayed control patterns only, not a new exposure network evaluation'))
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='bounded_B_TRAIN_photon_comparison'));selected=None;rows=[]
    try:
        for exposure in (6000,8000,12000):
            current=candidate(cfg,exposure,sha(a.hardware));case=a.out/f'exposure_{exposure}';case.mkdir();put(case/'hardware.json',current)
            results=[];saturated=False
            with Bench(current,case) as bench:
                phasefile=case/'phase.bmp';Image.fromarray(phase_image).save(phasefile);bench.stage(phasefile)
                for index,gray in zip(indices[:3],examples[:3]):
                    folder=case/f'TRAIN_pilot_{index:04d}'
                    try:stage_signal_admission(bench,folder,[gray,examples[-1]],drive,warp,None,'first',hardware_identity(current),model)
                    except RuntimeError:
                        if not (folder/'report.json').exists():raise
                    r=json.loads((folder/'report.json').read_text());saturated=max(v['signal']['saturation_fraction'] for v in r['rows'].values())>.15
                    results.append(dict(pilot_index=index,passed=r['passed'],SNR=r['SNR'],repeat_correlation=r['repeat_correlation'],repeat_consumer_drift=r['repeat_consumer_drift'],saturated=saturated))
                    if saturated:break
            passed=len(results)==3 and all(r['passed'] for r in results);row=dict(exposure_us=exposure,passed=passed,rows=results,hardware_SHA256=sha(case/'hardware.json'))
            rows.append(row);put(case/'summary.json',row);put(a.out/'progress.json',dict(candidates=rows))
            if passed:selected=current;put(a.out/'admitted_hardware.json',selected);break
            if saturated:break
    except BaseException:
        put(a.out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK ownership changed')
        LOCK.unlink()
    if selected is None:
        put(a.out/'report.json',dict(passed=False,candidates=rows,SDK_released=True,TEST_used=False,automatic_retry=False))
        raise RuntimeError('No passing exposure within declared B comparison, preserve all evidence')
    native_admission(selected,a.out/'native_admission')
    put(a.out/'report.json',dict(passed=True,candidates=rows,selected_exposure_us=selected['exposure_us'],
        admitted_hardware_SHA256=sha(a.out/'admitted_hardware.json'),native_admission_SHA256=sha(a.out/'native_admission/report.json'),
        SDK_released=True,TEST_used=False,network_frames=0,all_formal_network_layers_require_new_CCD=True,
        coordinates_LUT_PT_unchanged=True,phase_LUT_scope='linear voltage, not calibrated linear phase'))


if __name__=='__main__':main()
