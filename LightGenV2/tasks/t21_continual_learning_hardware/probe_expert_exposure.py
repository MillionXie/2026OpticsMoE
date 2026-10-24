"""Finite3/4ms TRAIN-only photon-budget check of a diagnosed boundary.

Every exposure has a new hardware identity. No old network CCD is mixed in.
Coordinates, LUT, PT and admission thresholds remain unchanged.
"""
import argparse,copy,json,os,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .hardware_bridge import (hardware,hardware_identity,native_admission,native_image,
                              load_checkpoint,read_source,stage_signal_admission)
from .train_native import put,sha


def candidate_hardware(parent,exposure_us,parent_sha):
    if exposure_us not in (3000,4000):raise ValueError('Only bounded3/4ms candidates')
    cfg=copy.deepcopy(parent)
    for name in ('camera_device_info','same_phase_TRAIN_exposure_calibrated',
                 'same_phase_TRAIN_phase_SHA256','same_phase_TRAIN_exposure_parent_SHA256',
                 'connection_spot_reference_exposure_us','bounded_TRAIN_comparison_reason','spatial_geometry_verified'):
        cfg.pop(name,None)
    cfg.update(exposure_us=exposure_us,exposure_probe_us=[exposure_us],
        scope='T21 new exposure session; fixed TRAIN-only expert comparison, not TEST calibration',
        parent_hardware_SHA256=parent_sha,calibration_TEST_used=False,automatic_retry=False,
        response_inverse_reused_unchanged=True,scalar_exposure_only=True,network_background_subtraction=False)
    return cfg


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('hardware','package','source_capture','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    cfg=json.loads(a.hardware.read_text());package=json.loads(a.package.read_text())
    if sha(a.hardware)!=package['hardware_sha256']:raise ValueError('Parent hardware changed')
    if json.loads((a.source_capture/'protocol.json').read_text())['package']!=package:
        raise ValueError('Wrong parent capture')
    previous=json.loads((a.source_capture/'first/TRAIN_signal_admission/report.json').read_text())
    if previous['passed']:raise ValueError('Only a diagnosed inadequate2ms boundary')
    indices=[0,450,950,999]
    spec=next(s for s in package['datasets'] if s['split']=='train')
    if spec['indices']!=list(range(1000)):raise ValueError('Original TRAIN order required')
    data=read_source(spec['path'],spec['sha256'],'train',np.array(indices))
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    if model.architecture!='moe':raise ValueError('MoE boundary only')
    examples=[];phase_image=None
    for j,index in enumerate(indices):
        folder=a.source_capture/'router'/f'{index:05d}';r=json.loads((folder/'receipt.json').read_text())
        if r['split']!='train' or r['source_index']!=int(data['source_indices'][j]):raise ValueError('TRAIN identity mismatch')
        if r['checkpoint_sha256']!=package['checkpoint_sha256'] or r['hardware_sha256']!=package['hardware_sha256']:
            raise ValueError('Wrong prefix physical identity')
        for name,key in (('ccd.npy','ccd_sha256'),('raw.npy','raw_sha256'),('input.bmp','input_BMP_sha256')):
            if sha(folder/name)!=r[key]:raise ValueError('Prefix SHA mismatch')
        prefix={'router':torch.from_numpy(np.load(folder/'ccd.npy'))[None]}
        amplitude,phase,_=model.payload(torch.from_numpy(data['x'][j])[None],'first',prefix)
        examples.append(np.rint(amplitude[0].numpy()*255).astype(np.uint8))
        image=native_image(np.rint(phase.numpy()*256).clip(0,255).astype(np.uint8),True)
        if phase_image is not None and not np.array_equal(image,phase_image):raise ValueError('Input-dependent phase')
        phase_image=image
    Bench,LOCK,_,drive,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK owned')
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(package_SHA256=sha(a.package),parent_hardware_SHA256=sha(a.hardware),
        checkpoint_SHA256=package['checkpoint_sha256'],selected_TRAIN_indices=indices[:3],
        source_indices=data['source_indices'].tolist(),candidate_exposure_us=[3000,4000],
        maximum_control_frames=42,first_passing_exposure=True,thresholds_unchanged=True,
        coordinates_changed=False,LUT_changed=False,TEST_used=False,automatic_retry=False,
        network_frames=0,old_network_CCD_reuse_authorized=False))
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='finite_TRAIN_only_exposure_comparison'))
    selected=None;rows=[]
    try:
        for exposure in (3000,4000):
            current=candidate_hardware(cfg,exposure,sha(a.hardware))
            case=a.out/f'exposure_{exposure}';case.mkdir();put(case/'hardware.json',current)
            results=[];saturated=False
            with Bench(current,case) as bench:
                phasefile=case/'phase.bmp';Image.fromarray(phase_image).save(phasefile);bench.stage(phasefile)
                for index,gray in zip(indices[:3],examples[:3]):
                    folder=case/f'TRAIN_{index:04d}'
                    try:
                        stage_signal_admission(bench,folder,[gray,examples[-1]],drive,warp,None,'first',
                                               hardware_identity(current),model)
                    except RuntimeError:
                        if not (folder/'report.json').exists():raise
                    r=json.loads((folder/'report.json').read_text())
                    saturated=max(row['signal']['saturation_fraction'] for row in r['rows'].values())>.15
                    results.append(dict(index=index,passed=r['passed'],SNR=r['SNR'],
                        repeat_correlation=r['repeat_correlation'],repeat_consumer_drift=r['repeat_consumer_drift'],saturated=saturated))
                    if saturated:break
            passed=len(results)==3 and all(r['passed'] for r in results)
            row=dict(exposure_us=exposure,passed=passed,rows=results,hardware_SHA256=sha(case/'hardware.json'))
            rows.append(row);put(case/'summary.json',row);put(a.out/'progress.json',dict(candidates=rows))
            if passed:selected=current;put(a.out/'admitted_hardware.json',selected);break
            if saturated:break
    except BaseException:
        put(a.out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK owner changed')
        LOCK.unlink()
    if selected is None:
        put(a.out/'report.json',dict(passed=False,candidates=rows,SDK_released=True,TEST_used=False,automatic_retry=False))
        raise RuntimeError('No passing exposure in bounded TRAIN comparison')
    native_admission(selected,a.out/'native_admission')
    put(a.out/'report.json',dict(passed=True,candidates=rows,selected_exposure_us=selected['exposure_us'],
        admitted_hardware_SHA256=sha(a.out/'admitted_hardware.json'),
        native_admission_SHA256=sha(a.out/'native_admission/report.json'),
        SDK_released=True,TEST_used=False,network_frames=0,old_2ms_network_frames_not_mixed=True,
        phase_LUT_scope='linear voltage, not measured linear phase'))


if __name__=='__main__':main()
