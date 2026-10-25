"""Bounded TRAIN-only FFT-boundary diagnostic, no hardware or model updates.

Same532nm/8um/0.1m at canvas1056/2112/3168. Nominal and empirical phase
response are separately compared; no rejected calibration becomes admission.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint,read_source
from .analyze_TRAIN_phase_response import bin4
from .train_native import sha,put
from LightGenV2.tasks.t20_reconfigurable_rgb_sar.reference.optics import AngularSpectrumPropagator


def padded_intensity(amplitude,phase,propagator):
    size=amplitude.shape[-1];height,width=propagator.grid_size
    if amplitude.ndim!=3 or amplitude.shape[-2:]!=(size,size) or height!=width or height<size or (height-size)%2:
        raise ValueError('Centered unchanged square aperture required')
    border=(height-size)//2;field=F.pad(amplitude*torch.exp(2j*torch.pi*phase),(border,)*4)
    value=propagator(field).abs().square()
    return value[:,border:border+size,border:border+size]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('probe','package','phase_response','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    report=json.loads((a.probe/'report.json').read_text());protocol=json.loads((a.probe/'protocol.json').read_text());phase=json.loads(a.phase_response.read_text());package=json.loads(a.package.read_text())
    if not report['passed'] or not report['SDK_released'] or sha(a.package)!=protocol['package_SHA256'] or sha(a.probe/'report.json')!=phase['source_probe_report_SHA256']:
        raise ValueError('Sealed original TRAIN controls required')
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256']);spec=next(v for v in package['datasets'] if v['task']=='clevr' and v['split']=='train')
    d=read_source(spec['path'],spec['sha256'],'train',np.array([0,1]));amp=[];voltage=[];actual=[]
    for j in range(2):
        aa,pp,_=model.payload(torch.from_numpy(d['x'][j])[None].float(),'first',{});amp.append(aa);voltage.append(pp)
        actual.append(bin4(np.load(a.probe/f'restored_TRAIN{j}_ccd.npy')-np.load(a.probe/f'cell32_voltage0_TRAIN{j}_ccd.npy')).ravel().astype(np.float64))
    levels=np.array(phase['voltage_levels']);curve=np.array(phase['joint']['positive_orientation_phase_radians'])*phase['chosen_sign_by_TRAIN0'];rows=[]
    started=time.time()
    for grid in (1056,2112,3168):
        propagator=AngularSpectrumPropagator(532e-9,8e-6,grid,.1)
        for mode in ('nominal_linear2pi','empirical_TRAIN_phase_candidate_not_admitted'):
            differences=[];energy=[]
            for j in range(2):
                pp=voltage[j] if mode=='nominal_linear2pi' else torch.from_numpy((np.interp(voltage[j].numpy()*256,levels,curve)/(2*np.pi)).astype(np.float32))
                value=padded_intensity(amp[j],pp,propagator)[0];zero=padded_intensity(amp[j],torch.zeros_like(pp),propagator)[0]
                if grid==1056 and mode=='nominal_linear2pi':
                    original=model.propagate_payload(amp[j],pp)[0]
                    if not torch.equal(value,original):raise RuntimeError('Original padding reference differs')
                differences.append(bin4((value-zero).numpy()).ravel().astype(np.float64))
                energy.append(dict(TRAIN=j,canonical_aperture_fraction=float(value.sum()/amp[j].square().sum()),zero_aperture_fraction=float(zero.sum()/amp[j].square().sum())))
            gain=float(actual[0]@differences[0]/max(differences[0]@differences[0],1e-30))
            rows.append(dict(grid_pixels=grid,padding_pixels_each_side=(grid-1016)//2,phase_mode=mode,
                TRAIN0_correlation=float(np.corrcoef(actual[0],differences[0])[0,1]),TRAIN0_radiometric_gain=gain,
                TRAIN1_correlation=float(np.corrcoef(actual[1],differences[1])[0,1]),
                TRAIN1_prediction_R2=float(1-np.sum((actual[1]-gain*differences[1])**2)/max(np.sum(actual[1]**2),1e-30)),energy=energy))
    result=dict(passed=True,scope='Numerical boundary diagnostic ONLY, not calibration admission or performance',rows=rows,
        fixed_distance_m=.1,fixed_pitch_um=8,fixed_wavelength_nm=532,TRAIN_only=True,TEST_used=False,SDK=False,GPU=False,
        hardware_unchanged=True,model_weights_unchanged=True,phase_candidate_admitted=False,new_network_frames=0,
        source_probe_report_SHA256=sha(a.probe/'report.json'),source_phase_candidate_SHA256=sha(a.phase_response),seconds=time.time()-started,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__))
    a.out.mkdir(parents=True,exist_ok=False);put(a.out/'report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
