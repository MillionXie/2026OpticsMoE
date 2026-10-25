"""Finite TRAIN-only blur-response diagnosis; no device or deployed-graph update."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .hardware_bridge import load_checkpoint,read_source
from .analyze_TRAIN_phase_response import bin4
from .audit_TRAIN_free_space_boundary import padded_intensity
from .train_native import sha,put
from LightGenV2.tasks.t20_reconfigurable_rgb_sar.reference.optics import AngularSpectrumPropagator


def blur(value,sigma):
    if value.ndim!=2 or sigma<0:raise ValueError('2D field and nonnegative width required')
    if sigma==0:return value.copy()
    radius=int(np.ceil(3*sigma));x=np.arange(-radius,radius+1);kernel=np.exp(-x*x/(2*sigma*sigma));kernel/=kernel.sum()
    out=value.copy()
    for axis in (0,1):
        pad=[(0,0),(0,0)];pad[axis]=(radius,radius)
        out=np.apply_along_axis(lambda v:np.convolve(v,kernel,mode='valid'),axis,np.pad(out,pad,mode='reflect'))
    return out


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('probe','package','phase_response','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);started=time.time()
    report=json.loads((a.probe/'report.json').read_text());protocol=json.loads((a.probe/'protocol.json').read_text());candidate=json.loads(a.phase_response.read_text());package=json.loads(a.package.read_text())
    if not report['passed'] or not report['SDK_released'] or sha(a.package)!=protocol['package_SHA256'] or sha(a.probe/'report.json')!=candidate['source_probe_report_SHA256']:raise ValueError('Sealed TRAIN controls required')
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256']);spec=next(v for v in package['datasets'] if v['task']=='clevr' and v['split']=='train')
    d=read_source(spec['path'],spec['sha256'],'train',np.array([0,1]));actual=[];sim=[]
    propagator=AngularSpectrumPropagator(532e-9,8e-6,2112,.1)
    levels=np.array(candidate['voltage_levels']);curve=np.array(candidate['joint']['positive_orientation_phase_radians'])*candidate['chosen_sign_by_TRAIN0']
    for j in range(2):
        amp,voltage,_=model.payload(torch.from_numpy(d['x'][j])[None].float(),'first',{})
        pp=torch.from_numpy((np.interp(voltage.numpy()*256,levels,curve)/(2*np.pi)).astype(np.float32))
        sim.append(bin4((padded_intensity(amp,pp,propagator)-padded_intensity(amp,torch.zeros_like(pp),propagator))[0].numpy()).astype(np.float64))
        actual.append(bin4(np.load(a.probe/f'restored_TRAIN{j}_ccd.npy')-np.load(a.probe/f'cell32_voltage0_TRAIN{j}_ccd.npy')).astype(np.float64))
    rows=[]
    for sigma in (0,.5,1,2,4,8):
        predictions=[blur(v,sigma).ravel() for v in sim];observed=[v.ravel() for v in actual]
        gain=float(observed[0]@predictions[0]/max(predictions[0]@predictions[0],1e-30));r=dict(sigma_original_pixels=sigma*4,TRAIN0_gain=gain)
        for j in range(2):
            r[f'TRAIN{j}_correlation']=float(np.corrcoef(observed[j],predictions[j])[0,1])
            r[f'TRAIN{j}_R2']=float(1-np.sum((observed[j]-gain*predictions[j])**2)/max(np.sum(observed[j]**2),1e-30))
        rows.append(r)
    chosen=max(rows,key=lambda r:r['TRAIN0_R2'])
    result=dict(passed=True,scope='Spatial blur diagnosis ONLY; two TRAIN controls are insufficient for new calibration admission',
        selection='Fixed six widths, TRAIN0 R2 only; TRAIN1 descriptive cross-input check, not independent validation',rows=rows,selected=chosen,
        empirical_phase_candidate_still_not_admitted=True,new_calibration_admitted=False,TRAIN_only=True,TEST_used=False,SDK=False,GPU=False,
        actual_coordinate_LUT_exposure_processing_and_network_unchanged=True,new_network_frames=0,seconds=time.time()-started,
        source_probe_report_SHA256=sha(a.probe/'report.json'),source_phase_candidate_SHA256=sha(a.phase_response),source_SHA256=sha(__file__))
    a.out.mkdir(parents=True,exist_ok=False);put(a.out/'report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
