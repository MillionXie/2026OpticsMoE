"""TRAIN-only empirical phase manifold and held-control forward check.

Two-level phase-only masks imply I(g)=a+b*cos(phi(g))+c*sin(phi(g)).
Ellipse angle has a SIGN ambiguity, resolved only by a separately recorded
known-model TRAIN spatial check. No hardware/LUT/model update is performed.
"""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from .train_native import sha,put


def phase_manifold(matrix):
    x=np.asarray(matrix,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]!=9 or not np.isfinite(x).all():raise ValueError('Nine finite voltage conditions required')
    x=x-x.mean(0,keepdims=True);gram=x@x.T;values,vectors=np.linalg.eigh(gram);order=np.argsort(values)[::-1]
    values=values[order].clip(0);vectors=vectors[:,order];fraction=float(values[:2].sum()/max(values.sum(),1e-30))
    if values[1]<1e-12:raise ValueError('Insufficient two-dimensional phase response')
    coords=vectors[:,:2]*np.sqrt(values[:2]);coords=coords/coords.std(0)
    a,b=coords.T;design=np.column_stack((a*a,a*b,b*b,a,b));coef=np.linalg.lstsq(design,np.ones(9),rcond=None)[0]
    q=np.array([[coef[0],coef[1]/2],[coef[1]/2,coef[2]]]);e,v=np.linalg.eigh(q)
    if np.min(e)<=0:raise ValueError('TRAIN response conic is not an ellipse')
    center=-.5*np.linalg.solve(q,coef[3:]);radius=1+center@q@center
    if radius<=0:raise ValueError('Invalid ellipse radius')
    circle=(coords-center)@(v@np.diag(np.sqrt(e))@v.T)/np.sqrt(radius)
    radial_error=float(np.sqrt(np.mean((np.linalg.norm(circle,axis=1)-1)**2)))
    phase=np.unwrap(np.arctan2(circle[:,1],circle[:,0]));phase-=phase[0]
    if phase[-1]<0:phase=-phase
    return dict(rank2_variance_fraction=fraction,ellipse_radial_RMS=radial_error,
        positive_orientation_phase_radians=phase.tolist(),phase_sign_not_identified_by_ellipse=True,
        largest_wrapped_step_radians=float(np.max(np.abs(np.diff(phase)))),
        negative_phase_steps=int((np.diff(phase)<-.15).sum()),
        raw_covariance_eigenvalues=values.tolist())


def bin4(x):return x.reshape(254,4,254,4).mean((1,3))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.probe/'report.json').read_text());meta=json.loads((a.probe/'protocol.json').read_text())
    if not r['passed'] or r['frames']!=40 or not r['SDK_released'] or sha(a.package)!=meta['package_SHA256']:raise ValueError('Sealed fixed TRAIN probe required')
    for item in r['frame_receipts']:
        if sha(a.probe/(item['name']+'_ccd.npy'))!=item['ccd_SHA256']:raise ValueError('Control CCD changed')
    levels=np.array(meta['voltage_levels']);blocks=[]
    for j in range(2):
        parts=[]
        for cell in meta['cell_pixels']:
            parts.append(np.stack([bin4(np.load(a.probe/f'cell{cell}_voltage{g}_TRAIN{j}_ccd.npy')).ravel() for g in levels]))
        blocks.append(np.concatenate(parts,1))
    result=dict(scope='TRAIN-only phase-response candidate, no device LUT or weight update',voltage_levels=levels.tolist(),TRAIN_only=True,TEST_used=False,SDK=False,GPU=False,
        source_probe_report_SHA256=sha(a.probe/'report.json'),source_probe_protocol_SHA256=sha(a.probe/'protocol.json'),
        joint=phase_manifold(np.concatenate(blocks,1)),per_TRAIN=[phase_manifold(v) for v in blocks],
        controls_share_one_source_image_pair=True,independent_device_calibration_claim=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__))
    # Resolve only the empirical sign, not by selecting TEST accuracy. Known
    # first-stage spatial changes on TRAIN0 fit a positive radiometric gain;
    # TRAIN1 is a held response check, not an independent semantic test.
    import torch
    from .hardware_bridge import load_checkpoint,read_source
    torch.set_num_threads(2);package=json.loads(a.package.read_text());model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    spec=next(v for v in package['datasets'] if v['task']=='clevr' and v['split']=='train');source=read_source(spec['path'],spec['sha256'],'train',np.array([0,1]))
    phase=np.array(result['joint']['positive_orientation_phase_radians']);actual=[];amp=[];voltage=[]
    for j in range(2):
        actual.append(bin4(np.load(a.probe/f'restored_TRAIN{j}_ccd.npy')-np.load(a.probe/f'cell32_voltage0_TRAIN{j}_ccd.npy')).ravel().astype(np.float64))
        with torch.no_grad():aa,pp,_=model.payload(torch.from_numpy(source['x'][j])[None].float(),'first',{})
        amp.append(aa);voltage.append(pp.numpy()*256)
    rows=[]
    with torch.no_grad():
        for sign in (1,-1):
            simulated=[]
            for j in range(2):
                mapped=np.interp(voltage[j],levels,sign*phase).astype(np.float32)/(2*np.pi)
                changed=model.propagate_payload(amp[j],torch.from_numpy(mapped)).numpy()[0]
                base=model.propagate_payload(amp[j],torch.zeros((1016,1016))).numpy()[0]
                simulated.append(bin4(changed-base).ravel().astype(np.float64))
            gain=float(actual[0]@simulated[0]/max(simulated[0]@simulated[0],1e-30))
            rows.append(dict(sign=sign,TRAIN0_positive_radiometric_gain=gain,
                TRAIN0_correlation=float(np.corrcoef(actual[0],simulated[0])[0,1]),
                TRAIN1_correlation=float(np.corrcoef(actual[1],simulated[1])[0,1]),
                TRAIN1_prediction_R2=float(1-np.sum((actual[1]-gain*simulated[1])**2)/max(np.sum(actual[1]**2),1e-30))))
    chosen=max(rows,key=lambda v:v['TRAIN0_correlation'] if v['TRAIN0_positive_radiometric_gain']>0 else -2)
    curve=result['joint'];admitted=curve['rank2_variance_fraction']>=.95 and curve['ellipse_radial_RMS']<=.1 and curve['negative_phase_steps']==0 and curve['largest_wrapped_step_radians']<np.pi-.1 and phase[-1]>=2*np.pi and chosen['TRAIN0_positive_radiometric_gain']>0 and chosen['TRAIN1_prediction_R2']>=.5
    result.update(spatial_forward_sign_checks=rows,chosen_sign_by_TRAIN0=chosen['sign'],candidate_admitted=bool(admitted),
        admission_rule='rank2>=.95, ellipse_RMS<=.1, no negative steps, wrapped steps<pi-.1, covers2pi, positive gain and held TRAIN1 R2>=.5',
        no_hardware_or_weights_written=True,independent_TEST_claim=False,completed=time.time())
    a.out.mkdir(parents=True,exist_ok=False);put(a.out/'phase_response_candidate.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
