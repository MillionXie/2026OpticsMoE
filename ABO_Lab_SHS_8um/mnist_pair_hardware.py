"""Paired A/B replay on the SAME predeclared 40 held-out MNIST inputs.

Reuses hashed native amplitude BMPs and independently measured geometry.
Captures BOTH arms anew, never substitutes the previous 60% hardware result.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from mnist_joint_capture import run as capture_run
from mnist_diagnostic import sha,read,pcc,energy
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--input-manifest',type=Path,required=True)
    p.add_argument('--link',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--remote-config',default='LAB.local.json');a=p.parse_args()
    a.out=a.out.resolve()
    if not a.out.is_relative_to(ROOT/'results'):raise ValueError('Output outside results')
    a.out.mkdir(exist_ok=False);m=read(a.input_manifest);ref=read(a.assets/'paired/pair_reference.json')
    assert sha(a.assets/'paired/pair_reference.npz')==ref['npz_sha256']
    z=np.load(a.assets/'paired/pair_reference.npz');result=read(a.assets/'result.json');assert result['status']=='complete'
    summary=dict(scope='40 fixed held-out inputs per arm, newly captured; not full dataset accuracy',arms={})
    for arm,name in [('A','A_old_fixed_native8'),('B','B_native8_best')]:
        phase=(a.assets/(name+'_xy_inverse.bmp')).resolve();expected=result['baseline_A']['export'] if arm=='A' else result['export_B']
        assert sha(phase)==expected['sha256'];mm=json.loads(json.dumps(m))
        for r in mm['rows']:r.update(phase=str(phase),phase_sha256=sha(phase),variant=arm)
        manifest=a.out/(arm+'_manifest.json');manifest.write_text(json.dumps(mm,indent=2))
        dest=a.out/arm;capture_run(manifest,a.link,dest,batch=True,config_rel=a.remote_config)
        report=read(dest/'report.json');rows=[];arrays={};H=np.array(m['H'])
        for r in report['rows']:
            assert sha(dest/(r['name']+'.png'))==r['raw_sha256']
            raw=np.asarray(Image.open(dest/(r['name']+'.png')))
            arr=cv2.warpPerspective(raw.astype(np.float32),H,(478,478),flags=cv2.INTER_LINEAR)
            arrays[r['name']]=arr;en=energy(arr,ref['detector_bounds']);idx=r['reference_index']
            rows.append(dict(name=r['name'],reference_index=idx,label=ref['rows'][idx]['label'],
                prediction=int(np.argmax(en)),energies=en,simulation_prediction=ref['predictions'][arm][idx],
                pcc=pcc(arr,z['ccd_'+arm][idx]),repeat_of=r.get('repeat_of'),
                saturation_fraction=float((arr>=254.5).mean())))
        evaluated=[r for r in rows if not r['repeat_of']];repeat=next(r for r in rows if r['repeat_of'])
        ar=dict(n=len(evaluated),accuracy=float(np.mean([r['label']==r['prediction'] for r in evaluated])),
            simulation_accuracy=float(np.mean([r['label']==r['simulation_prediction'] for r in evaluated])),
            mean_pcc=float(np.mean([r['pcc'] for r in evaluated])),
            repeat_pcc=pcc(arrays[repeat['name']],arrays[repeat['repeat_of']]),rows=rows,
            camera_config=report['remote_config_snapshot']['camera'])
        summary['arms'][arm]=ar
        np.savez_compressed(dest/'canonical.npz',**arrays)
        (a.out/'comparison.json').write_text(json.dumps(summary,indent=2));print(arm,{k:v for k,v in ar.items() if k!='rows'},flush=True)
    if summary['arms']['A']['camera_config']!=summary['arms']['B']['camera_config']:raise ValueError('A/B camera configuration changed')
    summary['complete']=True;(a.out/'comparison.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
