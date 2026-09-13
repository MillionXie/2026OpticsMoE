"""Export paired small-sample native8 simulations AFTER completed training.

Loads the exact committed native8 implementation without changing server tree.
Full-test metrics remain those in result.json; these are fixed hardware inputs.
"""
import argparse,json,subprocess,types,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    result=json.loads((a.run/'result.json').read_text());assert result['status']=='complete'
    protocol=json.loads((a.run/'protocol.json').read_text());commit=protocol['source_commit']
    mod=types.ModuleType('pinned_native8');sys.modules[mod.__name__]=mod
    source=subprocess.check_output(['git','show',commit+':ABO_Lab_SHS_8um/mnist_native8.py'],text=True)
    exec(compile(source,'pinned_native8.py','exec'),mod.__dict__)
    ref=json.loads((a.reference/'reference.json').read_text());assert mod.sha(a.reference/'reference.npz')==ref['npz_sha256']
    z=np.load(a.reference/'reference.npz');s=mod.load_settings(mod.BASE/'configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_angle_roi.yaml')
    model=mod.PitchModel(s).to(a.device).eval();x=torch.tensor(z['amplitude'][:,39:-39,39:-39,None]).permute(0,3,1,2)
    outputs={};preds={};a.out.mkdir(parents=True,exist_ok=False)
    for arm,name in [('A','A_old_fixed_native8'),('B','B_native8_best')]:
        phase=np.load(a.run/(name+'_phase_rad.npy'));phase=torch.tensor(phase,device=a.device)
        model.raw_phase.data.copy_(torch.logit((phase/(2*torch.pi)).clamp(1e-7,1-1e-7)))
        ccd=[];energies=[]
        with torch.inference_mode():
            for batch in x.split(4):
                I,e=model(batch.to(a.device));ccd.append(F.interpolate(I[:,None],size=(478,478),mode='area')[:,0].cpu().numpy());energies.append(e.cpu().numpy())
        outputs['ccd_'+arm]=np.concatenate(ccd);outputs['energy_'+arm]=np.concatenate(energies)
        preds[arm]=outputs['energy_'+arm].argmax(1).tolist()
    np.savez_compressed(a.out/'pair_reference.npz',amplitude=z['amplitude'],**outputs)
    report=dict(native_source_commit=commit,full_test=result,reference_sha256=ref['npz_sha256'],rows=ref['rows'],
        detector_bounds=ref['detector_bounds'],predictions=preds,simulation_display='native1016 intensity area-downsampled to478; labels use native physical-area detector energies',
        npz_sha256=mod.sha(a.out/'pair_reference.npz'))
    (a.out/'pair_reference.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(export=str(a.out),sha256=report['npz_sha256'])))
if __name__=='__main__':main()
