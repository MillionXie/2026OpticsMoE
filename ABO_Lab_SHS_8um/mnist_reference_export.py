"""Run in the original repository on GPU/CPU; export fixed held-out MNIST inputs.

No training, no success-based selection, no simulation used as measured CCD.
This standalone source can be executed from `git show COMMIT:path | python -`.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from experiments.d2nn_mnist4_single_layer_17um_10cm_v2.settings import load_settings
from experiments.d2nn_mnist4_single_layer_17um_10cm_v2.data import build_datasets
from experiments.d2nn_mnist4_single_layer_17um_10cm_v2.modeling import RobustRawCCDMNIST4D2NN

PIN='e297b9baa4c028b49695daed24bb291bb71cd93a23f43e2b01ee1d87b1607887'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--dataset-root',type=Path,required=True)
    a=p.parse_args();base=Path('experiments/d2nn_mnist4_single_layer_17um_10cm_v2')
    cfg=base/'configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_angle_roi.yaml'
    ckpt=base/'runs/mnist4_single_layer_17um_10cm_v2_angle_roi/mask_candidates/checkpoints/post_robust_best.pt'
    assert sha(ckpt)==PIN
    s=load_settings(cfg);s.download=False;s.dataset_root=a.dataset_root.resolve();bundle=build_datasets(s)
    model=RobustRawCCDMNIST4D2NN(s).to(a.device).eval()
    payload=torch.load(ckpt,map_location='cpu',weights_only=False) # trusted, SHA-pinned own checkpoint
    model.load_state_dict(payload['model_state_dict'],strict=True)
    selected={k:[] for k in s.classes}
    for i,original in enumerate(bundle.test.indices):
        y=int(bundle.test.dataset.targets[original])
        if len(selected[y])<12:selected[y].append((i,int(original)))
        if all(len(v)==12 for v in selected.values()):break
    rows=[];amplitudes=[];intensities=[]
    with torch.inference_mode():
        for rank in range(12):
            for label in s.classes:
                i,original=selected[label][rank];x,y=bundle.test[i];v=model(x.unsqueeze(0).to(a.device))
                scores=v['detector_energy'][0].cpu().numpy()
                rows.append(dict(key=f'mnist_i{original:05d}_y{label}',label=int(y),official_test_index=original,
                                 split='orientation_pilot' if rank<2 else 'heldout_measurement',simulation_prediction=int(scores.argmax()),
                                 simulation_raw_energies=scores.tolist()))
                amplitudes.append(v['active_amplitude'][0].cpu().numpy());intensities.append(v['ccd_intensity'][0].cpu().numpy())
    a.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.out/'reference.npz',amplitude=np.stack(amplitudes),ccd=np.stack(intensities),phase=model.phase().detach().cpu().numpy())
    src=Path('experiments/d2nn_mnist4_single_layer_17um_10cm_v2')
    report=dict(checkpoint_sha256=PIN,checkpoint=str(ckpt),epoch=payload['epoch'],selection='First 12 official test items per class, in original order; no outcome filtering',
                rows=rows,detector_bounds=s.detector_bounds(),logical_pixel_pitch_um=s.logical_pixel_pitch_um,
                active_size=s.active_size,distance_m=s.detector_distance_m,wavelength_nm=s.wavelength_nm,
                npz_sha256=sha(a.out/'reference.npz'),dataset_root=str(s.dataset_root),model_source_sha256=sha(src/'modeling.py'),
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                simulation_accuracy=float(np.mean([r['simulation_prediction']==r['label'] for r in rows])))
    (a.out/'reference.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
