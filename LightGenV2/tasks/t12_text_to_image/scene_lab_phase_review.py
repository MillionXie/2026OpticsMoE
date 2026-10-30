"""CPU-only original-propagator comparison of calibration phase ramps, not a fit."""
import argparse
import json
from pathlib import Path
import numpy as np
from .scene_lab_protocol import put
from .scene_lab_hardware import centroid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--physical-report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import torch
    from torch.nn import functional as F
    from .audited_unified import optical_path
    torch.set_num_threads(2);path=optical_path(224,224).eval();prop=path.core.propagator
    yy,xx=np.indices((478,478));amplitude=np.zeros((478,478),np.float32);amplitude[217:261,217:261]=1
    def simulate(code):
        field=torch.from_numpy(amplitude)[None]*torch.exp(2j*torch.pi*torch.from_numpy(code.astype(np.float32)/256)[None])
        with torch.inference_mode():return prop(F.pad(field,(20,20,20,20))).abs().square()[0,20:498,20:498].numpy()*20000
    bg=np.zeros_like(amplitude);origin=centroid(simulate(np.zeros_like(amplitude,dtype=np.uint8)),bg);rows=[]
    observed=json.loads(a.physical_report.read_text())
    for row,(name,axis,sign)in zip(observed['rows'],[('x_plus',xx,1),('x_minus',xx,-1),('y_plus',yy,1),('y_minus',yy,-1)]):
        if row['name']!=name:raise ValueError('Ramp order changed')
        code=np.floor(np.remainder(sign*axis*32/478,1)*256).astype(np.uint8);delta=centroid(simulate(code),bg)-origin
        rows.append(dict(name=name,original_simulation_delta_px=delta.tolist(),measured_delta_px=row['delta']))
    put(a.out,dict(complete=True,CPU=True,SDK=False,GPU=False,distance_fitted=False,phase_LUT_changed=False,model_changed=False,
        original_propagator_distance_m=float(prop.distance_m),original_grid_size=int(prop.grid_size),rows=rows,
        conclusion='Direction and finite ramp response only. A magnitude discrepancy cannot uniquely separate distance, phase-depth/diffraction-order and other optical effects. Not full phase linearity or physical SIM-EXP quality acceptance.'))
    print(json.dumps(rows))


if __name__=='__main__':main()
