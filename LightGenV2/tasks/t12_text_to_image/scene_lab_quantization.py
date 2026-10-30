"""CPU-only propagation audit of exported DEV example fields; no Qwen/SDK/TEST."""
import argparse
import json
from pathlib import Path
import numpy as np
from .scene_lab_protocol import encode,sha,put


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--boundary-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import torch
    from torch.nn import functional as F
    from .audited_unified import optical_path
    torch.set_num_threads(2);path=optical_path(224,224).eval();prop=path.core.propagator
    receipt=json.loads((a.boundary_run/'boundary_fields_download_receipt.json').read_text());rows=[]
    with torch.inference_mode():
        for rel,record in receipt['files'].items():
            file=a.boundary_run/rel
            if sha(file)!=record['sha256']:raise ValueError('Boundary field changed')
            with np.load(file,allow_pickle=False)as z:
                amplitude,phase,ampcode,phasecode=encode(z['amplitude'],z['total_phase_radians']);ideal=z['ideal_CCD'].copy()
            amplitude=torch.from_numpy(amplitude);phase=torch.from_numpy(phase)
            original=F.pad(amplitude*torch.exp(1j*phase),(20,20,20,20))
            quant=F.pad(torch.from_numpy(ampcode.astype(np.float32)/255)*torch.exp(2j*torch.pi*torch.from_numpy(phasecode.astype(np.float32)/256)),(20,20,20,20))
            expected=prop(original).abs().square()[:,20:498,20:498].numpy();observed=prop(quant).abs().square()[:,20:498,20:498].numpy()
            reference_scale=max(float(np.mean(ideal.astype(float)**2)),1e-20)
            parity=float(np.sqrt(np.mean((expected.astype(float)-ideal)**2)/reference_scale))
            if parity>1e-3:raise ValueError(('Export does not reconstruct original propagation',rel,parity))
            rows.append(dict(path=rel,source_sha256=record['sha256'],continuous_reconstruction_NRMSE=parity,
                quantized_CCD_NRMSE=float(np.sqrt(np.mean((observed.astype(float)-ideal)**2)/reference_scale)),
                relative_mean_intensity_change=float((observed.mean()-ideal.mean())/max(float(ideal.mean()),1e-20))))
    if len(rows)!=48:raise ValueError('Require all48 checked examples')
    put(a.out,dict(complete=True,CPU_only=True,SDK_used=False,GPU_used=False,TEST_used=False,n=48,rows=rows,
                  distance_m=prop.distance_m,grid_size=prop.grid_size,
                  boundary_manifest_sha256=sha(a.boundary_run/'boundary_fields_download_receipt.json'),
                  physical_or_generation_quality_gate=False,note='Local intensity quantization diagnostic, not end-to-end Qwen/SDXL quality or real camera acceptance'))
    print(json.dumps(dict(passed=True,n=48,max_continuous_reconstruction_NRMSE=max(r['continuous_reconstruction_NRMSE']for r in rows),max_quantized_CCD_NRMSE=max(r['quantized_CCD_NRMSE']for r in rows),distance_m=prop.distance_m,physical_or_generation_quality_gate=False)))


if __name__=='__main__':main()
