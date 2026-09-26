"""Network-pattern exposure scan; ideal upstream fields are diagnostics only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .ccd_bridge import attach
from ..sealed_editor import build_sealed

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--abo-project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    sys.path.insert(0,str(a.abo_project/'lab_dvp8um'))
    import four_image_flow as flow
    from shs_physical2400 import SHSBench,CORNERS
    flow.BASE_CORNERS=CORNERS.copy()
    model=build_sealed(torch.load(a.project/'assets/small.pt',map_location='cpu',weights_only=False)).cuda().eval()
    x=torch.load(a.project/'assets/pilot_inputs.pt',map_location='cpu',weights_only=False)
    selected={}
    def callback(stage,amplitude,phase,ideal):
        if stage.endswith('router'):
            phase_path=a.output/(stage+'_phase.bmp')
            Image.fromarray(flow.phase_gray(phase[0].detach().cpu().numpy(),'hv',True)).save(phase_path)
            ids=['network_0','network_1','network_2']
            if getattr(model,'bounded_amplitude',None):
                from .bounded_amplitude import save_bmps
                paths,scale=save_bmps(amplitude[:3].detach().cpu().numpy(),a.output,stage,ids,flow)
            else:
                paths,scale=flow.save_amplitudes(amplitude[:3].detach().cpu().numpy(),a.output,stage,ids)
            selected[stage]=(phase_path,paths,ids,scale)
        return ideal
    restore=attach(model,callback)
    values=[x[k].cuda() for k in ('reference','embeddings','mask','noise')];values[1]=values[1].float()
    with torch.inference_mode():model(*values)
    restore();del model;torch.cuda.empty_cache()
    dark=a.output/'dark.bmp';Image.fromarray(flow.active_to_native(np.zeros((478,478),np.uint8))).save(dark)
    summaries=[]
    for exposure in (400,1000,2000,4000):
        root=a.output/f'exposure_{exposure}'
        root.mkdir()
        with SHSBench(root,exposure,240,{}) as bench:
            for stage,(phase,paths,ids,scale) in selected.items():
                raw,_=bench.capture(stage,phase,[dark]+paths+paths,['dark']+ids+[s+'_repeat' for s in ids],'flip_v')
                background=raw[0].astype(float)
                rows=[]
                for i,sid in enumerate(ids):
                    signal=raw[i+1].astype(float)-background
                    rows.append(dict(sample_id=sid,p99=float(np.percentile(raw[i+1],99)),
                        signal_p99=float(np.percentile(signal,99)),signal_mean=float(signal.mean()),
                        repeat_pcc=flow.pcc(raw[i+1],raw[i+4]),
                        saturation_fraction=float((raw[i+1]>=255).mean())))
                summaries.append(dict(exposure_us=exposure,stage=stage,amplitude_scale=scale,
                    dark_mean=float(background.mean()),dark_p99=float(np.percentile(background,99)),images=rows))
                flow.write(a.output/'report.json',dict(status='running',scope='router-only diagnostic using simulated upstream vision inputs; NOT end-to-end task quality',rows=summaries))
    flow.write(a.output/'report.json',dict(status='complete',scope='router-only exposure diagnostic, NOT task metrics; no formal exposure changed',rows=summaries))

if __name__=='__main__':main()
