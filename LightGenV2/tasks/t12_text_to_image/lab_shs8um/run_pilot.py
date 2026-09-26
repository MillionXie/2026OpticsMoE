"""Six fixed image/text edits through real language and vision CCDs."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--abo-project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--inputs',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.abo_project/'lab_dvp8um'))
    from LightGenV2.tasks.t12_text_to_image.sealed_editor import build_sealed
    from LightGenV2.tasks.t12_text_to_image.audited_unified import architecture_report
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.ccd_bridge import attach
    import four_image_flow as flow
    from shs_physical2400 import SHSBench,CORNERS
    contract=json.loads((a.project/'assets/contract.json').read_text())
    assert flow.sha(a.project/'assets/small.pt')==contract['checkpoint_sha256']
    saved=torch.load(a.project/'assets/small.pt',map_location='cpu',weights_only=False)
    model=build_sealed(saved).cuda().eval().requires_grad_(False)
    assert architecture_report(model)['counted_parameters']==9958098
    input_path=a.inputs or a.project/'assets/pilot_inputs.pt'
    x=torch.load(input_path,map_location='cpu',weights_only=False)
    values=[x[k].cuda() for k in ['reference','embeddings','mask','noise']]
    values[1]=values[1].float()
    with torch.inference_mode():baseline=model(*values)
    phase_dir=a.output/'phase';phase_dir.mkdir()
    ids=[f'pilot_{i:03d}' for i in range(len(values[0]))]
    stages=[]
    def convert(image):
        return image.detach().float().cpu().add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).numpy()
    def callback(stage,amplitude,phase,ideal):
        stages.append(stage)
        if a.selftest:return ideal
        if float((phase-phase[:1]).abs().max())>1e-5:raise ValueError('Per-sample phase is not deployable with one shared mask')
        native=flow.phase_gray(phase[0].detach().cpu().numpy(),'hv',True)
        path=phase_dir/(stage+'.bmp');Image.fromarray(native).save(path)
        paths,scale=flow.save_amplitudes(amplitude.detach().cpu().numpy(),a.output,stage,ids)
        try:raw,receipt=bench.capture(stage,path,paths,ids,'flip_v')
        finally:
            for bmp in paths:bmp.unlink(missing_ok=True)
        print(json.dumps(dict(stage=stage,status='captured',amplitude_scale=scale)),flush=True)
        return torch.from_numpy(raw).cuda().float()
    flow.BASE_CORNERS=CORNERS.copy()
    if a.selftest:
        restore=attach(model,callback)
        with torch.inference_mode():actual=model(*values)
        restore();error=float((actual-baseline).abs().max())
        assert error<1e-6,error
    else:
        with SHSBench(a.output,400,240,{}) as bench:
            restore=attach(model,callback)
            try:
                with torch.inference_mode():actual=model(*values)
            finally:restore()
    assert stages==['language_router','language_expert','language_global','vision_router','vision_expert','vision_global']
    metrics=dict(physical_mse_to_target=float(F.mse_loss(actual.cpu(),x['target'])),
                 simulation_mse_to_target=float(F.mse_loss(baseline.cpu(),x['target'])),
                 physical_mse_to_simulation=float(F.mse_loss(actual,baseline)))
    for i,s in enumerate(ids):
        for label,tensor in [('reference',x['reference']),('target',x['target']),('simulation',baseline),('physical',actual)]:
            Image.fromarray(convert(tensor[i])).save(a.output/(s+'_'+label+'.png'))
    torch.save(dict(actual=actual.cpu(),simulation=baseline.cpu()),a.output/'outputs.pt')
    flow.write(a.output/'report.json',dict(status='complete',selftest=a.selftest,metrics=metrics,
        sample_count=len(ids),sample_metadata=x['metadata'],stages=stages,contract=contract,
        scope=x.get('scope','fixed pilot; not full test-set performance'),indices=x.get('indices'),corners=CORNERS.tolist()))
    from LightGenV2.tasks.t12_text_to_image.lab_shs8um.export_samples import export
    export(a.project,a.output,input_path)
    print(json.dumps(dict(status='complete',metrics=metrics)),flush=True)


if __name__=='__main__':main()
