"""Portable fixed SALICON release inspection/export. Never connects to hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['verify','simulate','export-reference-bmp'])
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--fields',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--profile',choices=['meadowlark17','shs8'],default='meadowlark17')
    p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.project.resolve()
    sys.path.insert(0,str(root/'runtime'))
    manifest=json.loads((root/'SHA256.json').read_text(encoding='utf-8'))
    if a.action=='verify':
        for name,expected in manifest.items():
            file=(root/name).resolve()
            if not file.is_relative_to(root):raise ValueError('Unsafe manifest path')
            if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('SHA mismatch: '+name)
        print('VERIFIED',len(manifest),'files');return
    import numpy as np
    import torch
    from PIL import Image
    from LightGenV2.tasks.t03_saliency.lab_runtime import load_model,replay,phase_planes,STAGES
    from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import raster
    from LightGenV2.tasks.t03_saliency.reproduce_baseline import independent_cc
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import density_from_logits
    if a.fields<0:raise ValueError('fields must be >=0; 0 means all 5000')
    out=a.output or root/('simulation_check' if a.action=='simulate' else 'reference_bmp_'+a.profile)
    if out.exists():raise FileExistsError('Use a new output directory')
    torch.set_num_threads(4)
    model=load_model(root,a.device)
    release=json.loads((root/'release.json').read_text())
    items=release['fields'][:a.fields] if a.fields else release['fields']
    out.mkdir(parents=True)
    rows=[]
    if a.action=='export-reference-bmp':
        amp={'pixel_pitch_um':17. if a.profile=='meadowlark17' else 8.,
             'size_wh':[1024,1024] if a.profile=='meadowlark17' else [1920,1080],
             'center_xy':[512,512] if a.profile=='meadowlark17' else [960,540]}
        phase={'pixel_pitch_um':8.,'size_wh':[1920,1200],'center_xy':[960,600]}
        for stage,plane in phase_planes(model).items():
            folder=out/stage;folder.mkdir()
            Image.fromarray(raster(plane,phase,'phase')).save(folder/'phase.bmp')
    for item in items:
        source=root/item['file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Cache changed')
        batch=torch.load(source,map_location='cpu',weights_only=False)
        logits,tap=replay(model,batch)
        density=density_from_logits(logits).cpu().numpy()
        cc=float(independent_cc(density,batch['density'].numpy())[0])
        rows.append(dict(key=item['key'],sample_id=item['sample_id'],cc=cc,reference_cc=item['simulation_cc']))
        if a.action=='export-reference-bmp':
            for stage in STAGES:
                Image.fromarray(raster(tap.amplitudes[stage][0].cpu().numpy(),amp,'amplitude')).save(out/stage/(item['key']+'.bmp'))
        if len(rows)<=4:
            for stage in STAGES:
                x=tap.detectors[stage][0].cpu().numpy()
                np.save(out/(item['key']+'_'+stage+'.npy'),x)
                # Display only; no gamma or image-dependent scaling is used in metrics.
                Image.fromarray(np.rint(np.clip(x/max(float(np.percentile(x,99.5)),1e-12),0,1)*255).astype('uint8')).save(out/(item['key']+'_'+stage+'.png'))
        if len(rows)%100==0:print('EVALUATED',len(rows),flush=True)
    report=dict(samples=len(rows),cc_float64=float(np.mean([r['cc'] for r in rows])),
                reference_same_subset=float(np.mean([r['reference_cc'] for r in rows])),rows=rows,
                reference_only=True,hardware_configuration_verified=False,
                warning='Reference BMPs use CENTERED panels, NO flips and NO phase gray inversion. Recalibrate hardware. Later-stage reference amplitudes use SIMULATED upstream CCD, not a measured closed loop.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
