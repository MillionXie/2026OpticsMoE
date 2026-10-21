"""TRAIN-only diagnostic: logical gray versus actual exporter complement.

The exporter writes 255-flipXY(gray); the fitted observation approximation
used logical gray. Compare the two gray-coordinate hypotheses, keeping the
established geometric convention fixed. Neither is a measured phase LUT.
No optimization, TEST, camera, altered BMP, mask or deployment release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def device_gray_turns(turns):
    return 255/256-turns


def run(a):
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_coherent_response_20261021 as response
    torch.set_num_threads(2)
    assert sha(response.__file__)=='c08da807054fb5716bd6c36923218ee192870cb8e1959f6d7732c8f6d9155cc6'
    assert not a.out.exists();a.out.mkdir()
    manifest=json.loads(a.manifest.read_text());observations=a.manifest.parent/manifest['file']
    assert sha(observations)==manifest['SHA256']=='c806c13cec4f3641952e71c18d17e010410690e80cf749a21d777537301a979d'
    fit=a.manifest.parent/'fit_GPU2_ownership_recovery_20261021'
    report=json.loads((fit/'report.json').read_text());assert report['state']=='complete' and not report['response_admitted']
    parameters=fit/'best_response.npz';assert sha(parameters)==report['response_PT_SHA256']
    raw=response.initial('cpu')
    with np.load(parameters) as z:raw.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    with np.load(observations) as z:
        amp=torch.from_numpy(z['amplitude'].astype(np.float32)/255)
        turns=torch.from_numpy(z['phase'].astype(np.float32)/256)
        observed=torch.from_numpy(z['observed'].astype(np.float32));transfer=torch.from_numpy(z['transfer'].copy())
        fitmask=z['fit'].copy();depth=z['depth'].copy()
    protocol=dict(source_SHA256=sha(__file__),response_source_SHA256=sha(response.__file__),
        observations_SHA256=sha(observations),parameters_SHA256=sha(parameters),
        exact_gray_identity='device=255-logical before established geometric flips',
        comparisons=['logical_turns','actual_export_gray_turns'],optimizer_updates=0,
        fit_TRAIN_records=216,other_TRAIN_records=80,TEST_used=False,VALID=0,
        SDK_used=False,GPU_used=False,geometry_LUT_exposure_unchanged=True,
        actual_phase_sign_not_identified_by_source=True,no_automatic_mask_or_capture_release=True)
    save(a.out/'protocol.json',protocol);results={};started=time.monotonic()
    with torch.no_grad():
        for label,phase in [('logical_turns',turns),('actual_export_gray_turns',device_gray_turns(turns))]:
            rows=[]
            for begin in range(0,len(amp),4):
                p=response.response(amp[begin:begin+4],phase[begin:begin+4],transfer,raw);o=observed[begin:begin+4]
                for j in range(len(p)):
                    pixel=torch.stack([p[j],o[j]]).flatten(1);pixel-=pixel.mean(1,keepdim=True)
                    coarse=F.adaptive_avg_pool2d(torch.stack([p[j],o[j]])[:,None],(16,16)).flatten(1);coarse-=coarse.mean(1,keepdim=True)
                    corr=lambda z:float((z[0]*z[1]).sum()/(z[0].norm()*z[1].norm()).clamp_min(1e-12))
                    rows.append(dict(index=begin+j,normalized_loss=float(response.normalized_loss(p[j:j+1],o[j:j+1])),pixel_PCC=corr(pixel),coarse_PCC=corr(coarse)))
                save(a.out/'progress.json',dict(state='TRAIN_export_coordinate_review',hypothesis=label,records=len(rows),expected=len(amp)))
            results[label]={}
            for role,mask in [('fit_TRAIN',fitmask),('other_TRAIN',~fitmask)]:
                for d in (4,6):
                    subset=[r for r in rows if mask[r['index']] and depth[r['index']]==d]
                    results[label][f'L{d}_{role}']=dict(records=len(subset),loss=float(np.mean([r['normalized_loss'] for r in subset])),pixel_PCC=float(np.median([r['pixel_PCC'] for r in subset])),coarse_PCC=float(np.median([r['coarse_PCC'] for r in subset])))
            save(a.out/(label+'_records.json'),rows)
    save(a.out/'report.json',dict(state='complete',results=results,optimizer_updates=0,
        elapsed_s=time.monotonic()-started,TEST_used=False,SDK_used=False,GPU_used=False,
        no_classifier_accuracy_claim=True,new_mask_generated=False,next_capture_started=False,
        actual_phase_LUT_not_calibrated=True,protocol_SHA256=sha(a.out/'protocol.json')))
    print(json.dumps(dict(complete=True,results=results,new_mask_generated=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
