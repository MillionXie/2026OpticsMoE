"""Numerical replay test only, NEVER real acquisition evidence."""
import numpy as np
from PIL import Image
from common import ROOT,STAGES,write,setup_imports

def main():
    from backend import create,forward,optical_contract
    from run import samples
    setup_imports()
    from abo_dual.backend import NeedCapture
    from abo_dual.common import routing_from_ccd
    import torch
    torch.set_num_threads(4)
    b=create('cpu'); contract=optical_contract(b)
    sample=next(x for x in samples(1) if x['kind']=='image')
    reference=forward(b,sample); measured={}; detectors={}
    for name,branch in b.branches.items():
        for stage,x in [('router',branch.core.router.last_detector_intensity),
                        ('expert',branch.last_raw_expert_ccd),('global',branch.last_raw_ccd)]:
            arr=x[0].detach().float().cpu().numpy().copy(); detectors[name+'_'+stage]=arr
            measured[name+'_'+stage]=routing_from_ccd(arr,contract) if stage=='router' else arr
    b.guard_optics(); supplied={}; seen=[]
    for stage in STAGES:
        try: forward(b,sample,supplied)
        except NeedCapture as e:
            if e.stage!=stage: raise AssertionError((stage,e.stage))
            seen.append(stage); supplied[stage]=measured[stage]
        else: raise AssertionError('Optical guard failed: '+stage)
    replay=forward(b,sample,supplied)
    err=float(np.max(np.abs(reference-replay)))
    # Tiny summation differences are allowed, but not any substituted stage.
    if err>2e-4: raise AssertionError(f'Numerical replay changed embedding by {err}')
    out=ROOT/'results/replay_check'; out.mkdir(parents=True,exist_ok=True)
    for stage,x in detectors.items():
        np.save(out/(stage+'.npy'),x)
        scale=max(float(np.percentile(x,99.5)),1e-12)
        Image.fromarray(np.rint(np.clip(x/scale,0,1)*255).astype(np.uint8)).save(out/(stage+'.png'))
    write(out/'report.json',{'mode':'simulation_boundary_unit_test_NOT_CCD','all_six_guards_verified':seen,
        'maximum_embedding_error':err,'sample_id':sample['id'],
        'preview_only':'PNG percentile99.5 linear display; npy is original intensity; never used as measured capture.'})
    print('Verified six sequential optical boundaries; maximum embedding error:',err)

if __name__=='__main__': main()
