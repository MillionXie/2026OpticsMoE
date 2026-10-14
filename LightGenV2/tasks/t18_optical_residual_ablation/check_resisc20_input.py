"""Fresh CPU input/native proofs for the new controls; no dataset or test."""
import argparse,math,os,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import resisc20 as s
import resisc20_input_study as study
import resisc20_mask_batch as batch
import export_resisc20_input as export
import export_caltech_repair234 as e
from check_caltech_repair234_export import reverse,rejected
from caltech_deploy234_model import rms_encode

def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists()
    torch.set_num_threads(4);batch.install();src=study.sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    torch.manual_seed(20261014);random=torch.randint(0,256,(1,3,150,150),dtype=torch.uint8)
    theta=torch.tensor([[[.98,.1,.03],[-.1,.98,-.04]]])
    assert torch.equal(study.encode(random,None,50),s.r.input_amplitude(random,None))
    assert torch.equal(study.encode(random,theta,50),s.r.input_amplitude(random,theta))
    x=F.interpolate(random.float()/255,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    mosaic=torch.cat([torch.cat([x[:,0:1],x[:,1:2]],3),torch.cat([x[:,2:3],x.mean(1,keepdim=True)],3)],2)
    reference100=rms_encode(F.interpolate(mosaic,(224,224),mode='bilinear',align_corners=False)[:,0])
    assert torch.equal(study.encode(random,None,100),reference100)
    probes={'random':random,'dim':torch.full_like(random,4),'black':torch.zeros_like(random)}
    out.mkdir(parents=True,exist_ok=False);checks=[];initial=[]
    for case,(tile,rho) in study.CASES.items():
        torch.manual_seed(17);model=s.r.CorrectedMoE(4,rho).eval();initial.append(s.digest(model.state_dict()))
        cfg=study.config(case);cfg.update(depth=4,residual_rho=rho);cp=out/(case+'.pt');torch.save(dict(model=model.state_dict(),scope='Fresh synthetic only'),cp)
        package=out/(case+'_static');m=export.export_selected(model,cfg,package,cp,e.sha(cp),[f'synthetic{k}' for k in range(20)],src)
        assert m['contract']['input_channel_tile']==tile and m['contract']['input_network_aperture_unchanged']==224
        for stage in m['stages']:
            from PIL import Image
            assert np.array_equal(reverse(package/f"phase_{stage['stage']}_native.bmp",'phase'),np.asarray(Image.open(package/f"phase_{stage['stage']}_canonical.bmp")))
        with np.load(package/'amplitude_multipliers.npz') as z:
            assert all(np.array_equal(z[k],model.plane(k).detach().abs().numpy()[20:498,20:498]) for k in z.files)
        rejected(lambda:e.encode_stage(model,torch.zeros(1,224,224),'1',torch.full((1,4),.5)))
        rejected(lambda:e.amplitude_from_ccd(model,torch.full((1,478,478),float('nan')),'global'))
        for name,image in probes.items():
            amp=study.encode(image,None,tile);assert torch.isfinite(amp).all()
            with torch.no_grad():reference=model(amp)[0]
            current=[amp];weights=[None];calls=[];brightness=[]
            def replay(stage,a,p,ids):
                encoded,phase=e.encode_stage(model,current[0],stage,weights[0]);assert np.array_equal(encoded,np.rint(a*255).astype(np.uint8)) and np.array_equal(phase,p)
                dest=out/(case+'_'+name+'_'+stage);receipt=e.save_sample_bmps(model,current[0],stage,dest,weights[0])
                qa=reverse(dest/'amplitude_000000_native.bmp','amplitude');qp=reverse(dest/'phase_native.bmp','phase')
                assert np.array_equal(qa,encoded[0]) and np.array_equal(qp,phase);calls.append(stage);brightness.append(receipt['brightness'][0])
                field=torch.from_numpy(qa.astype(np.float32)/255)[None]*torch.exp(2j*math.pi*torch.from_numpy(qp.astype(np.float32)/256))[None]
                intensity=model.propagator(F.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498]
                if stage=='router':weights[0]=e.routing_from_ccd(model,intensity)
                elif int(stage)<3:current[0]=e.amplitude_from_ccd(model,intensity,model.stages[int(stage)+1])
                return intensity.numpy()
            with torch.no_grad():actual=model(amp,callback=replay,ids=[name])[0]
            error=float((reference-actual).abs().max());assert error<=1e-6 and calls==['router','0','1','2','3']
            assert torch.equal(weights[0],model.routing_weights)
            if name=='black':assert all(v['active']['maximum']==0 for v in brightness)
            checks.append(dict(case=case,input_tile=tile,rho=rho,probe=name,probability_max_error=error,
                captures=5,native_disk_reverse_exact=True,ccd_routing_helpers_match=True,black_zero=(name!='black' or all(v['active']['maximum']==0 for v in brightness))))
    assert len(set(initial))==1
    s.save(out/'checks.json',dict(passed=True,checks=checks,sources=src,git_commit=commit,phase_initial_sha256=initial[0],
        tile50_original_exact=True,tile100_independent_recipe_exact=True,static_packages=3,model_forward_calls=18,
        dataset_arrays_read=0,test_arrays_read=0,test_evaluations=0,devices_opened=0,gpu_used=False,
        scope='Fresh synthetic input-aware offline contract, not selected trained weights or bench acceptance'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:check(a.out)
    except BaseException as e:
        if a.out.exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
