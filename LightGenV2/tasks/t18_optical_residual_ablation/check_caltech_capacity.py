"""Initial physical equivalence and fresh perturbed-mask native CPU replay."""
import argparse, math, os, subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import caltech_capacity as c
from caltech_capacity_model import CapacityMoE, GRIDS, COUNTS, input_amplitude
import export_caltech_capacity as e
import export_caltech_repair234 as physical_export
from check_caltech_repair234_export import reverse, rejected

def initial_equality():
    models=[]
    for grid in GRIDS:
        torch.manual_seed(17);models.append(CapacityMoE(4,.3,grid).eval())
    assert all(torch.equal(m.raw_router,models[0].raw_router) for m in models)
    assert all(not p.any() for m in models for p in m.phase_sets)
    torch.manual_seed(20261014);amplitude=torch.rand(1,224,224);global_amplitude=torch.rand(1,478,478)
    weights=torch.tensor([[.2,.3,.4,.5]]);weights=weights/weights.square().sum(1,keepdim=True).sqrt()
    field_sha={}
    with torch.no_grad():
        for stage in ['router','0','1','2','3']:
            a=global_amplitude if stage in ('1','3') else amplitude;w=None if stage=='router' else weights
            fields=[m.physical_fields(a,stage,w) for m in models]
            assert all(all(torch.equal(u,v) for u,v in zip(fields[0],f)) for f in fields[1:]),stage
            assert all(torch.equal(models[0].plane(stage),m.plane(stage)) for m in models[1:])
            field_sha[stage]=c.digest(fields[0][0].numpy())
        p=[m(amplitude)[0] for m in models];assert all(torch.equal(p[0],v) for v in p[1:])
    assert all(sum(p.numel() for p in m.parameters())==COUNTS[m.phase_grid] for m in models)
    # For identical represented physical phase, regularization has the same value.
    for m in models:
        with torch.no_grad():
            for p in m.phase_sets:p.fill_(.125)
    assert all(torch.equal(models[0].smoothness(),m.smoothness()) for m in models[1:])
    return dict(passed=True,initial_physical_field_sha256=field_sha,initial_output_exact=True,
                router_sha256=c.digest(models[0].raw_router.detach().numpy()),parameters=COUNTS,
                physical_apertures_unchanged=True,physical_smoothness_equal=True,model_forward_calls=3)

def native_probes(model,out,src):
    out.mkdir(parents=True,exist_ok=False);grid=model.phase_grid;cp=out/'synthetic_phase.pt'
    torch.save(dict(model=model.state_dict(),scope='Perturbed synthetic mask, not fit or selected weight'),cp)
    cfg=dict(depth=4,residual_rho=.3,phase_grid=grid,scope='Synthetic native contract')
    package=out/'static';manifest=e.export_selected(model,cfg,package,cp,c.sha(cp),[f'synthetic{k}' for k in range(20)],src)
    assert manifest['contract']['phase_grid']==grid and manifest['physical_main_shapes']==[list(s) for s in model.physical_shapes]
    for stage in manifest['stages']:
        s=stage['stage'];assert np.array_equal(reverse(package/f'phase_{s}_native.bmp','phase'),np.asarray(Image.open(package/f'phase_{s}_canonical.bmp')))
    with np.load(package/'amplitude_multipliers.npz') as z:
        for stage in manifest['stages']:
            s=stage['stage'];assert np.array_equal(z[s],model.plane(s).detach().abs().numpy()[20:498,20:498])
    rejected(lambda:physical_export.encode_stage(model,torch.zeros(1,224,224),'1',torch.full((1,4),.5)))
    rejected(lambda:physical_export.encode_stage(model,torch.zeros(1,478,478),'0',torch.full((1,4),.5)))
    rejected(lambda:physical_export.encode_stage(model,torch.zeros(1,224,224),'0',torch.tensor([[1.,0.,0.,0.]])))
    rejected(lambda:physical_export.amplitude_from_ccd(model,torch.full((1,478,478),float('nan')),'global'))
    torch.manual_seed(20261014);probes={'random':torch.randint(0,256,(1,3,150,150),dtype=torch.uint8),
                                     'dim':torch.full((1,3,150,150),4,dtype=torch.uint8),
                                     'black':torch.zeros(1,3,150,150,dtype=torch.uint8)}
    checks=[]
    for name,image in probes.items():
        amp=input_amplitude(image)
        with torch.no_grad():reference=model(amp)[0]
        current=[amp];weights=[None];calls=[];stats=[]
        def replay(stage,a,p,ids):
            encoded,phase=physical_export.encode_stage(model,current[0],stage,weights[0])
            assert np.array_equal(encoded,np.rint(a*255).astype(np.uint8)) and np.array_equal(phase,p)
            dest=out/(name+'_'+stage);receipt=physical_export.save_sample_bmps(model,current[0],stage,dest,weights[0])
            qa=reverse(dest/'amplitude_000000_native.bmp','amplitude');qp=reverse(dest/'phase_native.bmp','phase')
            assert np.array_equal(qa,encoded[0]) and np.array_equal(qp,phase)
            calls.append(stage);stats.append(receipt['brightness'][0])
            field=torch.from_numpy(qa.astype(np.float32)/255)[None]*torch.exp(2j*math.pi*torch.from_numpy(qp.astype(np.float32)/256))[None]
            field=torch.nn.functional.pad(field,(20,20,20,20));intensity=model.propagator(field).abs().square()[:,20:498,20:498]
            if stage=='router':weights[0]=physical_export.routing_from_ccd(model,intensity)
            elif int(stage)<3:current[0]=physical_export.amplitude_from_ccd(model,intensity,model.stages[int(stage)+1])
            return intensity.numpy()
        with torch.no_grad():actual=model(amp,callback=replay,ids=[name])[0]
        error=float((actual-reference).abs().max());assert error<=1e-6 and calls==['router','0','1','2','3']
        assert torch.equal(weights[0],model.routing_weights)
        if name=='black':assert all(s['active']['maximum']==0 for s in stats) and torch.allclose(actual,torch.full_like(actual,.05),atol=1e-8,rtol=0)
        checks.append(dict(phase_grid=grid,probe=name,passed=True,captures=5,probability_max_error=error,
                           native_inverse_exact=True,black_amplitude_zero=(name=='black'),brightness=stats))
    return checks

def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists()
    src=c.sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip();c.engine.verify_sources(src,commit)
    torch.set_num_threads(4);out.mkdir(parents=True,exist_ok=False)
    try:
        equality=initial_equality();checks=[]
        for grid in GRIDS:
            torch.manual_seed(17);model=CapacityMoE(4,.3,grid).eval()
            with torch.no_grad():
                for p in model.phase_sets:p.uniform_(-.4,.4)
            checks.extend(native_probes(model,out/grid,src))
        c.save(out/'checks.json',dict(passed=True,initial_equality=equality,checks=checks,sources=src,git_commit=commit,
              synthetic_static_packages=3,model_forward_calls=21,test_arrays_read=0,test_evaluations=0,devices_opened=0,gpu_used=False,
              scope='Initial physical equality plus fresh perturbed native contract; not selected trained weights or performance'),exclusive=True)
    except BaseException as x:c.save(out/'failure.json',dict(error=repr(x),automatic_retry=False),exclusive=True);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);check(p.parse_args().out)
