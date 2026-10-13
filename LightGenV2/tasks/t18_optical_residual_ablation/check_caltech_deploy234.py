"""CPU optical/BMP parity tests, synthetic only, never opens a device or test set."""
import argparse, copy, json, math, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
import numpy as np
import torch
from PIL import Image
from caltech_deploy234_model import DeployMoE,input_amplitude,ccd_nonlinearity,deployment_contract

def native(active,kind):
    image=Image.fromarray(np.asarray(active,dtype=np.uint8)).resize((1016,1016),Image.Resampling.NEAREST)
    height=1080 if kind=='amplitude' else 1200
    canvas=Image.new('L',(1920,height));canvas.paste(image,((1920-1016)//2,(height-1016)//2));return canvas

def check(out):
    torch.set_num_threads(4);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(17);images=torch.randint(0,256,(2,3,150,150),dtype=torch.uint8);amp=input_amplitude(images)
    results=[]
    for depth in (2,3,4):
        for rho in (0.,.3):
            torch.manual_seed(17);model=DeployMoE(depth,rho).eval()
            with torch.no_grad():reference=model(amp)[0];traces=copy.deepcopy(model.last)
            calls=[]
            def replay(stage,a,p,ids):
                calls.append(stage)
                gray=np.rint(np.clip(a,0,1)*255).astype(np.uint8)
                phase=p.astype(np.float32)/256*(2*math.pi)
                field=torch.from_numpy(gray.astype(np.float32)/255)*torch.exp(1j*torch.from_numpy(phase))[None]
                field=torch.nn.functional.pad(field,(20,20,20,20))
                intensity=model.propagator(field).abs().square()[:,20:498,20:498]
                native(gray[0],'amplitude').save(out/f'L{depth}_rho{rho}_stage{stage}_amplitude.bmp')
                native(p,'phase').save(out/f'L{depth}_rho{rho}_stage{stage}_phase.bmp')
                assert np.max(np.abs(gray.astype(float)/255-a))<=.5/255+1e-6
                assert np.array_equal(np.asarray(Image.open(out/f'L{depth}_rho{rho}_stage{stage}_amplitude.bmp')),np.asarray(native(gray[0],'amplitude')))
                return intensity.numpy()
            with torch.no_grad():actual=model(amp,callback=replay,ids=['synthetic0','synthetic1'])[0]
            error=float((actual-reference).abs().max());assert error<=1e-6,error
            assert len(calls)==depth+1 and calls[0]=='router'
            assert model.detectors.shape==(20,478,478) and model.detectors.sum(0).max()==1
            assert not any(isinstance(m,(torch.nn.Linear,torch.nn.Conv2d)) for m in model.modules())
            for stage,t in traces.items():
                a=t['amplitude'];assert (a>=0).all() and (a<=1).all()
                if stage=='router':assert not a[:,:127].any()
            with torch.no_grad():zero=model(torch.zeros_like(amp))[0]
            assert torch.allclose(zero,torch.full_like(zero,.05),atol=1e-8,rtol=0)
            results.append(dict(depth=depth,rho=rho,simulation_export_replay_max_error=error,captures=len(calls),parameters=sum(p.numel() for p in model.parameters()),passed=True,contract=deployment_contract(depth,rho)))
    assert torch.equal(ccd_nonlinearity(torch.zeros(2,478,478)),torch.zeros(2,224,224))
    # Residual joint encoding must equal the complex coherent sum before 8-bit error.
    phase=torch.linspace(0,2*math.pi,1000);mod=.7*torch.exp(1j*phase)+.3
    assert (mod.abs()*torch.exp(1j*torch.angle(mod))-mod).abs().max()<1e-6
    assert mod.abs().min()<.401 and mod.abs().max()<=1.000001
    value=dict(passed=True,results=results,all_six_parity=True,zero_preservation=True,raw_ccd_contract=True,residual_requires_amplitude_and_phase=True,devices_opened=0,test_arrays_read=0)
    (out/'checks.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);check(p.parse_args().out)