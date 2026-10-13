"""CPU native BMP roundtrip for every repaired depth/rho; synthetic inputs only."""
import argparse,math,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from PIL import Image
from caltech_repair234_model import CorrectedMoE,input_amplitude,deployment_contract
from check_caltech_deploy234 import native
from caltech_repair234 import save,sha,sources,engine

def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    try:
        torch.manual_seed(20261014);images=[('random',torch.randint(0,256,(1,3,150,150),dtype=torch.uint8)),('dim',torch.full((1,3,150,150),4,dtype=torch.uint8)),('black',torch.zeros((1,3,150,150),dtype=torch.uint8))];checks=[]
        for depth in (2,3,4):
            for rho in (0.,.3):
                torch.manual_seed(17);model=CorrectedMoE(depth,rho).eval();assert not any(isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)) for m in model.modules())
                for probe,image in images:
                    a=input_amplitude(image)
                    with torch.no_grad():reference=model(a)[0]
                    calls=[];assets=[];brightness=[]
                    def replay(stage,amplitude,phase,ids):
                        calls.append(stage);gray=np.rint(np.clip(amplitude,0,1)*255).astype(np.uint8);decoded=[]
                        for kind,x in [('amplitude',gray[0]),('phase',phase)]:
                            path=out/f'L{depth}_rho{rho}_{probe}_{stage}_{kind}.bmp';native(x,kind).save(path);height=1080 if kind=='amplitude' else 1200
                            with Image.open(path) as im:
                                assert im.size==(1920,height);value=np.asarray(im.crop((452,(height-1016)//2,1468,(height+1016)//2)).resize((478,478),Image.Resampling.NEAREST));assert np.array_equal(value,x);decoded.append(value.copy())
                            assets.append(dict(path=path.name,sha256=sha(path)))
                        qa,qp=decoded;field=torch.from_numpy(qa.astype(np.float32)/255)[None]*torch.exp(2j*math.pi*torch.from_numpy(qp.astype(np.float32)/256))[None];field=torch.nn.functional.pad(field,(20,20,20,20))
                        brightness.append(dict(stage=stage,active_mean=float(qa.mean()/255),maximum=float(qa.max()/255),zero_fraction=float((qa==0).mean()),extra_peak_stretch=False))
                        return model.propagator(field).abs().square()[:,20:498,20:498].numpy()
                    with torch.no_grad():actual=model(a,callback=replay,ids=[probe])[0]
                    error=float((actual-reference).abs().max());assert error<=1e-6 and calls==['router']+[str(k) for k in range(depth)]
                    if probe=='black':assert all(v['maximum']==0 for v in brightness) and torch.allclose(actual,torch.full_like(actual,.05),atol=1e-8,rtol=0)
                    checks.append(dict(depth=depth,rho=rho,probe=probe,passed=True,probability_max_error=error,captures=len(calls),native_roundtrip_exact=True,brightness=brightness,assets=assets,contract=deployment_contract(depth,rho)))
        save(out/'checks.json',dict(passed=True,checks=checks,sources=sources(),model_forward_calls=36,test_arrays_read=0,test_evaluations=0,devices_opened=0,gpu_used=False,scope='Synthetic fresh parameters, not trained or physical calibration'))
    except BaseException as e:save(out/'failure.json',dict(error=repr(e),automatic_retry=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);check(p.parse_args().out)
